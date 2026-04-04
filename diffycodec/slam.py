"""
Lightweight visual SLAM for camera pose estimation.

Implements feature-based visual odometry in pure numpy (no OpenCV):
  1. Harris-like corner detection on grayscale
  2. BRIEF-inspired binary descriptors
  3. Hamming-distance matching with ratio test
  4. 8-point essential matrix estimation + RANSAC
  5. Pose recovery (R, t) with cheirality check
  6. Pose chain accumulation
  7. DLT point triangulation for sparse 3D map

The SLAM output feeds the 3D Gaussian Splatting module:
  - Camera poses let us render the 3D scene from each frame's viewpoint
  - The sparse point cloud initialises 3D Gaussian positions

Storage cost
------------
52 bytes/pose (4B idx + 36B R + 12B t).
30 fps × 8 hours = 864K poses × 52B ≈ 43 MB raw → ~2 MB zlib.
"""

import struct
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CameraPose:
    """Camera-to-world transform for a single frame."""
    R: np.ndarray          # 3×3 rotation (world ← camera)
    t: np.ndarray          # 3×1 translation (camera centre in world)
    frame_idx: int = 0


@dataclass
class SLAMResult:
    """Full SLAM output after processing all frames."""
    poses: List[CameraPose] = field(default_factory=list)
    point_cloud: Optional[np.ndarray] = None   # (M, 3) float32 xyz
    intrinsics: Optional[np.ndarray] = None    # 3×3 K matrix
    success: bool = False


# ─────────────────────────────────────────────────────────────────────
# Feature detection — Harris corners
# ─────────────────────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC to float32 HW grayscale."""
    if frame.ndim == 3:
        return (0.299 * frame[:, :, 0] +
                0.587 * frame[:, :, 1] +
                0.114 * frame[:, :, 2]).astype(np.float32)
    return frame.astype(np.float32)


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k2d = np.outer(k, k)
    return (k2d / k2d.sum()).astype(np.float32)


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution via numpy (no scipy needed)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (kh, kw))
        return np.einsum('ijkl,kl->ij', windows, kernel)
    except (ImportError, AttributeError):
        # Fallback for older numpy / Pyodide environments
        H, W = img.shape
        out = np.zeros_like(img)
        for dy in range(kh):
            for dx in range(kw):
                out += kernel[dy, dx] * padded[dy:dy + H, dx:dx + W]
        return out


def detect_features(gray: np.ndarray,
                    max_features: int = 500,
                    block_size: int = 3,
                    k: float = 0.04,
                    threshold_ratio: float = 0.01) -> np.ndarray:
    """
    Harris corner detector.

    Returns (N, 2) array of (x, y) corner coordinates.
    """
    # Sobel gradients
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = sx.T
    Ix = _convolve2d(gray, sx)
    Iy = _convolve2d(gray, sy)

    # Structure tensor components
    gauss = _gaussian_kernel(block_size, sigma=block_size / 3)
    Ixx = _convolve2d(Ix * Ix, gauss)
    Iyy = _convolve2d(Iy * Iy, gauss)
    Ixy = _convolve2d(Ix * Iy, gauss)

    # Harris response: det(M) - k * trace(M)^2
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    response = det - k * trace * trace

    # Threshold
    thresh = threshold_ratio * response.max()
    if thresh <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Non-maximum suppression: keep local maxima in 5×5 windows
    H, W = response.shape
    suppressed = np.zeros_like(response)
    pad = 2
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = np.roll(np.roll(response, dy, axis=0), dx, axis=1)
            suppressed = np.maximum(suppressed, shifted)

    mask = (response == suppressed) & (response > thresh)

    # Border exclusion
    border = 8
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Take top-N by response strength
    strengths = response[ys, xs]
    order = np.argsort(-strengths)[:max_features]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Feature description — BRIEF-like binary descriptors
# ─────────────────────────────────────────────────────────────────────

# Pre-computed sampling pattern (256 pairs, fixed seed)
_rng = np.random.RandomState(0xB01EF)
_BRIEF_PAIRS = _rng.randint(-12, 13, (256, 4)).astype(np.int32)  # (dy1,dx1,dy2,dx2)


def compute_descriptors(gray: np.ndarray,
                        keypoints: np.ndarray) -> np.ndarray:
    """
    Compute 256-bit binary descriptors for each keypoint.

    Returns (N, 32) uint8 array (256 bits packed into 32 bytes).
    """
    H, W = gray.shape
    N = len(keypoints)
    if N == 0:
        return np.zeros((0, 32), dtype=np.uint8)

    # Smooth image to reduce noise sensitivity
    gauss = _gaussian_kernel(5, 2.0)
    smooth = _convolve2d(gray, gauss)

    descs = np.zeros((N, 32), dtype=np.uint8)
    for i in range(N):
        cx, cy = int(keypoints[i, 0]), int(keypoints[i, 1])
        for j in range(256):
            dy1, dx1, dy2, dx2 = _BRIEF_PAIRS[j]
            y1 = min(max(cy + dy1, 0), H - 1)
            x1 = min(max(cx + dx1, 0), W - 1)
            y2 = min(max(cy + dy2, 0), H - 1)
            x2 = min(max(cx + dx2, 0), W - 1)
            if smooth[y1, x1] < smooth[y2, x2]:
                descs[i, j // 8] |= (1 << (j % 8))
    return descs


# ─────────────────────────────────────────────────────────────────────
# Feature matching
# ─────────────────────────────────────────────────────────────────────

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two binary descriptor vectors."""
    xor = np.bitwise_xor(a, b)
    # popcount per byte via lookup
    return sum(bin(byte).count('1') for byte in xor)


def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   kp1: np.ndarray, kp2: np.ndarray,
                   ratio_thresh: float = 0.75,
                   max_distance: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match descriptors using brute-force Hamming + ratio test.

    Returns (pts1, pts2) — matched point pairs, shape (M, 2) each.
    """
    N1, N2 = len(desc1), len(desc2)
    if N1 == 0 or N2 == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    # Vectorised hamming: expand to (N1, N2, 32), XOR, popcount
    # Pack as uint64 for fast popcount
    d1 = desc1.view(np.uint64).reshape(N1, -1)   # (N1, 4)
    d2 = desc2.view(np.uint64).reshape(N2, -1)   # (N2, 4)

    # Compute all pairwise distances
    # Use broadcasting: (N1, 1, 4) XOR (1, N2, 4)
    xor = d1[:, None, :] ^ d2[None, :, :]  # (N1, N2, 4)

    # Popcount via bit manipulation (Kernighan's method in numpy)
    # For uint64, use a lookup table approach
    # Simpler: convert to bytes and use lookup
    xor_bytes = xor.view(np.uint8).reshape(N1, N2, -1)  # (N1, N2, 32)
    # Pre-compute popcount table
    _popcount_table = np.zeros(256, dtype=np.int32)
    for i in range(256):
        _popcount_table[i] = bin(i).count('1')
    distances = _popcount_table[xor_bytes].sum(axis=2)  # (N1, N2)

    # For each point in desc1, find best and second-best in desc2
    pts1_list = []
    pts2_list = []

    for i in range(N1):
        sorted_idx = np.argsort(distances[i])
        best = distances[i, sorted_idx[0]]
        if best > max_distance:
            continue
        if N2 > 1:
            second = distances[i, sorted_idx[1]]
            if second > 0 and best / second > ratio_thresh:
                continue
        pts1_list.append(kp1[i])
        pts2_list.append(kp2[sorted_idx[0]])

    if not pts1_list:
        return np.zeros((0, 2)), np.zeros((0, 2))

    return np.array(pts1_list), np.array(pts2_list)


# ─────────────────────────────────────────────────────────────────────
# Essential matrix estimation
# ─────────────────────────────────────────────────────────────────────

def _normalise_points(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates to normalised camera coordinates."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    norm = np.empty_like(pts)
    norm[:, 0] = (pts[:, 0] - cx) / fx
    norm[:, 1] = (pts[:, 1] - cy) / fy
    return norm


def estimate_essential(pts1: np.ndarray, pts2: np.ndarray,
                       K: np.ndarray,
                       ransac_iters: int = 200,
                       ransac_thresh: float = 0.005) -> Optional[np.ndarray]:
    """
    Estimate essential matrix using 8-point algorithm + RANSAC.

    pts1, pts2: (N, 2) pixel coordinates.
    Returns 3×3 essential matrix or None if too few inliers.
    """
    if len(pts1) < 8:
        return None

    n1 = _normalise_points(pts1, K)
    n2 = _normalise_points(pts2, K)
    N = len(n1)

    best_E = None
    best_inliers = 0

    rng = np.random.RandomState(42)

    for _ in range(ransac_iters):
        # Sample 8 correspondences
        idx = rng.choice(N, 8, replace=False)
        p1, p2 = n1[idx], n2[idx]

        # Build constraint matrix A: each row is [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        A = np.zeros((8, 9))
        for j in range(8):
            x1, y1 = p1[j]
            x2, y2 = p2[j]
            A[j] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

        # Solve Ax = 0 via SVD
        _, _, Vt = np.linalg.svd(A)
        E = Vt[-1].reshape(3, 3)

        # Enforce rank-2 constraint
        U, S, Vt2 = np.linalg.svd(E)
        S = np.array([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
        E = U @ np.diag(S) @ Vt2

        # Count inliers: |p2.T @ E @ p1| < threshold
        ones1 = np.hstack([n1, np.ones((N, 1))])
        ones2 = np.hstack([n2, np.ones((N, 1))])
        errors = np.abs(np.sum(ones2 * (ones1 @ E.T), axis=1))
        inliers = np.sum(errors < ransac_thresh)

        if inliers > best_inliers:
            best_inliers = inliers
            best_E = E

    if best_inliers < 8:
        return None

    return best_E


# ─────────────────────────────────────────────────────────────────────
# Pose recovery from essential matrix
# ─────────────────────────────────────────────────────────────────────

def recover_pose(E: np.ndarray,
                 pts1: np.ndarray, pts2: np.ndarray,
                 K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose E into (R, t) with cheirality check.

    Returns (R, t) where R is 3×3, t is (3,).
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Four possible solutions
    candidates = [
        (U @ W @ Vt, U[:, 2]),
        (U @ W @ Vt, -U[:, 2]),
        (U @ W.T @ Vt, U[:, 2]),
        (U @ W.T @ Vt, -U[:, 2]),
    ]

    n1 = _normalise_points(pts1[:20], K)
    n2 = _normalise_points(pts2[:20], K)

    best_R, best_t, best_count = np.eye(3), np.zeros(3), -1

    for R, t in candidates:
        if np.linalg.det(R) < 0:
            R = -R
            t = -t

        # Cheirality: triangulated points should have positive depth in both cameras
        count = 0
        for j in range(min(len(n1), 10)):
            p1_h = np.array([n1[j, 0], n1[j, 1], 1.0])
            p2_h = np.array([n2[j, 0], n2[j, 1], 1.0])

            # Simple triangulation via DLT
            A_tri = np.zeros((4, 4))
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, t.reshape(3, 1)])
            A_tri[0] = p1_h[0] * P1[2] - P1[0]
            A_tri[1] = p1_h[1] * P1[2] - P1[1]
            A_tri[2] = p2_h[0] * P2[2] - P2[0]
            A_tri[3] = p2_h[1] * P2[2] - P2[1]

            _, _, Vt_tri = np.linalg.svd(A_tri)
            X = Vt_tri[-1]
            X = X[:3] / X[3]

            # Check positive depth in both cameras
            depth1 = X[2]
            X_cam2 = R @ X + t
            depth2 = X_cam2[2]
            if depth1 > 0 and depth2 > 0:
                count += 1

        if count > best_count:
            best_count = count
            best_R = R
            best_t = t

    # Normalise t to unit length
    t_norm = np.linalg.norm(best_t)
    if t_norm > 1e-10:
        best_t = best_t / t_norm

    return best_R, best_t


# ─────────────────────────────────────────────────────────────────────
# Point triangulation
# ─────────────────────────────────────────────────────────────────────

def triangulate_points(pts1: np.ndarray, pts2: np.ndarray,
                       K: np.ndarray,
                       R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from two-view correspondences.

    Returns (N, 3) world-frame 3D points.
    """
    n1 = _normalise_points(pts1, K)
    n2 = _normalise_points(pts2, K)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    N = len(n1)
    points_3d = []

    for i in range(N):
        p1_h = np.array([n1[i, 0], n1[i, 1], 1.0])
        p2_h = np.array([n2[i, 0], n2[i, 1], 1.0])

        A = np.zeros((4, 4))
        A[0] = p1_h[0] * P1[2] - P1[0]
        A[1] = p1_h[1] * P1[2] - P1[1]
        A[2] = p2_h[0] * P2[2] - P2[0]
        A[3] = p2_h[1] * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / (X[3] + 1e-12)

        # Skip points behind camera or too far
        if X[2] > 0 and X[2] < 100:
            points_3d.append(X)

    if not points_3d:
        return np.zeros((0, 3), dtype=np.float32)

    return np.array(points_3d, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# Visual SLAM pipeline
# ─────────────────────────────────────────────────────────────────────

class VisualSLAM:
    """
    Lightweight monocular visual SLAM.

    Parameters
    ----------
    K           : 3×3 camera intrinsics matrix
    width       : frame width
    height      : frame height
    max_features: max corners to detect per frame
    keyframe_interval: process every N-th frame as a keyframe for speed
    """

    def __init__(self,
                 K: Optional[np.ndarray] = None,
                 width: int = 1920,
                 height: int = 1080,
                 max_features: int = 300,
                 keyframe_interval: int = 5):
        if K is None:
            # Auto-estimate intrinsics from image size
            f = max(width, height) * 1.2
            K = np.array([[f, 0, width / 2],
                          [0, f, height / 2],
                          [0, 0, 1]], dtype=np.float64)
        self.K = K.astype(np.float64)
        self.width = width
        self.height = height
        self.max_features = max_features
        self.keyframe_interval = keyframe_interval

        self._poses: List[CameraPose] = []
        self._point_cloud_parts: List[np.ndarray] = []
        self._frame_idx = 0

        # Previous keyframe state
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp: Optional[np.ndarray] = None
        self._prev_desc: Optional[np.ndarray] = None

        # Cumulative world pose
        self._R_world = np.eye(3, dtype=np.float64)
        self._t_world = np.zeros(3, dtype=np.float64)

    def process_frame(self, frame: np.ndarray) -> CameraPose:
        """
        Process one frame. Returns camera pose (R, t) in world frame.

        For non-keyframes, interpolates from the last keyframe pose.
        """
        is_keyframe = (self._frame_idx % self.keyframe_interval == 0)

        if is_keyframe:
            # Downsample for speed on large images
            gray = _to_gray(frame)
            ds = max(1, min(gray.shape) // 256)
            if ds > 1:
                gray_ds = gray[::ds, ::ds]
            else:
                gray_ds = gray

            kp = detect_features(gray_ds, max_features=self.max_features)
            if ds > 1:
                kp = kp * ds  # scale back to original coordinates

            desc = compute_descriptors(gray if ds == 1 else gray, kp)

            if self._prev_gray is not None and len(kp) >= 8 and len(self._prev_kp) >= 8:
                # Match with previous keyframe
                pts_prev, pts_curr = match_features(
                    self._prev_desc, desc,
                    self._prev_kp, kp)

                if len(pts_prev) >= 8:
                    E = estimate_essential(pts_prev, pts_curr, self.K)
                    if E is not None:
                        R, t = recover_pose(E, pts_prev, pts_curr, self.K)

                        # Accumulate world pose
                        self._t_world = self._t_world + self._R_world @ t
                        self._R_world = self._R_world @ R

                        # Triangulate sparse points
                        pts_3d = triangulate_points(
                            pts_prev, pts_curr, self.K, R, t)
                        if len(pts_3d) > 0:
                            # Transform to world frame
                            pts_world = (self._R_world @ pts_3d.T).T + self._t_world
                            self._point_cloud_parts.append(pts_world)

            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc

        pose = CameraPose(
            R=self._R_world.copy(),
            t=self._t_world.copy(),
            frame_idx=self._frame_idx,
        )
        self._poses.append(pose)
        self._frame_idx += 1
        return pose

    def get_all_poses(self) -> List[CameraPose]:
        return self._poses

    def get_point_cloud(self) -> np.ndarray:
        """Return accumulated sparse 3D point cloud (N, 3)."""
        if not self._point_cloud_parts:
            return np.zeros((0, 3), dtype=np.float32)
        cloud = np.concatenate(self._point_cloud_parts, axis=0)
        # Remove outliers: clip to 3× median distance from centroid
        centroid = np.median(cloud, axis=0)
        dists = np.linalg.norm(cloud - centroid, axis=1)
        med_dist = np.median(dists)
        if med_dist > 0:
            mask = dists < 3 * med_dist
            cloud = cloud[mask]
        return cloud

    def get_result(self) -> SLAMResult:
        return SLAMResult(
            poses=self._poses,
            point_cloud=self.get_point_cloud(),
            intrinsics=self.K.copy(),
            success=len(self._poses) > 1 and len(self.get_point_cloud()) > 10,
        )


# ─────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────

def pack_camera_k(K: np.ndarray) -> bytes:
    """Serialise 3×3 intrinsics matrix to bytes."""
    return K.astype(np.float64).tobytes()


def unpack_camera_k(data: bytes) -> np.ndarray:
    """Deserialise intrinsics matrix."""
    return np.frombuffer(data, dtype=np.float64).reshape(3, 3).copy()


def pack_slam_poses(poses: List[CameraPose]) -> bytes:
    """
    Pack camera poses to bytes.

    Format per pose: [4B frame_idx][36B R (float32×9)][12B t (float32×3)] = 52 bytes.
    Header: [4B n_poses].
    """
    parts = [struct.pack(">I", len(poses))]
    for pose in poses:
        parts.append(struct.pack(">I", pose.frame_idx))
        parts.append(pose.R.astype(np.float32).tobytes())
        parts.append(pose.t.astype(np.float32).tobytes())
    return b"".join(parts)


def unpack_slam_poses(data: bytes) -> List[CameraPose]:
    """Deserialise packed camera poses."""
    n = struct.unpack_from(">I", data, 0)[0]
    offset = 4
    poses = []
    for _ in range(n):
        frame_idx = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        R = np.frombuffer(data[offset:offset + 36], dtype=np.float32).reshape(3, 3).copy()
        offset += 36
        t = np.frombuffer(data[offset:offset + 12], dtype=np.float32).reshape(3).copy()
        offset += 12
        poses.append(CameraPose(R=R, t=t, frame_idx=frame_idx))
    return poses
