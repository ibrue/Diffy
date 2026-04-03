"""
Gaussian Splatting for background scene representation (2D and 3D).

2D mode (chunk 0x08):
  Decomposes the background into positioned 2D Gaussians — each with
  (x, y), covariance (scale_x, scale_y, rotation), colour (r, g, b),
  and opacity (alpha).  Used as fallback when no SLAM data is available.

3D mode (chunk 0x09):
  Full 3D Gaussian Splatting — each splat has 3D position (xyz),
  3D scale, quaternion rotation, spherical-harmonics colour (DC + band1),
  and opacity.  Initialised from SLAM point cloud, optimised against
  multi-view keyframes.  Can be rendered from ANY camera pose, giving
  a view-correct background for each frame.

Benefits of 3D over 2D:
  1. True viewpoint independence — render correct background at any camera pose
  2. Handles parallax — near objects shift differently than far objects
  3. View-dependent colour via spherical harmonics
  4. Interactive 3D exploration in the WebGL viewer
"""

import struct
import numpy as np
from typing import Optional, Tuple


class GaussianSplat:
    """A single 2D Gaussian splat."""
    __slots__ = ('x', 'y', 'sx', 'sy', 'rot', 'r', 'g', 'b', 'opacity')

    def __init__(self, x: float, y: float, sx: float, sy: float,
                 rot: float, r: float, g: float, b: float, opacity: float):
        self.x = x
        self.y = y
        self.sx = sx       # scale x (std dev)
        self.sy = sy       # scale y (std dev)
        self.rot = rot     # rotation in radians
        self.r = r
        self.g = g
        self.b = b
        self.opacity = opacity


# Number of float32 values per splat in the serialised format
FLOATS_PER_SPLAT = 9  # x, y, sx, sy, rot, r, g, b, opacity


class GaussianSplatModel:
    """
    A collection of 2D Gaussians that approximate an image (the background).

    Parameters
    ----------
    n_splats    : initial number of splats
    iterations  : EM refinement iterations
    lr          : learning rate for gradient steps
    min_opacity : splats below this are pruned
    """

    def __init__(self,
                 n_splats: int = 2000,
                 iterations: int = 100,
                 lr: float = 0.01,
                 min_opacity: float = 0.01):
        self.n_splats = n_splats
        self.iterations = iterations
        self.lr = lr
        self.min_opacity = min_opacity

        # Splat parameters as contiguous arrays for vectorised ops
        # Shape: (N,) for each
        self.positions: Optional[np.ndarray] = None   # (N, 2) float32  [x, y]
        self.scales: Optional[np.ndarray] = None      # (N, 2) float32  [sx, sy]
        self.rotations: Optional[np.ndarray] = None   # (N,)   float32  radians
        self.colours: Optional[np.ndarray] = None     # (N, 3) float32  [r, g, b] 0-255
        self.opacities: Optional[np.ndarray] = None   # (N,)   float32  0-1

        self._width: int = 0
        self._height: int = 0

    def fit(self, background: np.ndarray) -> None:
        """
        Train splats to approximate the background image.

        Parameters
        ----------
        background : uint8 H x W x 3 RGB image
        """
        H, W = background.shape[:2]
        self._height, self._width = H, W
        bg = background.astype(np.float32)

        N = self.n_splats
        # Initialise on a grid with jitter
        grid_side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(0, W - 1, grid_side, dtype=np.float32)
        ys = np.linspace(0, H - 1, grid_side, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel()[:N]
        gy = gy.ravel()[:N]
        actual_n = len(gx)

        # Add small jitter
        rng = np.random.RandomState(42)
        gx += rng.uniform(-W / grid_side * 0.3, W / grid_side * 0.3, actual_n).astype(np.float32)
        gy += rng.uniform(-H / grid_side * 0.3, H / grid_side * 0.3, actual_n).astype(np.float32)
        gx = np.clip(gx, 0, W - 1)
        gy = np.clip(gy, 0, H - 1)

        self.positions = np.stack([gx, gy], axis=1)  # (N, 2)

        # Initial scales proportional to grid spacing
        cell_w = W / grid_side
        cell_h = H / grid_side
        self.scales = np.full((actual_n, 2), [cell_w * 0.6, cell_h * 0.6], dtype=np.float32)

        self.rotations = np.zeros(actual_n, dtype=np.float32)
        self.opacities = np.full(actual_n, 0.8, dtype=np.float32)

        # Sample initial colours from the background at splat positions
        ix = np.clip(gx.astype(int), 0, W - 1)
        iy = np.clip(gy.astype(int), 0, H - 1)
        self.colours = bg[iy, ix].copy()  # (N, 3)

        # EM-style optimisation via stochastic gradient descent
        # We render into a downsampled canvas for speed
        ds = max(1, min(H, W) // 128)  # downsample factor
        h_ds, w_ds = H // ds, W // ds
        target = bg[::ds, ::ds, :].copy()

        for it in range(self.iterations):
            rendered = self._render_low(w_ds, h_ds, ds)
            error = target - rendered  # (h, w, 3)

            # Decay learning rate
            lr = self.lr * (1.0 - 0.5 * it / self.iterations)

            # Update each splat based on the error at its position
            for i in range(actual_n):
                px = int(self.positions[i, 0] / ds)
                py = int(self.positions[i, 1] / ds)
                px = min(max(px, 0), w_ds - 1)
                py = min(max(py, 0), h_ds - 1)

                # Sample error in a small neighbourhood
                r_patch = max(1, int(self.scales[i, 0] / ds * 0.5))
                y0 = max(0, py - r_patch)
                y1 = min(h_ds, py + r_patch + 1)
                x0 = max(0, px - r_patch)
                x1 = min(w_ds, px + r_patch + 1)

                patch_err = error[y0:y1, x0:x1]
                if patch_err.size == 0:
                    continue

                mean_err = patch_err.mean(axis=(0, 1))  # (3,)

                # Update colour toward the error
                self.colours[i] += lr * 40.0 * mean_err * self.opacities[i]
                self.colours[i] = np.clip(self.colours[i], 0, 255)

                # Update opacity based on magnitude of local error
                err_mag = np.abs(mean_err).mean()
                if err_mag > 10:
                    self.opacities[i] = min(1.0, self.opacities[i] + lr * 0.1)
                elif err_mag < 2:
                    self.opacities[i] = max(0.0, self.opacities[i] - lr * 0.05)

                # Nudge position toward higher-error regions
                if patch_err.shape[0] > 1 and patch_err.shape[1] > 1:
                    err_y = np.abs(patch_err).mean(axis=2).mean(axis=1)
                    err_x = np.abs(patch_err).mean(axis=2).mean(axis=0)
                    cy = np.average(np.arange(len(err_y)), weights=err_y + 1e-6)
                    cx = np.average(np.arange(len(err_x)), weights=err_x + 1e-6)
                    dy = (cy - len(err_y) / 2) * ds * lr * 2.0
                    dx = (cx - len(err_x) / 2) * ds * lr * 2.0
                    self.positions[i, 0] = np.clip(self.positions[i, 0] + dx, 0, W - 1)
                    self.positions[i, 1] = np.clip(self.positions[i, 1] + dy, 0, H - 1)

            # Adaptive density: prune dead splats every 20 iterations
            if (it + 1) % 20 == 0:
                alive = self.opacities > self.min_opacity
                if alive.sum() < actual_n:
                    self.positions = self.positions[alive]
                    self.scales = self.scales[alive]
                    self.rotations = self.rotations[alive]
                    self.colours = self.colours[alive]
                    self.opacities = self.opacities[alive]
                    actual_n = len(self.opacities)

        self.n_splats = actual_n

    def _render_low(self, w: int, h: int, ds: int) -> np.ndarray:
        """Render splats into a downsampled canvas for training."""
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(len(self.opacities)):
            cx = self.positions[i, 0] / ds
            cy = self.positions[i, 1] / ds
            sx = max(self.scales[i, 0] / ds, 0.5)
            sy = max(self.scales[i, 1] / ds, 0.5)
            alpha = self.opacities[i]

            # Bounding box (3-sigma)
            r = 3
            x0 = max(0, int(cx - r * sx))
            x1 = min(w, int(cx + r * sx) + 1)
            y0 = max(0, int(cy - r * sy))
            y1 = min(h, int(cy + r * sy) + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx = (xx - cx) / sx
            dy = (yy - cy) / sy

            # Apply rotation
            rot = self.rotations[i]
            if abs(rot) > 0.01:
                c, s = np.cos(rot), np.sin(rot)
                dx2 = c * dx + s * dy
                dy2 = -s * dx + c * dy
                dx, dy = dx2, dy2

            gauss = np.exp(-0.5 * (dx * dx + dy * dy))
            w_splat = (gauss * alpha)[:, :, np.newaxis]

            canvas[y0:y1, x0:x1] += w_splat * self.colours[i]
            weight[y0:y1, x0:x1] += w_splat

        # Normalise
        mask = weight[:, :, 0] > 1e-6
        canvas[mask] /= weight[mask]

        return np.clip(canvas, 0, 255)

    def render(self, width: Optional[int] = None,
               height: Optional[int] = None) -> np.ndarray:
        """
        Render the splat model to an image.

        Returns uint8 H x W x 3 RGB.
        """
        w = width or self._width
        h = height or self._height

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        # Scale factor if rendering at different resolution
        sx_scale = w / self._width if self._width else 1.0
        sy_scale = h / self._height if self._height else 1.0

        for i in range(len(self.opacities)):
            cx = self.positions[i, 0] * sx_scale
            cy = self.positions[i, 1] * sy_scale
            sx = max(self.scales[i, 0] * sx_scale, 0.5)
            sy = max(self.scales[i, 1] * sy_scale, 0.5)
            alpha = self.opacities[i]

            r = 3
            x0 = max(0, int(cx - r * sx))
            x1 = min(w, int(cx + r * sx) + 1)
            y0 = max(0, int(cy - r * sy))
            y1 = min(h, int(cy + r * sy) + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx = (xx - cx) / sx
            dy = (yy - cy) / sy

            rot = self.rotations[i]
            if abs(rot) > 0.01:
                c, s = np.cos(rot), np.sin(rot)
                dx2 = c * dx + s * dy
                dy2 = -s * dx + c * dy
                dx, dy = dx2, dy2

            gauss = np.exp(-0.5 * (dx * dx + dy * dy))
            w_splat = (gauss * alpha)[:, :, np.newaxis]

            canvas[y0:y1, x0:x1] += w_splat * self.colours[i]
            weight[y0:y1, x0:x1] += w_splat

        mask = weight[:, :, 0] > 1e-6
        canvas[mask] /= weight[mask]

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def psnr(self, reference: np.ndarray) -> float:
        """Compute PSNR of rendered splats vs reference image."""
        rendered = self.render(reference.shape[1], reference.shape[0])
        mse = np.mean((rendered.astype(np.float32) - reference.astype(np.float32)) ** 2)
        if mse < 1e-10:
            return 99.0
        return float(10 * np.log10(255.0 ** 2 / mse))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """
        Serialise splat model to bytes for .dfy storage.

        Format:
          [2B u16] width
          [2B u16] height
          [4B u32] n_splats
          [N * 9 * 4B] float32 array: x, y, sx, sy, rot, r, g, b, opacity
        """
        N = len(self.opacities)
        header = struct.pack(">HHI", self._width, self._height, N)

        # Pack all splats into a contiguous float32 array
        data = np.empty((N, FLOATS_PER_SPLAT), dtype=np.float32)
        data[:, 0:2] = self.positions
        data[:, 2:4] = self.scales
        data[:, 4] = self.rotations
        data[:, 5:8] = self.colours
        data[:, 8] = self.opacities

        return header + data.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "GaussianSplatModel":
        """Deserialise a splat model from bytes."""
        width, height, n = struct.unpack_from(">HHI", data, 0)
        offset = 8

        model = cls(n_splats=n)
        model._width = width
        model._height = height

        arr = np.frombuffer(data[offset:offset + n * FLOATS_PER_SPLAT * 4],
                            dtype=np.float32).reshape(n, FLOATS_PER_SPLAT)

        model.positions = arr[:, 0:2].copy()
        model.scales = arr[:, 2:4].copy()
        model.rotations = arr[:, 4].copy()
        model.colours = arr[:, 5:8].copy()
        model.opacities = arr[:, 8].copy()
        model.n_splats = n

        return model

    def to_json(self) -> dict:
        """Export splats as JSON-friendly dict for the WebGL viewer."""
        N = len(self.opacities)
        return {
            "width": self._width,
            "height": self._height,
            "count": N,
            "positions": self.positions.tolist(),
            "scales": self.scales.tolist(),
            "rotations": self.rotations.tolist(),
            "colours": self.colours.tolist(),
            "opacities": self.opacities.tolist(),
        }


# ------------------------------------------------------------------
# High-level helpers
# ------------------------------------------------------------------

def fit_splat_model(background: np.ndarray,
                    quality: int = 50) -> "GaussianSplatModel":
    """
    Fit a splat model to a background image.  Splat count and iteration
    budget scale automatically with image area and quality.

    quality 1-100:
      10  -> fewer splats, fewer iterations (faster, coarser)
      50  -> balanced (default)
      100 -> more splats, more iterations (slower, sharper)

    Splat count: ~1 splat per 200 pixels, capped at 2000.
    """
    H, W = background.shape[:2]
    n_splats  = max(50, min(2000, H * W // 200))
    # Scale iteration count with quality; clamp so small-image tests stay fast
    iterations = max(20, min(200, int(40 * (quality / 50.0))))
    model = GaussianSplatModel(n_splats=n_splats, iterations=iterations)
    model.fit(background)
    return model


# ══════════════════════════════════════════════════════════════════════
# 3D Gaussian Splatting
# ══════════════════════════════════════════════════════════════════════

FLOATS_PER_SPLAT_3D = 23  # xyz(3) + scale(3) + quat(4) + sh_color(12) + opacity(1)


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] → 3×3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _sh_eval(sh_coeffs: np.ndarray, view_dir: np.ndarray) -> np.ndarray:
    """
    Evaluate degree-1 spherical harmonics for RGB colour.

    sh_coeffs: (12,) — [r_dc, r_1, r_2, r_3, g_dc, g_1, g_2, g_3, b_dc, b_1, b_2, b_3]
    view_dir: (3,) — unit direction from splat to camera

    Returns (3,) RGB in [0, 255].
    """
    C0 = 0.28209479  # 1 / (2*sqrt(pi))
    C1 = 0.48860251  # sqrt(3) / (2*sqrt(pi))

    x, y, z = view_dir
    rgb = np.zeros(3, dtype=np.float64)
    for c in range(3):
        base = c * 4
        val = C0 * sh_coeffs[base]         # DC
        val += C1 * y * sh_coeffs[base + 1]  # Y_1^-1
        val += C1 * z * sh_coeffs[base + 2]  # Y_1^0
        val += C1 * x * sh_coeffs[base + 3]  # Y_1^1
        rgb[c] = val
    return np.clip(rgb * 255.0, 0, 255)


class GaussianSplatModel3D:
    """
    3D Gaussian Splatting scene model.

    Parameters per splat (23 float32):
      position:   (N, 3)  xyz world coordinates
      scales:     (N, 3)  sx, sy, sz (in log-space)
      rotations:  (N, 4)  quaternion [x, y, z, w]
      sh_colors:  (N, 12) spherical harmonics coefficients (DC + band-1, per RGB)
      opacities:  (N,)    sigmoid-space opacity
    """

    def __init__(self, n_splats: int = 5000, iterations: int = 50,
                 lr: float = 0.01):
        self.n_splats = n_splats
        self.iterations = iterations
        self.lr = lr

        self.positions:  Optional[np.ndarray] = None   # (N, 3)
        self.scales:     Optional[np.ndarray] = None   # (N, 3)
        self.rotations:  Optional[np.ndarray] = None   # (N, 4)
        self.sh_colors:  Optional[np.ndarray] = None   # (N, 12)
        self.opacities:  Optional[np.ndarray] = None   # (N,)

        self._width: int = 0
        self._height: int = 0
        self._K: Optional[np.ndarray] = None

    def fit(self, point_cloud: np.ndarray,
            images: list, poses: list,
            K: np.ndarray,
            width: int, height: int) -> None:
        """
        Fit 3D Gaussians from a SLAM point cloud and multi-view images.

        Parameters
        ----------
        point_cloud : (M, 3) float32 world-frame 3D points
        images      : list of uint8 HxWx3 keyframe images
        poses       : list of CameraPose (R, t for each image)
        K           : 3×3 intrinsics
        width, height : target resolution
        """
        self._width = width
        self._height = height
        self._K = K.astype(np.float64)

        M = len(point_cloud)
        N = min(self.n_splats, max(M, 50))

        if M == 0:
            # No point cloud — initialise random splats in front of camera
            self._init_random(N, poses, K)
        elif M <= N:
            # Use all points, pad with jittered duplicates
            self.positions = point_cloud[:N].astype(np.float32).copy()
            if M < N:
                extra = N - M
                idx = np.random.randint(0, M, extra)
                jitter = np.random.randn(extra, 3).astype(np.float32) * 0.05
                self.positions = np.vstack([
                    self.positions, point_cloud[idx] + jitter
                ])
        else:
            # Subsample
            idx = np.random.choice(M, N, replace=False)
            self.positions = point_cloud[idx].astype(np.float32).copy()

        N = len(self.positions)
        self.n_splats = N

        # Initialise scales (log-space): set to ~distance to nearest neighbour
        if N > 1:
            # Sample a subset for NN distance estimation
            sample_n = min(N, 200)
            sample_idx = np.random.choice(N, sample_n, replace=False)
            sample = self.positions[sample_idx]
            dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
            np.fill_diagonal(dists, 1e10)
            nn_dist = np.median(np.min(dists, axis=1))
            init_scale = max(nn_dist * 0.5, 0.01)
        else:
            init_scale = 0.1

        self.scales = np.full((N, 3), np.log(init_scale), dtype=np.float32)
        self.rotations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (N, 1))
        self.opacities = np.full(N, 0.8, dtype=np.float32)

        # Initialise SH colours from nearest image observation
        self.sh_colors = np.zeros((N, 12), dtype=np.float32)
        if images and poses:
            self._init_colors_from_images(images, poses, K)

        # Optimise against keyframe images
        if images and poses:
            self._optimize(images, poses, K, width, height)

    def _init_random(self, N: int, poses: list, K: np.ndarray) -> None:
        """Initialise splats in a random volume visible from the cameras."""
        if poses:
            centers = np.array([p.t for p in poses])
            mean_pos = centers.mean(axis=0) if len(centers) > 0 else np.zeros(3)
            spread = max(np.std(centers) * 3, 1.0) if len(centers) > 1 else 1.0
        else:
            mean_pos = np.array([0, 0, 2.0])
            spread = 2.0

        self.positions = (mean_pos + np.random.randn(N, 3).astype(np.float32) * spread)

    def _init_colors_from_images(self, images: list, poses: list,
                                 K: np.ndarray) -> None:
        """Set DC SH coefficient from projected colour in nearest keyframe."""
        C0 = 0.28209479
        if not images or not poses:
            return

        # Use first image as reference
        img = images[0].astype(np.float32)
        pose = poses[0]
        H, W = img.shape[:2]

        for i in range(len(self.positions)):
            # Project to image
            p_cam = pose.R.T @ (self.positions[i].astype(np.float64) - pose.t)
            if p_cam[2] <= 0.01:
                continue
            px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < W and 0 <= iy < H:
                rgb = img[iy, ix] / 255.0 / C0
                self.sh_colors[i, 0] = rgb[0]   # r DC
                self.sh_colors[i, 4] = rgb[1]   # g DC
                self.sh_colors[i, 8] = rgb[2]   # b DC

    def _optimize(self, images: list, poses: list,
                  K: np.ndarray, width: int, height: int) -> None:
        """EM-style optimisation of splat parameters against keyframe images."""
        # Use a subset of keyframes for speed
        n_views = min(len(images), 5)
        step = max(1, len(images) // n_views)
        sel_images = images[::step][:n_views]
        sel_poses = poses[::step][:n_views]

        # Downsample for speed
        ds = max(1, min(height, width) // 64)
        K_ds = K.copy()
        K_ds[0, :] /= ds
        K_ds[1, :] /= ds
        h_ds, w_ds = height // ds, width // ds

        for it in range(self.iterations):
            lr = self.lr * (1.0 - 0.5 * it / self.iterations)

            for view_idx in range(len(sel_images)):
                target = sel_images[view_idx][::ds, ::ds].astype(np.float32)
                pose = sel_poses[view_idx]

                rendered = self._render_internal(pose, K_ds, w_ds, h_ds)
                error = target - rendered  # (h, w, 3)

                # Update colours based on per-splat error
                for i in range(self.n_splats):
                    p_cam = pose.R.T @ (self.positions[i].astype(np.float64) - pose.t)
                    if p_cam[2] <= 0.01:
                        continue
                    px = int(K_ds[0, 0] * p_cam[0] / p_cam[2] + K_ds[0, 2])
                    py = int(K_ds[1, 1] * p_cam[1] / p_cam[2] + K_ds[1, 2])

                    if 0 <= px < w_ds and 0 <= py < h_ds:
                        r = max(1, int(np.exp(self.scales[i].mean()) / ds))
                        y0 = max(0, py - r)
                        y1 = min(h_ds, py + r + 1)
                        x0 = max(0, px - r)
                        x1 = min(w_ds, px + r + 1)
                        patch = error[y0:y1, x0:x1]
                        if patch.size == 0:
                            continue
                        mean_err = patch.mean(axis=(0, 1))

                        C0 = 0.28209479
                        self.sh_colors[i, 0] += lr * mean_err[0] / (255.0 * C0 + 1e-6)
                        self.sh_colors[i, 4] += lr * mean_err[1] / (255.0 * C0 + 1e-6)
                        self.sh_colors[i, 8] += lr * mean_err[2] / (255.0 * C0 + 1e-6)

    def _render_internal(self, pose, K, w, h) -> np.ndarray:
        """Render splats from a camera pose (internal, float32 output)."""
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(self.n_splats):
            # Transform to camera frame
            p_world = self.positions[i].astype(np.float64)
            p_cam = pose.R.T @ (p_world - pose.t)

            if p_cam[2] <= 0.01:
                continue

            # Project centre
            px = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            py = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]

            # Compute screen-space scale (approximate)
            s = np.exp(self.scales[i]).astype(np.float64)
            depth = p_cam[2]
            sx_screen = max(K[0, 0] * s[0] / depth, 0.5)
            sy_screen = max(K[1, 1] * s[1] / depth, 0.5)

            alpha = float(1.0 / (1.0 + np.exp(-self.opacities[i])))  # sigmoid

            # Bounding box
            r = 2.5
            x0 = max(0, int(px - r * sx_screen))
            x1 = min(w, int(px + r * sx_screen) + 1)
            y0 = max(0, int(py - r * sy_screen))
            y1 = min(h, int(py + r * sy_screen) + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            dx = (xx - px) / sx_screen
            dy = (yy - py) / sy_screen
            gauss = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)

            # View-dependent colour via SH
            view_dir = p_world - pose.t
            view_norm = np.linalg.norm(view_dir)
            if view_norm > 1e-8:
                view_dir = view_dir / view_norm
            else:
                view_dir = np.array([0, 0, 1.0])

            rgb = _sh_eval(self.sh_colors[i], view_dir)

            w_splat = (gauss * alpha)[:, :, np.newaxis]
            canvas[y0:y1, x0:x1] += w_splat * rgb.astype(np.float32)
            weight[y0:y1, x0:x1] += w_splat

        mask = weight[:, :, 0] > 1e-6
        canvas[mask] /= weight[mask]
        return np.clip(canvas, 0, 255)

    def render(self, pose, K: Optional[np.ndarray] = None,
               width: Optional[int] = None,
               height: Optional[int] = None) -> np.ndarray:
        """
        Render the 3D scene from a given camera pose.

        Parameters
        ----------
        pose   : CameraPose (R, t) or compatible object with .R and .t
        K      : 3×3 intrinsics (defaults to stored K)
        width  : output width
        height : output height

        Returns uint8 H×W×3 RGB.
        """
        w = width or self._width
        h = height or self._height
        k = K if K is not None else self._K
        if k is None:
            raise ValueError("No camera intrinsics K available")

        rendered = self._render_internal(pose, k, w, h)
        return np.clip(rendered, 0, 255).astype(np.uint8)

    def psnr(self, reference: np.ndarray, pose, K=None) -> float:
        """PSNR of rendered scene vs reference image from a given pose."""
        rendered = self.render(pose, K, reference.shape[1], reference.shape[0])
        mse = np.mean((rendered.astype(np.float32) - reference.astype(np.float32)) ** 2)
        if mse < 1e-10:
            return 99.0
        return float(10 * np.log10(255.0 ** 2 / mse))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """
        Serialise 3D splat model.

        Format:
          [2B u16] width
          [2B u16] height
          [4B u32] n_splats
          [72B]    K matrix (float64 × 9)
          [N * 23 * 4B] float32 array per splat
        """
        N = self.n_splats
        header = struct.pack(">HHI", self._width, self._height, N)

        K_bytes = (self._K if self._K is not None
                   else np.eye(3, dtype=np.float64)).astype(np.float64).tobytes()

        data = np.empty((N, FLOATS_PER_SPLAT_3D), dtype=np.float32)
        data[:, 0:3] = self.positions
        data[:, 3:6] = self.scales
        data[:, 6:10] = self.rotations
        data[:, 10:22] = self.sh_colors
        data[:, 22] = self.opacities

        return header + K_bytes + data.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "GaussianSplatModel3D":
        """Deserialise a 3D splat model."""
        width, height, n = struct.unpack_from(">HHI", data, 0)
        offset = 8

        K = np.frombuffer(data[offset:offset + 72], dtype=np.float64).reshape(3, 3).copy()
        offset += 72

        model = cls(n_splats=n)
        model._width = width
        model._height = height
        model._K = K

        arr = np.frombuffer(data[offset:offset + n * FLOATS_PER_SPLAT_3D * 4],
                            dtype=np.float32).reshape(n, FLOATS_PER_SPLAT_3D)

        model.positions = arr[:, 0:3].copy()
        model.scales = arr[:, 3:6].copy()
        model.rotations = arr[:, 6:10].copy()
        model.sh_colors = arr[:, 10:22].copy()
        model.opacities = arr[:, 22].copy()
        model.n_splats = n

        return model

    def to_json(self) -> dict:
        """Export 3D splats as JSON for the WebGL viewer."""
        return {
            "version": "3d",
            "width": self._width,
            "height": self._height,
            "count": self.n_splats,
            "K": self._K.tolist() if self._K is not None else None,
            "positions": self.positions.tolist(),
            "scales": self.scales.tolist(),
            "rotations": self.rotations.tolist(),
            "sh_colors": self.sh_colors.tolist(),
            "opacities": self.opacities.tolist(),
        }


def fit_splat_model_3d(point_cloud: np.ndarray,
                       images: list, poses: list,
                       K: np.ndarray,
                       width: int, height: int,
                       quality: int = 50) -> GaussianSplatModel3D:
    """
    Fit a 3D Gaussian Splat model from SLAM output.

    quality 1-100 controls splat count and iteration budget.
    """
    n_splats = max(100, min(5000, len(point_cloud) * 2))
    n_splats = max(100, int(n_splats * (quality / 50.0)))
    iterations = max(10, min(100, int(30 * (quality / 50.0))))

    model = GaussianSplatModel3D(n_splats=n_splats, iterations=iterations)
    model.fit(point_cloud, images, poses, K, width, height)
    return model
