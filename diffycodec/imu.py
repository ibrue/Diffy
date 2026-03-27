"""
IMU integration for head-motion stabilisation.

The camera is worn on a worker's head.  Between frames the head rotates by a
quaternion dq estimated from the gyroscope.  If we warp each frame to undo
this rotation before encoding, the inter-frame residual drops dramatically –
the background truly appears static even during head-turns.

Stabilisation pipeline
----------------------
1. Integrate gyroscope readings → absolute orientation quaternion Q_t.
2. Choose a reference orientation Q_ref (e.g. the first frame of each cycle).
3. For each frame compute the relative rotation dQ = Q_ref^{-1} ⊗ Q_t.
4. Map dQ to a homographic warp H (valid for small rotations or known focal
   length) and apply cv2.warpPerspective to the frame.
5. The warped frame is what the background model and cycle encoder receive.

The raw IMU stream is stored separately (f16 quaternion × N) and used by the
decoder to reverse the warp at playback time.

IMU storage cost
----------------
6-DOF at 200 Hz for 8 hours:
  200 Hz × 28800 s × (4 floats × 2 bytes f16) = 46 MB raw.
After zlib compression (quaternions are very smooth): ~0.3 MB.
"""

import numpy as np
import struct
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Quaternion helpers (scalar-last convention: [x, y, z, w])
# --------------------------------------------------------------------------

def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions [x,y,z,w]."""
    x0, y0, z0, w0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x1, y1, z1, w1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
    ], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions)."""
    c = q.copy()
    c[..., :3] *= -1
    return c


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x,y,z,w] to 3×3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def rotation_to_homography(R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Plane-at-infinity homography for pure rotation:  H = K · R · K^{-1}
    Valid for scenes at infinity (background), which is exactly what we have.
    """
    return K @ R @ np.linalg.inv(K)


# --------------------------------------------------------------------------
# IMU integrator
# --------------------------------------------------------------------------

class IMUIntegrator:
    """
    Integrates gyroscope measurements to absolute orientation quaternions.

    Parameters
    ----------
    imu_hz          : IMU sample rate (Hz)
    camera_hz       : camera frame rate (fps)
    gyro_noise_std  : gyroscope noise standard deviation (rad/s)
    """

    def __init__(self, imu_hz: float = 200.0, camera_hz: float = 30.0,
                 gyro_noise_std: float = 0.003):
        self.imu_hz         = imu_hz
        self.camera_hz      = camera_hz
        self.gyro_noise_std = gyro_noise_std
        self._q  = np.array([0.0, 0.0, 0.0, 1.0])   # identity orientation
        self._quats_at_frames: List[np.ndarray] = []
        self._imu_since_frame: int = 0
        self._samples_per_frame = imu_hz / camera_hz

    def push_gyro(self, gyro_xyz: np.ndarray, dt: Optional[float] = None) -> None:
        """
        gyro_xyz : angular velocity vector in rad/s  (shape [3])
        dt       : time since last sample (default 1/imu_hz)
        """
        if dt is None:
            dt = 1.0 / self.imu_hz
        # Integrate: small-angle quaternion integration
        angle = np.linalg.norm(gyro_xyz) * dt
        if angle > 1e-10:
            axis = gyro_xyz / np.linalg.norm(gyro_xyz)
            dq   = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])
            self._q = quat_mul(self._q, dq)
            self._q /= np.linalg.norm(self._q)   # renormalize

        self._imu_since_frame += 1
        if self._imu_since_frame >= self._samples_per_frame:
            self._quats_at_frames.append(self._q.copy())
            self._imu_since_frame -= self._samples_per_frame

    def get_frame_orientations(self) -> np.ndarray:
        """Return N×4 array of quaternions, one per camera frame."""
        return np.array(self._quats_at_frames)

    def reset_reference(self) -> np.ndarray:
        """Call at cycle start; returns the reference quaternion."""
        return self._q.copy()


# --------------------------------------------------------------------------
# Frame stabiliser
# --------------------------------------------------------------------------

class FrameStabilizer:
    """
    Applies the IMU-derived homography warp to de-rotate each frame.
    The result has a static background regardless of head rotation.

    Parameters
    ----------
    K : 3×3 camera intrinsics matrix
        If None, a reasonable default for a 1080p 90-degree-HFOV camera is used.
    """

    def __init__(self, K: Optional[np.ndarray] = None,
                 width: int = 1920, height: int = 1080):
        self.width  = width
        self.height = height
        if K is None:
            # Estimate from typical egocentric camera (90° HFOV)
            fx = width / (2 * np.tan(np.radians(45)))
            self.K = np.array([[fx,  0, width  / 2],
                                [ 0, fx, height / 2],
                                [ 0,  0,           1]], dtype=np.float64)
        else:
            self.K = K.astype(np.float64)
        self._q_ref: Optional[np.ndarray] = None

    def set_reference(self, q_ref: np.ndarray) -> None:
        self._q_ref = q_ref.copy()

    def warp_frame(self, frame: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """
        Apply stabilisation warp.  Returns a warped frame of the same size.
        frame : uint8 H×W×C
        q_current : [x,y,z,w] orientation at this frame
        """
        import cv2
        if self._q_ref is None:
            self._q_ref = q_current.copy()
        dq  = quat_mul(quat_conjugate(self._q_ref), q_current)
        R   = quat_to_rotation_matrix(dq)
        H   = rotation_to_homography(R, self.K)
        return cv2.warpPerspective(frame, H, (self.width, self.height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# IMU bitstream packing
# --------------------------------------------------------------------------

def pack_imu_quats(quats: np.ndarray) -> bytes:
    """
    Pack N×4 float32 quaternions to f16 bytes.
    Quaternion components are in [-1, 1]; f16 gives ~0.001 rad precision.
    """
    return quats.astype(np.float16).tobytes()


def unpack_imu_quats(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.float16).reshape(-1, 4)
    return arr.astype(np.float32)
