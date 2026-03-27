"""
Background modelling for static factory environments.

Strategy
--------
We assume the factory floor is largely static.  A running median over the
first `warmup_frames` frames produces a clean background plate.  After that
we maintain a slow exponential update to handle gradual lighting changes
(shift transitions, clouds outside windows, etc.).

The background is stored **once** in the bitstream as a JPEG-compressed
keyframe.  All subsequent frames are encoded as foreground masks + sparse
residuals on top of this plate.

Foreground isolation uses a learned per-pixel variance threshold derived
from the warmup window, making it robust to flickering lights without
triggering false positives.
"""

import numpy as np
from typing import Optional, Tuple


class BackgroundModel:
    """
    Parameters
    ----------
    warmup_frames   : number of frames used to build the initial model
    update_alpha    : EMA coefficient for slow background drift (0 = frozen)
    fg_sigma_thresh : foreground = pixels deviating > N sigma from background
    abs_thresh      : absolute intensity fallback — a pixel is foreground if
                      any channel differs from bg_mean by more than this value,
                      regardless of sigma.  Catches objects that inflated bg_std
                      during warmup (e.g. a robot always in frame makes std high,
                      diluting the z-score so the robot is never detected).
    """

    def __init__(self,
                 warmup_frames: int = 300,
                 update_alpha: float = 0.002,
                 fg_sigma_thresh: float = 3.5,
                 abs_thresh: float = 25.0):
        self.warmup_frames   = warmup_frames
        self.update_alpha    = update_alpha
        self.fg_sigma_thresh = fg_sigma_thresh
        self.abs_thresh      = abs_thresh

        self._bg_mean:   Optional[np.ndarray] = None   # float32 H×W×C
        self._bg_std:    Optional[np.ndarray] = None   # float32 H×W×C
        self._bg_m2:     Optional[np.ndarray] = None   # Welford M2 accumulator
        self._frame_idx: int = 0
        self._warmed_up: bool = False

    def update(self, frame: np.ndarray) -> None:
        f = frame.astype(np.float32)
        if not self._warmed_up:
            # Online Welford accumulation — constant memory, in-place ops
            n = self._frame_idx + 1
            if self._bg_mean is None:
                self._bg_mean = f.copy()
                self._bg_m2   = np.zeros_like(f)
            else:
                # delta = f - mean (reuse _bg_m2 temp space via in-place)
                delta = np.subtract(f, self._bg_mean)
                self._bg_mean += delta / n
                # delta2 = f - updated_mean
                delta2 = np.subtract(f, self._bg_mean)
                self._bg_m2 += delta * delta2
                del delta, delta2
            if n >= self.warmup_frames:
                self._bg_std = self._bg_m2
                self._bg_std /= max(n - 1, 1)
                np.sqrt(self._bg_std, out=self._bg_std)
                np.clip(self._bg_std, 2.0, None, out=self._bg_std)
                self._bg_m2 = None
                self._warmed_up = True
        else:
            # EMA update — fully in-place to avoid temporaries
            fg_mask = self._foreground_mask(f)
            bg_px   = ~fg_mask
            alpha   = self.update_alpha
            # In-place: mean = mean + alpha * (f - mean) for bg pixels only
            for c in range(self._bg_mean.shape[2]):
                ch_mean = self._bg_mean[:, :, c]
                ch_f    = f[:, :, c]
                diff    = ch_f - ch_mean
                diff   *= alpha
                ch_mean[bg_px] += diff[bg_px]
        self._frame_idx += 1

    def get_background(self) -> np.ndarray:
        if self._bg_mean is None:
            raise RuntimeError("Background not yet estimated (need warmup frames)")
        return np.clip(self._bg_mean, 0, 255).astype(np.uint8)

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        if self._bg_mean is None:
            return np.ones(frame.shape[:2], dtype=bool)
        return self._foreground_mask(frame.astype(np.float32))

    def get_residual(self, frame: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self._bg_mean is None:
            raise RuntimeError("Background not yet estimated")
        residual = frame.astype(np.int16) - self._bg_mean.astype(np.int16)
        if mask is not None:
            residual[~mask] = 0
        return residual

    @property
    def is_ready(self) -> bool:
        return self._warmed_up

    def _foreground_mask(self, frame_f32: np.ndarray) -> np.ndarray:
        # Process one channel at a time to avoid allocating large (H,W,C) intermediates
        # that fragment the WASM heap in Pyodide (each H×W×C float32 array at 960p
        # is ~6 MB; allocating three of them at once caused OOM in practice).
        H, W = frame_f32.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        for c in range(3):
            diff_c = np.abs(frame_f32[:, :, c] - self._bg_mean[:, :, c])
            mask |= diff_c > self.abs_thresh
            mask |= (diff_c / self._bg_std[:, :, c]) > self.fg_sigma_thresh
        return mask


# ------------------------------------------------------------------
# Compression helpers
# ------------------------------------------------------------------

def encode_background_jpeg(bg: np.ndarray, quality: int = 80) -> bytes:
    """JPEG-encode a background plate to bytes (PIL, no cv2 needed)."""
    from PIL import Image
    import io
    img = Image.fromarray(bg.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def decode_background_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(data))
    return np.array(img)
