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
    """

    def __init__(self,
                 warmup_frames: int = 300,
                 update_alpha: float = 0.002,
                 fg_sigma_thresh: float = 3.5):
        self.warmup_frames   = warmup_frames
        self.update_alpha    = update_alpha
        self.fg_sigma_thresh = fg_sigma_thresh

        self._buffer: list[np.ndarray] = []
        self._bg_mean:   Optional[np.ndarray] = None   # float32 H×W×C
        self._bg_std:    Optional[np.ndarray] = None   # float32 H×W×C
        self._frame_idx: int = 0
        self._warmed_up: bool = False

    def update(self, frame: np.ndarray) -> None:
        f = frame.astype(np.float32)
        if not self._warmed_up:
            self._buffer.append(f)
            if len(self._buffer) >= self.warmup_frames:
                self._fit_from_buffer()
        else:
            fg_mask = self._foreground_mask(f)
            bg_px   = ~fg_mask[:, :, None]
            alpha   = self.update_alpha
            self._bg_mean = np.where(bg_px,
                                     (1 - alpha) * self._bg_mean + alpha * f,
                                     self._bg_mean)
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

    def _fit_from_buffer(self) -> None:
        stack = np.stack(self._buffer, axis=0)
        self._bg_mean = np.median(stack, axis=0)
        self._bg_std  = np.std(stack, axis=0).clip(min=2.0)
        self._warmed_up = True
        self._buffer.clear()

    def _foreground_mask(self, frame_f32: np.ndarray) -> np.ndarray:
        diff = np.abs(frame_f32 - self._bg_mean)
        z    = diff / self._bg_std
        return z.max(axis=2) > self.fg_sigma_thresh


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
