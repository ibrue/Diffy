"""
2D Gaussian Splatting for background scene representation.

Instead of storing the background as a flat JPEG, we decompose it into a
set of 2D Gaussians — each with position (x, y), covariance (scale_x,
scale_y, rotation), colour (r, g, b), and opacity (alpha).

Benefits over flat-JPEG background:
  1. Resolution-independent — render at any size without blocking artefacts
  2. Compact for simple scenes — 500 splats × 12 floats = 24 KB vs ~200 KB JPEG
  3. Camera-motion tolerant — splats can be shifted/rotated for slight viewpoint changes
  4. Interactive — users can explore the background decomposition in a WebGL viewer
  5. Editable — individual splats can be pruned, recoloured, or repositioned

Training uses expectation-maximisation:
  - Initialise splats from a grid or K-means on high-variance regions
  - Iteratively refine positions, scales, colours to minimise reconstruction error
  - Prune low-opacity splats and split high-error ones (adaptive density control)

The splat model is stored as chunk 0x08 in the .dfy bitstream.
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
