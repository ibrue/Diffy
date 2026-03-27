"""
Work-cycle detector for repetitive industrial tasks.

The key insight: a factory worker performing a pick-and-place task (or any
assembly operation) produces a near-periodic signal in:
  a) optical flow magnitude summed over the foreground region
  b) IMU angular velocity magnitude

We detect cycle boundaries by finding the minima of a smoothed version of
this energy signal – the instants when the worker returns to a "home"
posture between repetitions.

Once we have a cycle segmentation we can:
  • Choose the *median* cycle as the canonical reference.
  • Encode all other cycles as residuals vs. the canonical.
  • For cycles that differ by < epsilon (worker paused, same motion),
    store a 1-byte "clone" pointer instead of any pixel data.

Compression wins
----------------
If a worker does 500 identical cycles in 8 hours and each cycle is 10 s
long at 30 fps (300 frames), naive storage = 500 × 300 = 150,000 frames.
After cycle dedup we store 1 canonical (300 frames) + 499 residuals.
If the residual is 99% zeros, entropy coding compresses each residual cycle
to a few kilobytes.

Expected budget breakdown for 8 h / 10 MB target
-------------------------------------------------
  background keyframe           ~  200 KB   (1× JPEG)
  canonical cycle(s)            ~  500 KB   (1-5 cycles, JPEG sequence)
  residual stream (499 cycles)  ~  9.0 MB   (~18 KB/cycle average)
  IMU stream                    ~  0.3 MB   (f16 quaternions, zlib)
  ─────────────────────────────────────────────
  Total                         ~ 10.0 MB
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Cycle:
    start_frame: int
    end_frame: int            # exclusive
    energy: float             # mean foreground energy over cycle
    is_canonical: bool = False
    canonical_idx: Optional[int] = None   # which canonical this maps to


@dataclass
class CycleSegmentation:
    cycles: List[Cycle] = field(default_factory=list)
    canonical_indices: List[int] = field(default_factory=list)   # indices into cycles[]


class CycleDetector:
    """
    Online cycle detector.  Feed foreground energy values one frame at a time;
    call `flush()` at end of stream to get the final segmentation.

    Parameters
    ----------
    min_cycle_frames    : shortest plausible work cycle (default 3 s @ 30 fps = 90)
    max_cycle_frames    : longest plausible work cycle (default 120 s = 3600)
    smoothing_window    : Gaussian smoothing half-width for energy signal (frames)
    valley_threshold    : energy below this fraction of rolling max = cycle boundary
    canonical_max_count : keep at most N canonical prototypes (covers task diversity)
    similarity_threshold: residual MSE below this → "clone" of canonical, no pixel data
    """

    def __init__(self,
                 fps: float = 30.0,
                 min_cycle_frames: int = 90,
                 max_cycle_frames: int = 3600,
                 smoothing_window: int = 15,
                 valley_threshold: float = 0.25,
                 canonical_max_count: int = 5,
                 similarity_threshold: float = 50.0):
        self.fps                  = fps
        self.min_cycle_frames     = min_cycle_frames
        self.max_cycle_frames     = max_cycle_frames
        self.smoothing_window     = smoothing_window
        self.valley_threshold     = valley_threshold
        self.canonical_max_count  = canonical_max_count
        self.similarity_threshold = similarity_threshold

        self._energies: List[float] = []
        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    # Online feed
    # ------------------------------------------------------------------

    def push_energy(self, fg_energy: float) -> None:
        """
        Push foreground energy for one frame.
        fg_energy = mean absolute foreground residual (scalar float).
        """
        self._energies.append(float(fg_energy))
        self._frame_idx += 1

    # ------------------------------------------------------------------
    # Segmentation (call once, after all frames pushed)
    # ------------------------------------------------------------------

    def segment(self) -> CycleSegmentation:
        """Run full segmentation on buffered energy signal."""
        energies = np.array(self._energies, dtype=np.float32)
        boundaries = self._find_boundaries(energies)
        cycles = self._boundaries_to_cycles(boundaries, energies)
        self._assign_canonicals(cycles)
        seg = CycleSegmentation(cycles=cycles)
        seg.canonical_indices = [i for i, c in enumerate(cycles) if c.is_canonical]
        return seg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        """Gaussian-kernel smoothing with reflect padding."""
        w = self.smoothing_window
        if len(x) < 2 * w + 1:
            return x
        kernel = np.exp(-0.5 * (np.arange(-w, w + 1) / (w / 2)) ** 2)
        kernel /= kernel.sum()
        return np.convolve(x, kernel, mode="same")

    def _find_boundaries(self, energies: np.ndarray) -> List[int]:
        """Return frame indices that are local valleys (cycle boundaries)."""
        smoothed = self._smooth(energies)
        rolling_max = np.maximum.accumulate(smoothed)
        threshold   = rolling_max * self.valley_threshold

        boundaries = [0]
        last_b = 0

        for i in range(1, len(smoothed) - 1):
            gap = i - last_b
            if gap < self.min_cycle_frames:
                continue
            if gap > self.max_cycle_frames:
                # Force a cut – worker may have stopped
                boundaries.append(i)
                last_b = i
                continue
            # Valley condition: below threshold AND local minimum
            if (smoothed[i] < threshold[i] and
                    smoothed[i] <= smoothed[i - 1] and
                    smoothed[i] <= smoothed[i + 1]):
                boundaries.append(i)
                last_b = i

        boundaries.append(len(energies))
        return boundaries

    def _boundaries_to_cycles(self, boundaries: List[int],
                               energies: np.ndarray) -> List[Cycle]:
        cycles = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            energy = float(energies[start:end].mean()) if end > start else 0.0
            cycles.append(Cycle(start_frame=start, end_frame=end, energy=energy))
        return cycles

    def _assign_canonicals(self, cycles: List[Cycle]) -> None:
        """
        Pick representative canonical cycles using a greedy diversity strategy:
        start with the median-energy cycle, then add the cycle most dissimilar
        from existing canonicals (by energy proxy) until we hit canonical_max_count.
        """
        if not cycles:
            return

        energies = np.array([c.energy for c in cycles])
        lengths  = np.array([c.end_frame - c.start_frame for c in cycles])

        # Normalised feature = (energy, length) per cycle
        features = np.stack([energies / (energies.max() + 1e-6),
                              lengths  / (lengths.max()  + 1e-6)], axis=1)

        canonicals: List[int] = []

        # Seed: closest to median energy
        median_e = float(np.median(energies))
        seed = int(np.argmin(np.abs(energies - median_e)))
        canonicals.append(seed)

        # Expand by max-min distance
        while len(canonicals) < self.canonical_max_count and len(canonicals) < len(cycles):
            dists = np.min(
                [np.linalg.norm(features - features[c], axis=1) for c in canonicals],
                axis=0
            )
            dists[canonicals] = -1  # already selected
            nxt = int(np.argmax(dists))
            if dists[nxt] < 0.05:   # all remaining cycles are very similar
                break
            canonicals.append(nxt)

        for idx in canonicals:
            cycles[idx].is_canonical = True

        # Assign each non-canonical cycle to its closest canonical
        canon_features = features[canonicals]
        for i, cycle in enumerate(cycles):
            if cycle.is_canonical:
                cycle.canonical_idx = canonicals.index(i)
                continue
            dists = np.linalg.norm(canon_features - features[i], axis=1)
            best  = int(np.argmin(dists))
            cycle.canonical_idx = best


# ------------------------------------------------------------------
# Energy computation helper
# ------------------------------------------------------------------

def compute_fg_energy(residual: np.ndarray, fg_mask: np.ndarray) -> float:
    """
    Scalar energy = mean |residual| over foreground pixels.
    residual : int16 H×W×C
    fg_mask  : bool  H×W
    """
    if fg_mask.sum() == 0:
        return 0.0
    return float(np.abs(residual[fg_mask]).mean())
