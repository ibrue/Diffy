"""
EgoCodec: A difference video codec for industrial physical AI.

Design priors that H.265 ignores but we exploit:
  1. Background stationarity  - factory floor doesn't move; encode it once.
  2. Cycle periodicity        - workers repeat the same motion thousands of times/shift.
  3. Head-motion predictability - IMU-aided warp removes camera shake before coding.
  4. Sparse foreground        - only the worker's hands + held objects change meaningfully.

Target: 8 hours of 1080p30 → ≤ 10 MB  (~1000× better than H.265).
"""

from .encoder import DiffyEncoder
from .decoder import DiffyDecoder
from .bitstream import BitstreamWriter, BitstreamReader

__all__ = ["DiffyEncoder", "DiffyDecoder", "BitstreamWriter", "BitstreamReader"]
__version__ = "0.1.0"
