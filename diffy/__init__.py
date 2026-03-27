"""
Diffy - the difference video encoder.

Public API:
    DiffyEncoder  - encode frames into a .dfy file
    DiffyDecoder  - decode a .dfy file back to frames
"""

from egocodec.encoder import EgoEncoder as DiffyEncoder
from egocodec.decoder import EgoDecoder as DiffyDecoder

__all__ = ["DiffyEncoder", "DiffyDecoder"]
__version__ = "0.1.0"
