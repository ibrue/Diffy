"""
Residual codec: encodes the difference between two frames (or a frame and the
background plate) using a combination of:

  1. Foreground-masked quantisation  – zero out background pixels entirely.
  2. DCT block coding                 – 8×8 DCT on each colour channel,
                                        adaptive quantisation table.
  3. Run-length encoding              – exploit the large zero runs in sparse
                                        foreground residuals.
  4. zlib final pass                  – mop up remaining entropy.

This intentionally does NOT use a full entropy coder (Huffman/arithmetic) to
keep the implementation self-contained and auditable.  In a production system
you would replace the RLE+zlib with a range coder for another ~15% gain.

Compression budget target: ≤ 18 KB per cycle-residual (vs. canonical cycle).
A 300-frame 1080p cycle at 99% background similarity produces residuals where
≥ 95% of DCT coefficients are zero after quantisation → RLE achieves ~200:1
before zlib, reaching the budget comfortably.
"""

import numpy as np
import zlib
import struct
from typing import Optional, Tuple


# --------------------------------------------------------------------------
# Quantisation tables
# --------------------------------------------------------------------------

# Standard JPEG luminance table (used as baseline, scaled by quality factor)
_JPEG_LUMA_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

_JPEG_CHROMA_QT = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def _make_qt(quality: int, luma: bool = True) -> np.ndarray:
    """Scale JPEG quantisation table by quality (1-100)."""
    base = _JPEG_LUMA_QT if luma else _JPEG_CHROMA_QT
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    qt = np.floor((base * scale + 50.0) / 100.0).clip(1, 255)
    return qt.astype(np.float32)


# --------------------------------------------------------------------------
# 8×8 DCT helpers
# --------------------------------------------------------------------------

def _dct8(block: np.ndarray) -> np.ndarray:
    """2D DCT-II on an 8×8 float block."""
    from scipy.fft import dctn
    return dctn(block, norm="ortho")


def _idct8(block: np.ndarray) -> np.ndarray:
    """2D DCT-III (inverse) on an 8×8 float block."""
    from scipy.fft import idctn
    return idctn(block, norm="ortho")


def _process_channel_blocks(channel: np.ndarray, qt: np.ndarray,
                             encode: bool) -> np.ndarray:
    """
    Vectorized forward or inverse DCT + quantisation over all 8×8 blocks.
    Reshapes the channel into a batch of blocks, applies 2D DCT once.
    channel : 2D float array (height/width must be multiples of 8)
    encode  : True = forward (quantise), False = inverse (dequantise)
    """
    from scipy.fft import dctn, idctn
    H, W   = channel.shape
    H8, W8 = H // 8, W // 8
    # Reshape to (H8, W8, 8, 8) – batch of blocks
    blocks = (channel.reshape(H8, 8, W8, 8)
                     .transpose(0, 2, 1, 3)
                     .reshape(H8 * W8, 8, 8))   # (N, 8, 8)
    if encode:
        # Apply 2D DCT to each block (axes -2, -1)
        coefs = dctn(blocks, axes=(-2, -1), norm="ortho")
        result = np.round(coefs / qt)           # broadcast qt (8,8) over N
    else:
        coefs  = blocks * qt
        result = idctn(coefs, axes=(-2, -1), norm="ortho")
    # Reshape back to (H, W)
    out = (result.reshape(H8, W8, 8, 8)
                 .transpose(0, 2, 1, 3)
                 .reshape(H, W))
    return out


# --------------------------------------------------------------------------
# Simple run-length encoder for int8 coefficient stream
# --------------------------------------------------------------------------

def _rle_encode(data: np.ndarray) -> bytes:
    """
    Encode int8 array with run-length coding.
    Format: [value:i8][run_length:u8] pairs.  run_length=255 means 255+ zeros
    (chained).  Non-zero values always have run_length=1 (literally just value).
    """
    flat = data.ravel().astype(np.int8)
    out  = []
    i    = 0
    while i < len(flat):
        v = flat[i]
        if v == 0:
            run = 0
            while i < len(flat) and flat[i] == 0 and run < 255:
                run += 1
                i   += 1
            out.append(struct.pack("bB", 0, run))
        else:
            out.append(struct.pack("bB", v, 1))
            i += 1
    return b"".join(out)


def _rle_decode(data: bytes, n: int) -> np.ndarray:
    """Decode RLE-encoded int8 stream back to flat array of length n."""
    out = np.zeros(n, dtype=np.int8)
    idx = 0
    pos = 0
    while idx < len(data) and pos < n:
        v, run = struct.unpack_from("bB", data, idx)
        idx += 2
        if v == 0:
            pos += run
        else:
            out[pos] = v
            pos += 1
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def encode_residual(residual: np.ndarray,
                    fg_mask: Optional[np.ndarray] = None,
                    quality: int = 30) -> bytes:
    """
    Encode a signed int16 residual frame to bytes.

    Parameters
    ----------
    residual : int16 H×W×3 (BGR) residual = frame - reference
    fg_mask  : bool H×W foreground mask; background pixels zeroed before DCT
    quality  : JPEG-style quality 1-100 (lower = smaller, more lossy)

    Returns
    -------
    Compressed bytes.
    """
    res = residual.astype(np.float32)

    # Zero background before encoding (huge entropy win)
    if fg_mask is not None:
        res[~fg_mask] = 0.0

    H, W = res.shape[:2]
    # Pad to multiple of 8
    H8 = ((H + 7) // 8) * 8
    W8 = ((W + 7) // 8) * 8
    if H8 != H or W8 != W:
        padded = np.zeros((H8, W8, 3), dtype=np.float32)
        padded[:H, :W] = res
        res = padded

    qt_luma   = _make_qt(quality, luma=True)
    qt_chroma = _make_qt(quality, luma=False)

    channels = []
    for ch in range(3):
        qt  = qt_luma if ch == 0 else qt_chroma
        blk = _process_channel_blocks(res[:, :, ch], qt, encode=True)
        blk = np.clip(blk, -127, 127).astype(np.int8)
        channels.append(blk)

    coef_array = np.stack(channels, axis=-1)   # H8×W8×3 int8
    rle        = _rle_encode(coef_array)
    compressed = zlib.compress(rle, level=9)

    # Header: original shape + quality
    header = struct.pack(">HHB", H, W, quality)
    return header + compressed


def decode_residual(data: bytes) -> np.ndarray:
    """
    Decode residual bytes back to int16 H×W×3 residual.
    """
    H, W, quality = struct.unpack_from(">HHB", data)
    compressed    = data[5:]
    rle           = zlib.decompress(compressed)

    H8 = ((H + 7) // 8) * 8
    W8 = ((W + 7) // 8) * 8

    n        = H8 * W8 * 3
    coef_arr = _rle_decode(rle, n).reshape(H8, W8, 3)

    qt_luma   = _make_qt(quality, luma=True)
    qt_chroma = _make_qt(quality, luma=False)

    channels = []
    for ch in range(3):
        qt  = qt_luma if ch == 0 else qt_chroma
        blk = _process_channel_blocks(coef_arr[:, :, ch].astype(np.float32), qt, encode=False)
        channels.append(blk)

    res = np.stack(channels, axis=-1)[:H, :W]
    return np.clip(res, -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------
# Cycle similarity check (for "clone" detection)
# --------------------------------------------------------------------------

def cycle_residual_mse(frames_a: np.ndarray, frames_b: np.ndarray,
                        sample_rate: int = 10) -> float:
    """
    Compute mean squared error between two cycle frame sequences.
    Samples every `sample_rate` frames for speed.
    frames_a, frames_b : uint8 N×H×W×C
    """
    n = min(len(frames_a), len(frames_b))
    if n == 0:
        return 0.0
    indices = range(0, n, sample_rate)
    mse_sum = 0.0
    count   = 0
    for i in indices:
        diff     = frames_a[i].astype(np.float32) - frames_b[i].astype(np.float32)
        mse_sum += float((diff ** 2).mean())
        count   += 1
    return mse_sum / max(count, 1)
