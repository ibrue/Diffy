"""
Temporal prediction codec for within-cycle inter-frame coding.

Within a canonical cycle the camera is head-mounted and the environment
is static.  Between consecutive frames:
  • Background pixels: zero delta (already removed by background model)
  • Foreground blob: moves ~3-10 pixels/frame → only the leading/trailing
    edge of the blob changes → residual is a thin shell, not the full blob

This reduces each P-frame to ~5-15% of the equivalent I-frame size.

Frame types
-----------
I-frame  : residual = frame - background   (full foreground encoding)
P-frame  : residual = frame[k] - frame[k-1]  (temporal delta only)

Storage format per cycle
------------------------
[4B  n_frames   : uint32]
[1B  flags      : bit0=temporal_coding, bit1=bbox_encoding]
per frame:
  [1B  frame_type : 0=I, 1=P]
  [4B  payload_sz : uint32]
  [N bytes payload]

Foreground bounding-box encoding
---------------------------------
Instead of sending the full H×W residual frame (99% zeros), we compute the
tight bounding box of the foreground content and encode only that crop.
The decoder places the crop back at the right position.

Bbox header (prepended to residual payload when bbox_encoding flag is set):
  [2B full_H][2B full_W][2B y0][2B x0][2B crop_H][2B crop_W]
  = 12 bytes overhead per frame, saves ~50× DCT block processing.

Compression chain
-----------------
  residual (int16) →  fg mask zero-out  →  bbox crop
    →  8×8 DCT blocks (vectorised)
    →  quantise  →  int8 clip
    →  RLE  →  zlib deflate
"""

import struct
import zlib
import numpy as np
from typing import Optional, List, Tuple

from .residual_codec import _make_qt, _process_channel_blocks, _rle_encode, _rle_decode


# ── Flags ────────────────────────────────────────────────────────────────────────────
FLAG_TEMPORAL = 0x01
FLAG_BBOX     = 0x02
FLAG_VQ       = 0x04

IFRAME = 0
PFRAME = 1


# ── zlib helpers ───────────────────────────────────────────────────────────────────────────

def _zlib_compress(data: bytes) -> bytes:
    return zlib.compress(data, level=6)


def _zlib_decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


# ── Bounding-box helpers ───────────────────────────────────────────────────────────────────

def _fg_bbox(fg_mask: np.ndarray, margin: int = 8) -> Tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) tight bounding box of True pixels + margin."""
    rows = np.where(fg_mask.any(axis=1))[0]
    cols = np.where(fg_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, fg_mask.shape[0], fg_mask.shape[1]
    H, W = fg_mask.shape
    y0 = max(0, int(rows[0])  - margin)
    y1 = min(H, int(rows[-1]) + margin + 1)
    x0 = max(0, int(cols[0])  - margin)
    x1 = min(W, int(cols[-1]) + margin + 1)
    return y0, x0, y1, x1


# ── Core frame encoder / decoder ──────────────────────────────────────────────────────────

def encode_frame(residual: np.ndarray,
                 fg_mask:  Optional[np.ndarray] = None,
                 quality:  int = 30,
                 use_bbox: bool = True) -> bytes:
    """
    Encode an int16 H×W×3 residual to bytes.
    When use_bbox=True, only the foreground bounding-box region is encoded.

    Returns: [12B bbox header if use_bbox] + [5B std header] + [zlib payload]
    """
    H_full, W_full = residual.shape[:2]
    res = residual.astype(np.float32)

    if fg_mask is not None:
        res[~fg_mask] = 0.0

    # Bounding-box crop
    bbox_header = b""
    if use_bbox and fg_mask is not None:
        y0, x0, y1, x1 = _fg_bbox(fg_mask)
        res      = res[y0:y1, x0:x1]
        fg_mask  = fg_mask[y0:y1, x0:x1]
        bbox_header = struct.pack(">HHHHHH",
                                  H_full, W_full, y0, x0,
                                  y1 - y0, x1 - x0)
    else:
        y0, x0 = 0, 0

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
        channels.append(np.clip(blk, -127, 127).astype(np.int8))

    coef_array = np.stack(channels, axis=-1)
    rle        = _rle_encode(coef_array)
    compressed = _zlib_compress(rle)

    std_header = struct.pack(">HHB", H, W, quality)
    return bbox_header + std_header + compressed


def decode_frame(data: bytes, use_bbox: bool = True) -> np.ndarray:
    """
    Decode bytes back to int16 H×W×3 residual, placed in the full frame.
    Returns full-frame int16 residual (zeros outside bbox if bbox was used).
    """
    offset = 0
    if use_bbox and len(data) >= 17:
        # Peek at first 12 bytes to see if they look like a bbox header
        H_full, W_full, y0, x0, cH, cW = struct.unpack_from(">HHHHHH", data, 0)
        if H_full > 0 and W_full > 0 and cH > 0 and cW > 0:
            offset = 12
            H, W, quality = struct.unpack_from(">HHB", data, offset)
            offset += 5
            rle  = _zlib_decompress(data[offset:])
            H8   = ((H + 7) // 8) * 8
            W8   = ((W + 7) // 8) * 8
            n    = H8 * W8 * 3
            coef = _rle_decode(rle, n).reshape(H8, W8, 3)
            qt_luma   = _make_qt(quality, luma=True)
            qt_chroma = _make_qt(quality, luma=False)
            channels  = []
            for ch in range(3):
                qt  = qt_luma if ch == 0 else qt_chroma
                blk = _process_channel_blocks(coef[:, :, ch].astype(np.float32),
                                               qt, encode=False)
                channels.append(blk)
            crop = np.clip(np.stack(channels, axis=-1)[:H, :W], -32768, 32767).astype(np.int16)
            full = np.zeros((H_full, W_full, 3), dtype=np.int16)
            full[y0:y0+H, x0:x0+W] = crop
            return full

    # Fall back: no bbox header
    H, W, quality = struct.unpack_from(">HHB", data, 0)
    rle  = _zlib_decompress(data[5:])
    H8   = ((H + 7) // 8) * 8
    W8   = ((W + 7) // 8) * 8
    n    = H8 * W8 * 3
    coef = _rle_decode(rle, n).reshape(H8, W8, 3)
    qt_luma   = _make_qt(quality, luma=True)
    qt_chroma = _make_qt(quality, luma=False)
    channels  = []
    for ch in range(3):
        qt  = qt_luma if ch == 0 else qt_chroma
        blk = _process_channel_blocks(coef[:, :, ch].astype(np.float32), qt, encode=False)
        channels.append(blk)
    res = np.clip(np.stack(channels, axis=-1)[:H, :W], -32768, 32767).astype(np.int16)
    return res


# ── Cycle encoder / decoder ─────────────────────────────────────────────────────────────────────

def encode_cycle_temporal(frames: List[np.ndarray],
                           background: np.ndarray,
                           fg_model,         # BackgroundModel instance
                           quality: int = 30,
                           use_bbox: bool = True,
                           vq_codebook=None) -> bytes:
    """
    Encode a list of uint8 frames using temporal prediction.

    Frame 0: I-frame  (residual vs background)
    Frame k: P-frame  (residual vs previous reconstructed frame)

    When vq_codebook is provided, VQ-quantised DCT blocks replace the standard
    DCT+RLE+zlib path, giving an additional ~27× reduction on foreground blocks.

    Returns packed bytes: [4B n][1B flags][per-frame: [1B type][4B size][payload]]
    """
    n      = len(frames)
    use_vq = vq_codebook is not None
    flags  = FLAG_TEMPORAL | (FLAG_BBOX if use_bbox else 0) | (FLAG_VQ if use_vq else 0)
    parts  = [struct.pack(">IB", n, flags)]

    if use_vq:
        from .vq_codec import encode_frame_vq

    bg_f32   = background.astype(np.float32)
    prev_rec = background.copy().astype(np.float32)   # reconstructed previous frame

    for k, frame in enumerate(frames):
        fg_mask  = fg_model.get_foreground_mask(frame) if fg_model.is_ready else None

        if k == 0:
            # I-frame: residual vs background
            residual   = frame.astype(np.int16) - background.astype(np.int16)
            frame_type = IFRAME
        else:
            # P-frame: residual vs previous reconstructed frame
            residual   = frame.astype(np.int16) - np.clip(prev_rec, 0, 255).astype(np.int16)
            frame_type = PFRAME

        if use_vq:
            payload = encode_frame_vq(residual, fg_mask, vq_codebook, quality, use_bbox)
        else:
            payload = encode_frame(residual, fg_mask, quality, use_bbox)

        # Update prev_rec: reconstruct = prev_rec + residual  (for P) or bg + residual (for I)
        if k == 0:
            prev_rec = bg_f32 + residual.astype(np.float32)
        else:
            prev_rec = prev_rec + residual.astype(np.float32)
        prev_rec = np.clip(prev_rec, 0, 255)

        parts.append(struct.pack(">BI", frame_type, len(payload)))
        parts.append(payload)

    return b"".join(parts)


def decode_cycle_temporal(data: bytes,
                           background: np.ndarray,
                           vq_codebook=None) -> np.ndarray:
    """
    Decode a temporal-coded cycle blob back to N×H×W×3 uint8.
    Pass vq_codebook when the cycle was encoded with FLAG_VQ.
    """
    offset = 0
    n, flags = struct.unpack_from(">IB", data, offset)
    offset += 5

    use_bbox = bool(flags & FLAG_BBOX)
    use_vq   = bool(flags & FLAG_VQ)

    if use_vq:
        from .vq_codec import decode_frame_vq

    H, W = background.shape[:2]
    frames   = []
    prev_rec = background.copy().astype(np.float32)

    for _ in range(n):
        frame_type, payload_sz = struct.unpack_from(">BI", data, offset)
        offset += 5
        payload  = data[offset:offset + payload_sz]
        offset  += payload_sz

        if use_vq and vq_codebook is not None:
            residual = decode_frame_vq(payload, vq_codebook)
        else:
            residual = decode_frame(payload, use_bbox=use_bbox)

        if frame_type == IFRAME:
            recon    = background.astype(np.int16) + residual
            prev_rec = np.clip(recon, 0, 255).astype(np.float32)
        else:  # PFRAME
            recon    = np.clip(prev_rec, 0, 255).astype(np.int16) + residual
            prev_rec = np.clip(recon, 0, 255).astype(np.float32)

        frames.append(np.clip(recon, 0, 255).astype(np.uint8))

    return np.array(frames) if frames else np.zeros((0, H, W, 3), dtype=np.uint8)


# ── Phase-aligned cycle similarity ────────────────────────────────────────────────────────────

def find_best_phase_offset(frames_a: np.ndarray, frames_b: np.ndarray,
                            bg: np.ndarray,
                            max_offset: int = 20,
                            n_sample:   int = 10) -> Tuple[int, float]:
    """
    Try circular offsets ±max_offset and return (best_offset, best_mse).
    Uses background-subtracted frames sampled at n_sample points for speed.

    best_offset > 0  means  frames_b[offset:] aligns with frames_a
    best_offset < 0  means  frames_a[-offset:] aligns with frames_b
    """
    len_a, len_b = len(frames_a), len(frames_b)
    bg_f32 = bg.astype(np.float32)

    def _bg_sub(frames):
        return np.clip(frames.astype(np.float32) - bg_f32, -128, 127)

    best_mse    = float("inf")
    best_offset = 0

    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            a_slice = frames_a[:len_a - offset] if offset > 0 else frames_a
            b_slice = frames_b[offset:]         if offset > 0 else frames_b
        else:
            a_slice = frames_a[-offset:]
            b_slice = frames_b[:len_b + offset] if (-offset) < len_b else frames_b

        n = min(len(a_slice), len(b_slice))
        if n < 5:
            continue

        idx  = np.linspace(0, n - 1, min(n_sample, n), dtype=int)
        diff = _bg_sub(a_slice[idx]) - _bg_sub(b_slice[idx])
        mse  = float((diff ** 2).mean())

        if mse < best_mse:
            best_mse    = mse
            best_offset = offset

    return best_offset, best_mse
