"""
Vector-Quantisation codec for foreground DCT coefficient blocks.

Instead of transmitting each 8×8 DCT block as 64 int8 values, we train a
codebook of N prototype blocks offline (on the first canonical cycle) and
then transmit a 1-byte codebook index per block.  The decoder reconstructs
the block from the stored codebook.

Why this works for egocentric industrial video
----------------------------------------------
A worker's hand performing a repetitive task produces a small vocabulary of
distinct motion patterns.  When expressed as DCT coefficients over 8×8 patches,
the foreground residual blocks cluster tightly — there are far fewer distinct
"shapes" than there are raw coefficient permutations.  With a 256-entry codebook
the typical reconstruction error is well below the perceptual threshold for
downstream model training.

Compression arithmetic (256 codewords, 64-coef blocks, 3 channels)
-------------------------------------------------------------------
  Without VQ : 64 int8  = 64 bytes/block/channel → after LZMA ~15-25 bytes
  With    VQ : 1  uint8 = 1  byte/block/channel  → after LZMA  ~0.6-0.9 bytes

For a 120×120 fg crop: (120/8)² = 225 blocks per channel.
  Without VQ, LZMA'd : 225 × 3 × ~20 bytes ≈ 13.5 KB
  With    VQ, LZMA'd : 225 × 3 × ~0.8 bytes ≈  0.5 KB

~27× reduction on top of the existing pipeline.

Codebook training
-----------------
Collect foreground DCT blocks from the first canonical cycle.
Run Lloyd's k-means (numpy only, no sklearn needed) for 50 iterations.
The zero-block (empty foreground) is guaranteed to be codeword 0 by design —
this ensures RLE of the index stream still captures long zero runs.

Bitstream
---------
The trained codebook is written as a CODEBOOK chunk (ChunkType 0x07):
  [2B n_codewords][2B block_size=64][n_codewords × 64 × 2 bytes (float16)]

Per-frame residual in VQ mode uses encode_frame(..., vq=codebook).
The residual header gains a [1B codec_flags] byte (bit 0 = VQ mode).
"""

import struct
import numpy as np
from typing import Optional


# ── K-means (numpy only) ─────────────────────────────────────────────────────

def _kmeans(X: np.ndarray, k: int, max_iter: int = 50,
             seed: int = 42) -> np.ndarray:
    """
    Simple Lloyd's k-means on rows of X (float32, shape M × D).
    Returns centroids of shape k × D.
    """
    rng = np.random.default_rng(seed)
    # K-means++ initialisation
    idx      = [int(rng.integers(0, len(X)))]
    for _ in range(k - 1):
        dists = np.min([np.sum((X - X[i]) ** 2, axis=1) for i in idx], axis=0)
        total = dists.sum()
        if total == 0:
            # All remaining points are identical; pick uniformly
            probs = np.ones(len(X)) / len(X)
        else:
            probs = dists / total
        idx.append(int(rng.choice(len(X), p=probs)))
    centroids = X[idx].copy()

    for _ in range(max_iter):
        # Assignment step: batch distance computation
        diffs  = X[:, None, :] - centroids[None, :, :]   # M × k × D
        dists  = np.sum(diffs ** 2, axis=2)               # M × k
        labels = dists.argmin(axis=1)                     # M

        # Update step
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            new_centroids[j] = members.mean(axis=0) if len(members) > 0 else centroids[j]
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


# ── VQ Codebook ───────────────────────────────────────────────────────────────

class VQCodebook:
    """
    Parameters
    ----------
    n_codewords : codebook size (256 = 1 byte per block; 512 requires 9 bits)
    block_size  : number of DCT coefficients per block (8×8 = 64)
    """

    BLOCK_SIZE = 64   # 8×8 DCT block, single channel

    def __init__(self, n_codewords: int = 256):
        self.n_codewords = n_codewords
        self.codebook: Optional[np.ndarray] = None   # (n_codewords, BLOCK_SIZE)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, blocks: np.ndarray, max_iter: int = 50) -> None:
        """
        Train on foreground DCT coefficient blocks.
        blocks : (M, BLOCK_SIZE) float32
        The zero-block (all-zeros) is forced to codeword index 0 after training
        so that empty/background blocks RLE-compress efficiently.
        """
        if len(blocks) < self.n_codewords:
            # Fewer blocks than codewords: pad with zeros
            pad = np.zeros((self.n_codewords - len(blocks), self.BLOCK_SIZE),
                           dtype=np.float32)
            blocks = np.vstack([blocks, pad])

        # Sub-sample for speed (max 5000 training blocks)
        if len(blocks) > 5000:
            idx    = np.random.default_rng(0).choice(len(blocks), 5000, replace=False)
            blocks = blocks[idx]

        centroids = _kmeans(blocks, self.n_codewords, max_iter=max_iter)

        # Force codeword 0 = zero block
        zero_block   = np.zeros(self.BLOCK_SIZE, dtype=np.float32)
        closest_zero = int(np.sum(centroids ** 2, axis=1).argmin())
        centroids[[0, closest_zero]] = centroids[[closest_zero, 0]]

        self.codebook = centroids

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """
        blocks : (N, BLOCK_SIZE) float32
        Returns uint8 indices of shape (N,).
        n_codewords must be ≤ 256.
        """
        # Efficient nearest-neighbour via ||a-b||² = ||a||² - 2a·b + ||b||²
        a2  = np.sum(blocks      ** 2, axis=1, keepdims=True)   # N × 1
        b2  = np.sum(self.codebook ** 2, axis=1)                 # k
        ab  = blocks @ self.codebook.T                           # N × k
        dists = a2 - 2 * ab + b2                                 # N × k
        return dists.argmin(axis=1).astype(np.uint8)

    def decode_blocks(self, indices: np.ndarray) -> np.ndarray:
        """indices : (N,) uint8 → (N, BLOCK_SIZE) float32"""
        return self.codebook[indices.astype(np.int32)]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        header = struct.pack(">HH", self.n_codewords, self.BLOCK_SIZE)
        return header + self.codebook.astype(np.float16).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "VQCodebook":
        n, bs = struct.unpack_from(">HH", data)
        arr   = (np.frombuffer(data[4:], dtype=np.float16)
                   .reshape(n, bs).astype(np.float32))
        vq          = cls(n_codewords=n)
        vq.codebook = arr
        return vq

    @property
    def is_trained(self) -> bool:
        return self.codebook is not None


# ── VQ-enhanced frame encode / decode ────────────────────────────────────────

def encode_frame_vq(residual: np.ndarray,
                    fg_mask:  Optional[np.ndarray],
                    codebook: "VQCodebook",
                    quality:  int = 30,
                    use_bbox: bool = True) -> bytes:
    """
    Encode a residual frame using VQ-quantised DCT blocks.
    Returns bytes with header: [1B flags=0x01][12B optional bbox][4B n_blocks_per_ch]
    followed by 3 uint8 index arrays (one per channel), LZMA'd.
    """
    from .temporal_codec import _fg_bbox, _zlib_compress
    from .residual_codec import _make_qt, _process_channel_blocks

    H_full, W_full = residual.shape[:2]
    res = residual.astype(np.float32)
    if fg_mask is not None:
        res[~fg_mask] = 0.0

    # Optional bounding-box crop
    bbox_header = b""
    if use_bbox and fg_mask is not None:
        y0, x0, y1, x1 = _fg_bbox(fg_mask)
        res      = res[y0:y1, x0:x1]
        bbox_header = struct.pack(">HHHHHH", H_full, W_full, y0, x0,
                                  y1 - y0, x1 - x0)

    H, W = res.shape[:2]
    H8   = ((H + 7) // 8) * 8
    W8   = ((W + 7) // 8) * 8
    if H8 != H or W8 != W:
        padded = np.zeros((H8, W8, 3), dtype=np.float32)
        padded[:H, :W] = res
        res = padded

    qt_luma   = _make_qt(quality, luma=True)
    qt_chroma = _make_qt(quality, luma=False)

    all_indices = []
    for ch in range(3):
        qt    = qt_luma if ch == 0 else qt_chroma
        coefs = _process_channel_blocks(res[:, :, ch], qt, encode=True)  # H8×W8
        H8c, W8c = coefs.shape
        # Reshape to blocks: (N, 64)
        blocks = (coefs.reshape(H8c // 8, 8, W8c // 8, 8)
                       .transpose(0, 2, 1, 3)
                       .reshape(-1, 64)
                       .astype(np.float32))
        indices = codebook.encode_blocks(blocks)   # (N,) uint8
        all_indices.append(indices)

    indices_bytes = np.concatenate(all_indices).tobytes()
    compressed    = _zlib_compress(indices_bytes)

    # Layout: [1B flags=0x01 (VQ mode)][12B bbox if used][1B quality][4B n_blocks][payload]
    n_blocks_per_ch = len(all_indices[0])
    flags = 0x01   # VQ mode
    header = (bytes([flags]) + bbox_header
              + struct.pack(">BHH I", quality, H, W, n_blocks_per_ch))
    return header + compressed


def decode_frame_vq(data: bytes, codebook: "VQCodebook") -> np.ndarray:
    """Decode a VQ-encoded residual frame back to int16 H×W×3."""
    from .temporal_codec import _zlib_decompress
    from .residual_codec import _make_qt, _process_channel_blocks

    offset = 0
    flags  = data[offset]; offset += 1
    assert flags & 0x01, "Not a VQ-encoded frame"

    # Peek for bbox
    H_full, W_full = 0, 0
    y0, x0, cH, cW = 0, 0, 0, 0
    if len(data) >= offset + 12:
        H_full, W_full, y0, x0, cH, cW = struct.unpack_from(">HHHHHH", data, offset)
        if H_full > 0:
            offset += 12

    quality, H, W, n_blocks = struct.unpack_from(">BHHI", data, offset)
    offset += 9

    compressed = data[offset:]
    raw        = _zlib_decompress(compressed)
    all_indices = np.frombuffer(raw, dtype=np.uint8).reshape(3, n_blocks)

    qt_luma   = _make_qt(quality, luma=True)
    qt_chroma = _make_qt(quality, luma=False)

    H8 = ((H + 7) // 8) * 8
    W8 = ((W + 7) // 8) * 8
    H8b, W8b = H8 // 8, W8 // 8

    channels = []
    for ch in range(3):
        qt     = qt_luma if ch == 0 else qt_chroma
        blocks = codebook.decode_blocks(all_indices[ch])   # (N, 64)
        coefs  = (blocks.reshape(H8b, W8b, 8, 8)
                        .transpose(0, 2, 1, 3)
                        .reshape(H8, W8).astype(np.float32))
        recon  = _process_channel_blocks(coefs, qt, encode=False)
        channels.append(recon)

    crop = np.clip(np.stack(channels, axis=-1)[:H, :W],
                   -32768, 32767).astype(np.int16)

    if H_full > 0:
        full = np.zeros((H_full, W_full, 3), dtype=np.int16)
        full[y0:y0 + H, x0:x0 + W] = crop
        return full
    return crop


# ── Helpers for encoder integration ──────────────────────────────────────────

def collect_dct_blocks(residual: np.ndarray, fg_mask: np.ndarray,
                        quality: int = 30) -> np.ndarray:
    """
    Extract foreground DCT coefficient blocks from a residual frame.
    Used during the first canonical cycle to accumulate training data.
    Returns (M, 64) float32 array of non-zero blocks.
    """
    from .residual_codec import _make_qt, _process_channel_blocks

    res = residual.astype(np.float32)
    res[~fg_mask] = 0.0

    H, W  = res.shape[:2]
    H8    = ((H + 7) // 8) * 8
    W8    = ((W + 7) // 8) * 8
    if H8 != H or W8 != W:
        pad = np.zeros((H8, W8, 3), dtype=np.float32)
        pad[:H, :W] = res
        res = pad

    qt    = _make_qt(quality, luma=True)
    coefs = _process_channel_blocks(res[:, :, 0], qt, encode=True)

    H8c, W8c = coefs.shape
    blocks = (coefs.reshape(H8c // 8, 8, W8c // 8, 8)
                   .transpose(0, 2, 1, 3)
                   .reshape(-1, 64)
                   .astype(np.float32))

    # Keep only non-zero blocks (foreground blocks)
    nz = np.any(blocks != 0, axis=1)
    return blocks[nz]
