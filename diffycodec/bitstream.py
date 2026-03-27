"""
Bitstream I/O for the .dfy container format.

File layout
-----------
[4B magic]  "DFY\x01"
[8B u64]    total_frames
[4B f32]    fps
[2B u16]    width
[2B u16]    height
[1B]        flags  (bit0=has_imu, bit1=stereo_rgb)
--- chunks, each prefixed by ---
[1B]        chunk_type  (see ChunkType)
[4B u32]    chunk_byte_length
[N bytes]   chunk payload
"""

import struct
import zlib
import io
from enum import IntEnum
from typing import Optional


MAGIC = b"DFY\x01"


class ChunkType(IntEnum):
    BACKGROUND   = 0x01   # JPEG-compressed background keyframe
    CYCLE_CANON  = 0x02   # canonical work cycle (JPEG sequence, delta-coded)
    CYCLE_DELTA  = 0x03   # per-cycle deviation vs canonical
    IMU_BLOCK    = 0x04   # IMU quaternions for this cycle (f16 × 4 × N)
    FRAME_SKIP   = 0x05   # run of skipped (interpolated) frame indices
    METADATA     = 0x06   # JSON metadata blob
    CODEBOOK     = 0x07   # VQ codebook (float16 centroids)


class BitstreamWriter:
    def __init__(self, path: str, total_frames: int, fps: float,
                 width: int, height: int, has_imu: bool = False):
        self._f = open(path, "wb")
        self._write_header(total_frames, fps, width, height, has_imu)

    def _write_header(self, total_frames, fps, width, height, has_imu):
        flags = 0x01 if has_imu else 0x00
        self._f.write(MAGIC)
        self._f.write(struct.pack(">QfHHB", total_frames, fps, width, height, flags))

    def write_chunk(self, chunk_type: ChunkType, payload: bytes, compress: bool = True):
        if compress:
            payload = zlib.compress(payload, level=6)
        header = struct.pack(">BBI", int(chunk_type), int(compress), len(payload))
        self._f.write(header)
        self._f.write(payload)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def bytes_written(self) -> int:
        return self._f.tell()


class BitstreamReader:
    def __init__(self, path: str):
        self._f = open(path, "rb")
        self.header = self._read_header()

    def _read_header(self) -> dict:
        magic = self._f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not a .dfy file (magic={magic!r})")
        total_frames, fps, width, height, flags = struct.unpack(">QfHHB", self._f.read(17))
        return dict(total_frames=total_frames, fps=fps,
                    width=width, height=height,
                    has_imu=bool(flags & 0x01))

    def read_chunks(self):
        """Yield (ChunkType, payload_bytes) pairs until EOF."""
        while True:
            raw = self._f.read(6)
            if not raw:
                break
            chunk_type, compressed, length = struct.unpack(">BBI", raw)
            payload = self._f.read(length)
            if compressed:
                payload = zlib.decompress(payload)
            yield ChunkType(chunk_type), payload

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
