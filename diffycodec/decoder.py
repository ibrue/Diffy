"""
DiffyDecoder: reconstruct frames from a .dfy bitstream.
"""

import json
import struct
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None  # optional; only needed for decode_to_video()
from typing import Iterator, List, Optional, Tuple

from .bitstream      import BitstreamReader, ChunkType
from .background     import decode_background_jpeg
from .residual_codec import decode_residual
from .temporal_codec import decode_cycle_temporal, decode_frame
from .imu            import FrameStabilizer, unpack_imu_quats
from .vq_codec       import VQCodebook
from .gaussian_splatting import GaussianSplatModel


class DiffyDecoder:
    def __init__(self, input_path: str):
        self._path = input_path
        self._bg:           Optional[np.ndarray]            = None
        self._canonicals:   List[np.ndarray]               = []
        self._cycle_chunks: List[Tuple[ChunkType, bytes]]  = []
        self._imu_quats:    Optional[np.ndarray]           = None
        self._meta:         dict                           = {}
        self._vq_codebook:  Optional[VQCodebook]           = None
        self._splat_model:  Optional[GaussianSplatModel]   = None
        self._load()

    def _load(self) -> None:
        # Collect raw cycle payloads so we can resolve the background reference
        # before decoding any frames (splat model must be rendered first).
        pending_canons: list = []
        pending_noncanons: list = []

        with BitstreamReader(self._path) as r:
            self.header = r.header
            for chunk_type, payload in r.read_chunks():
                if chunk_type == ChunkType.METADATA:
                    self._meta = json.loads(payload.decode())
                elif chunk_type == ChunkType.BACKGROUND:
                    # Store JPEG as fallback for files without splat model.
                    if self._bg is None:
                        self._bg = decode_background_jpeg(payload)
                elif chunk_type == ChunkType.IMU_BLOCK:
                    self._imu_quats = unpack_imu_quats(payload)
                elif chunk_type == ChunkType.CODEBOOK:
                    self._vq_codebook = VQCodebook.from_bytes(payload)
                elif chunk_type == ChunkType.SPLAT_MODEL:
                    self._splat_model = GaussianSplatModel.from_bytes(payload)
                elif chunk_type == ChunkType.CYCLE_CANON:
                    pending_canons.append(payload)
                elif chunk_type in (ChunkType.CYCLE_DELTA, ChunkType.FRAME_SKIP):
                    pending_noncanons.append((chunk_type, payload))

        # If a splat model is present, render it to get the same background
        # the encoder used.  This is the authoritative reference for all
        # residual reconstruction — must match encoder exactly.
        if self._splat_model is not None:
            self._bg = self._splat_model.render(
                self.header["width"], self.header["height"])

        # Now decode cycles against the resolved background.
        for payload in pending_canons:
            frames = self._decode_canon_chunk(payload)
            self._canonicals.append(frames)
        self._cycle_chunks = pending_noncanons

    def _decode_canon_chunk(self, payload: bytes) -> np.ndarray:
        """Handles both temporal-coded and legacy canonical chunks."""
        if len(payload) < 5:
            return np.array([])
        n, flags = struct.unpack_from(">IB", payload, 0)
        if flags & 0x01:  # FLAG_TEMPORAL
            return decode_cycle_temporal(payload, self._bg,
                                         vq_codebook=self._vq_codebook)
        else:
            return self._decode_legacy_cycle(payload[5:], n)

    def _decode_legacy_cycle(self, data: bytes, n: int) -> np.ndarray:
        offset = 0
        frames = []
        for _ in range(n):
            length   = struct.unpack_from(">I", data, offset)[0]
            offset  += 4
            residual = decode_residual(data[offset:offset + length])
            offset  += length
            recon    = (self._bg.astype(np.int16) + residual
                        if self._bg is not None else residual)
            frames.append(np.clip(recon, 0, 255).astype(np.uint8))
        return np.array(frames) if frames else np.array([])

    def iter_frames(self) -> Iterator[np.ndarray]:
        for frames in self._iter_all_cycles():
            yield from frames

    def iter_cycles(self) -> Iterator[np.ndarray]:
        yield from self._iter_all_cycles()

    def decode_to_video(self, output_path: str) -> None:
        if cv2 is None:
            raise ImportError("cv2 (opencv-python) is required for decode_to_video()")
        h = self.header
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, h["fps"],
                                 (h["width"], h["height"]))
        for frame in self.iter_frames():
            writer.write(frame)
        writer.release()

    def _decode_noncanon(self, chunk_type: ChunkType, payload: bytes) -> np.ndarray:
        if chunk_type == ChunkType.FRAME_SKIP:
            canon_idx, frame_count = struct.unpack_from(">HH", payload)
            ref = (self._canonicals[canon_idx]
                   if canon_idx < len(self._canonicals) else None)
            return (ref[:frame_count] if ref is not None
                    else np.stack([self._bg] * frame_count))
        elif chunk_type == ChunkType.CYCLE_DELTA:
            return self._decode_delta_cycle(payload)
        return np.zeros((0, self.header["height"], self.header["width"], 3), dtype=np.uint8)

    def _iter_all_cycles(self) -> Iterator[np.ndarray]:
        cycle_map = self._meta.get('cycle_map')
        if cycle_map:
            # Use the temporal order written by the encoder
            for entry in cycle_map:
                type_flag, idx = int(entry[0]), int(entry[1])
                if type_flag == 0:   # canonical
                    if idx < len(self._canonicals):
                        yield self._canonicals[idx]
                else:                # non-canonical (delta or skip)
                    if idx < len(self._cycle_chunks):
                        chunk_type, payload = self._cycle_chunks[idx]
                        yield self._decode_noncanon(chunk_type, payload)
        else:
            # Fallback for old files without cycle_map
            for canon_frames in self._canonicals:
                yield canon_frames
            for chunk_type, payload in self._cycle_chunks:
                yield self._decode_noncanon(chunk_type, payload)

    def _decode_delta_cycle(self, payload: bytes) -> np.ndarray:
        canon_idx, n = struct.unpack_from(">HI", payload)
        canon_frames = (self._canonicals[canon_idx]
                        if canon_idx < len(self._canonicals) else None)
        offset = 6
        frames = []
        for i in range(n):
            length   = struct.unpack_from(">I", payload, offset)[0]
            offset  += 4
            residual = decode_frame(payload[offset:offset + length], use_bbox=True)
            offset  += length
            ref      = (canon_frames[i].astype(np.int16)
                        if canon_frames is not None and i < len(canon_frames)
                        else np.zeros_like(residual))
            frames.append(np.clip(ref + residual, 0, 255).astype(np.uint8))
        return np.array(frames) if frames else np.array([])

    @property
    def total_frames(self) -> int:
        return self._meta.get("total_frames", self.header.get("total_frames", 0))

    @property
    def metadata(self) -> dict:
        return self._meta

    @property
    def fps(self) -> float:
        return self.header["fps"]

    @property
    def background(self) -> Optional[np.ndarray]:
        return self._bg

    @property
    def splat_model(self) -> Optional[GaussianSplatModel]:
        return self._splat_model

    @property
    def has_splats(self) -> bool:
        return self._splat_model is not None
