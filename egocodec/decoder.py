"""
EgoDecoder: reconstruct frames from an .ego bitstream.

Playback modes
--------------
  full          : reconstruct every frame (slow, for export/analysis)
  random_access : seek to a specific frame index efficiently
  cycle_iter    : iterate one decoded cycle at a time (lowest memory)

Memory model
------------
The decoder maintains:
  • background plate     : 1 uint8 H×W×3 array (constant)
  • canonical cycles     : list of N×H×W×3 uint8 arrays (small, kept in RAM)
  • current cycle buffer : one cycle at a time during iteration
"""

import json
import struct
import numpy as np
import cv2
from typing import Iterator, List, Optional, Tuple

from .bitstream      import BitstreamReader, ChunkType
from .background     import decode_background_jpeg
from .residual_codec import decode_residual
from .imu            import FrameStabilizer, unpack_imu_quats


class EgoDecoder:
    """
    Parameters
    ----------
    input_path : path to an .ego file
    """

    def __init__(self, input_path: str):
        self._path = input_path
        self._bg:          Optional[np.ndarray]       = None
        self._canonicals:  List[np.ndarray]            = []  # list of N×H×W×3 uint8
        self._cycle_chunks: List[Tuple[ChunkType, bytes]] = []
        self._imu_quats:   Optional[np.ndarray]        = None
        self._meta:        dict                        = {}

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with BitstreamReader(self._path) as r:
            self.header = r.header
            for chunk_type, payload in r.read_chunks():
                if chunk_type == ChunkType.METADATA:
                    self._meta = json.loads(payload.decode())
                elif chunk_type == ChunkType.BACKGROUND:
                    self._bg = decode_background_jpeg(payload)
                elif chunk_type == ChunkType.IMU_BLOCK:
                    self._imu_quats = unpack_imu_quats(payload)
                elif chunk_type == ChunkType.CYCLE_CANON:
                    frames = self._decode_cycle_blob(payload, reference=self._bg)
                    self._canonicals.append(frames)
                elif chunk_type in (ChunkType.CYCLE_DELTA, ChunkType.FRAME_SKIP):
                    self._cycle_chunks.append((chunk_type, payload))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield all decoded uint8 H×W×3 frames in order."""
        # Yield background warmup frames (they were not cycle-encoded)
        warmup = self._meta.get("warmup_frames", 0)
        if self._bg is not None:
            for _ in range(warmup):
                yield self._bg.copy()

        # Yield canonical cycles and then delta/skip cycles interleaved
        # (The encoder writes: all canonicals, then all deltas in order)
        # We replay them in the order they appear in the stream.
        for frames in self._iter_all_cycles():
            yield from frames

    def iter_cycles(self) -> Iterator[np.ndarray]:
        """Yield one cycle at a time as N×H×W×3 uint8 arrays."""
        for frames in self._iter_all_cycles():
            yield frames

    def decode_to_video(self, output_path: str) -> None:
        """Decode the full .ego stream to an mp4 file."""
        h = self.header
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, h["fps"], (h["width"], h["height"]))
        for frame in self.iter_frames():
            writer.write(frame)
        writer.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _iter_all_cycles(self) -> Iterator[np.ndarray]:
        """
        Replay all cycles (canonical + delta/skip) in emission order.
        The encoder emits: [all canonical chunks] [all delta/skip chunks].
        We replay canonicals first, then deltas.
        """
        # Emit canonical cycles
        for canon_frames in self._canonicals:
            yield canon_frames

        # Emit delta / skip cycles
        for chunk_type, payload in self._cycle_chunks:
            if chunk_type == ChunkType.FRAME_SKIP:
                canon_idx, frame_count = struct.unpack_from(">HH", payload)
                ref = self._canonicals[canon_idx] if canon_idx < len(self._canonicals) else None
                if ref is not None:
                    # Return first frame_count frames of the canonical (clone)
                    yield ref[:frame_count]
                else:
                    yield np.stack([self._bg] * frame_count) if self._bg is not None else np.array([])
            elif chunk_type == ChunkType.CYCLE_DELTA:
                yield self._decode_delta_cycle(payload)

    def _decode_cycle_blob(self, payload: bytes,
                            reference: Optional[np.ndarray]) -> np.ndarray:
        """
        Decode a CYCLE_CANON blob (frame residuals vs background).
        Returns N×H×W×C uint8 array.
        """
        offset     = 0
        frame_count = struct.unpack_from(">I", payload, offset)[0]
        offset += 4
        frames  = []
        for _ in range(frame_count):
            length   = struct.unpack_from(">I", payload, offset)[0]
            offset  += 4
            residual = decode_residual(payload[offset:offset + length])
            offset  += length
            if reference is not None:
                recon = reference.astype(np.int16) + residual
            else:
                recon = residual
            frames.append(np.clip(recon, 0, 255).astype(np.uint8))
        return np.array(frames) if frames else np.array([])

    def _decode_delta_cycle(self, payload: bytes) -> np.ndarray:
        """
        Decode a CYCLE_DELTA blob (frame residuals vs canonical).
        Canon index is embedded in the first 2 bytes.
        """
        canon_idx = struct.unpack_from(">H", payload)[0]
        rest      = payload[2:]

        offset      = 0
        frame_count = struct.unpack_from(">I", rest, offset)[0]
        offset += 4

        canon_frames = self._canonicals[canon_idx] if canon_idx < len(self._canonicals) else None
        frames       = []
        for i in range(frame_count):
            length   = struct.unpack_from(">I", rest, offset)[0]
            offset  += 4
            residual = decode_residual(rest[offset:offset + length])
            offset  += length
            if canon_frames is not None and i < len(canon_frames):
                recon = canon_frames[i].astype(np.int16) + residual
            else:
                recon = residual
            frames.append(np.clip(recon, 0, 255).astype(np.uint8))
        return np.array(frames) if frames else np.array([])

    @property
    def total_frames(self) -> int:
        return self._meta.get("total_frames", self.header.get("total_frames", 0))

    @property
    def fps(self) -> float:
        return self.header["fps"]

    @property
    def background(self) -> Optional[np.ndarray]:
        return self._bg
