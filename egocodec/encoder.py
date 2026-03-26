"""
EgoEncoder: top-level encoder.

Pipeline
--------
  Frame source (mp4 / camera / numpy array)
      │
      ▼
  IMU stabilisation  ─────────────── IMU stream saved to bitstream
      │
      ▼
  Background model warmup (first 300 frames)
      │                          │
      └──► background keyframe ──┘  (written once to bitstream)
      │
      ▼
  Foreground extraction
      │
      ▼
  Cycle energy push ──► CycleDetector
      │
      ▼  (after all frames ingested: segment + encode)
  For each canonical cycle:
      └──► JPEG sequence of foreground residuals vs background → CYCLE_CANON chunk
  For each non-canonical cycle:
      └──► residual vs canonical cycle → CYCLE_DELTA chunk
             (if MSE < threshold → 2-byte CLONE record instead)
      │
      ▼
  BitstreamWriter.close()
"""

import json
import struct
import numpy as np
from typing import Optional, List, Iterator
import cv2

from .bitstream   import BitstreamWriter, ChunkType
from .background  import BackgroundModel, encode_background_jpeg
from .cycle_detector import CycleDetector, CycleSegmentation, compute_fg_energy
from .residual_codec  import encode_residual, cycle_residual_mse
from .imu             import IMUIntegrator, FrameStabilizer, pack_imu_quats


class EgoEncoder:
    """
    Parameters
    ----------
    output_path     : path for the .ego file
    fps             : camera frame rate
    width, height   : frame dimensions
    quality         : residual codec quality (1-100; lower = more compression)
    warmup_frames   : background model warmup length
    has_imu         : whether IMU data will be supplied
    K               : camera intrinsics (3×3 numpy array); None = auto-estimate
    """

    def __init__(self,
                 output_path: str,
                 fps: float = 30.0,
                 width: int = 1920,
                 height: int = 1080,
                 quality: int = 25,
                 warmup_frames: int = 300,
                 has_imu: bool = False,
                 K: Optional[np.ndarray] = None):
        self.fps     = fps
        self.width   = width
        self.height  = height
        self.quality = quality
        self.has_imu = has_imu

        self._bg_model   = BackgroundModel(warmup_frames=warmup_frames)
        self._cycle_det  = CycleDetector(fps=fps)
        self._imu_int    = IMUIntegrator(camera_hz=fps) if has_imu else None
        self._stabilizer = FrameStabilizer(K=K, width=width, height=height) if has_imu else None

        # We buffer all frames to allow two-pass encoding
        # In a streaming implementation this would use a disk-based ring buffer
        self._frame_buffer: List[np.ndarray] = []
        self._imu_quats:    List[np.ndarray] = []
        self._prev_frame:   Optional[np.ndarray] = None

        # Will be populated after encode() is called
        self._writer: Optional[BitstreamWriter] = None
        self._output_path = output_path
        self._total_frames = 0

    # ------------------------------------------------------------------
    # Frame + IMU ingestion
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray,
                   imu_quat: Optional[np.ndarray] = None) -> None:
        """
        Push a single uint8 H×W×C frame.
        imu_quat : [x,y,z,w] orientation at this frame (required if has_imu=True)
        """
        # Stabilise if IMU available
        if self.has_imu and imu_quat is not None and self._stabilizer is not None:
            frame = self._stabilizer.warp_frame(frame, imu_quat)
            self._imu_quats.append(imu_quat)

        self._bg_model.update(frame)

        # Cycle energy = frame-to-frame motion.  This is robust to background
        # model quality, captures the cyclic motion signal from the first frame,
        # and naturally produces valleys when the worker returns to home posture.
        if self._prev_frame is not None:
            energy = float(np.abs(
                frame.astype(np.float32) - self._prev_frame.astype(np.float32)
            ).mean())
        else:
            energy = 0.0
        self._prev_frame = frame

        self._cycle_det.push_energy(energy)
        self._frame_buffer.append(frame)
        self._total_frames += 1

    def push_imu_gyro(self, gyro_xyz: np.ndarray, dt: Optional[float] = None) -> None:
        """
        Push raw gyroscope reading (rad/s).  Call at IMU rate, not camera rate.
        """
        if self._imu_int is not None:
            self._imu_int.push_gyro(gyro_xyz, dt)

    # ------------------------------------------------------------------
    # Two-pass encoding
    # ------------------------------------------------------------------

    def encode(self) -> None:
        """
        Run the second pass: segment cycles, encode canonicals + deltas,
        write the .ego file.  Call after all frames have been pushed.
        """
        seg = self._cycle_det.segment()
        bg  = self._bg_model.get_background()

        self._writer = BitstreamWriter(
            self._output_path,
            total_frames=self._total_frames,
            fps=self.fps,
            width=self.width,
            height=self.height,
            has_imu=self.has_imu,
        )

        # ── Metadata ─────────────────────────────────────────────────
        meta = dict(
            total_frames   = self._total_frames,
            fps            = self.fps,
            n_cycles       = len(seg.cycles),
            n_canonicals   = len(seg.canonical_indices),
            quality        = self.quality,
        )
        self._writer.write_chunk(ChunkType.METADATA,
                                  json.dumps(meta).encode(), compress=False)

        # ── Background keyframe ───────────────────────────────────────
        bg_jpeg = encode_background_jpeg(bg, quality=80)
        self._writer.write_chunk(ChunkType.BACKGROUND, bg_jpeg, compress=False)

        # ── IMU stream ────────────────────────────────────────────────
        if self.has_imu and self._imu_quats:
            imu_bytes = pack_imu_quats(np.array(self._imu_quats))
            self._writer.write_chunk(ChunkType.IMU_BLOCK, imu_bytes)

        # ── Canonical cycles ──────────────────────────────────────────
        # For each canonical: encode full foreground residual sequence vs BG
        canonical_frame_seqs: List[np.ndarray] = []   # one per canonical

        for canon_idx in seg.canonical_indices:
            cycle  = seg.cycles[canon_idx]
            frames = self._frame_buffer[cycle.start_frame:cycle.end_frame]
            encoded_seq = self._encode_cycle_vs_bg(frames, bg)
            canonical_frame_seqs.append(np.array(frames))
            self._writer.write_chunk(ChunkType.CYCLE_CANON, encoded_seq)

        # ── Per-cycle delta or clone records ─────────────────────────
        for i, cycle in enumerate(seg.cycles):
            if cycle.is_canonical:
                continue
            frames     = self._frame_buffer[cycle.start_frame:cycle.end_frame]
            canon_idx  = cycle.canonical_idx or 0
            canon_frames = canonical_frame_seqs[canon_idx]

            # Similarity check on background-subtracted frames:
            # background pixels are identical across cycles (static environment)
            # and would inflate MSE due to sensor noise; only foreground matters.
            n_cmp       = min(len(frames), len(canon_frames))
            cmp_frames  = np.array(frames[:n_cmp])
            cmp_canon   = canon_frames[:n_cmp]
            if self._bg_model.is_ready:
                bg_f32  = bg.astype(np.float32)
                sub_a   = np.clip(cmp_frames.astype(np.float32) - bg_f32, -128, 127)
                sub_b   = np.clip(cmp_canon.astype(np.float32)  - bg_f32, -128, 127)
                diff    = sub_a - sub_b
                mse     = float((diff ** 2).mean())
            else:
                mse = cycle_residual_mse(cmp_frames, cmp_canon)

            if mse < self._cycle_det.similarity_threshold:
                # CLONE: store only the 4-byte header (canon_idx + frame_count)
                clone_payload = struct.pack(">HH", canon_idx, len(frames))
                self._writer.write_chunk(ChunkType.FRAME_SKIP, clone_payload, compress=False)
            else:
                delta_bytes = self._encode_cycle_delta(frames, canon_frames, bg, canon_idx)
                self._writer.write_chunk(ChunkType.CYCLE_DELTA, delta_bytes)

        self._writer.close()

    # ------------------------------------------------------------------
    # Convenience: encode from a video file
    # ------------------------------------------------------------------

    @classmethod
    def from_video(cls, input_path: str, output_path: str,
                   quality: int = 25, **kwargs) -> "EgoEncoder":
        """Encode a video file to .ego format."""
        cap = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        enc = cls(output_path, fps=fps, width=width, height=height,
                  quality=quality, **kwargs)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            enc.push_frame(frame)
        cap.release()
        enc.encode()
        return enc

    @property
    def bytes_written(self) -> int:
        return self._writer.bytes_written if self._writer else 0

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------

    def _encode_cycle_vs_bg(self, frames: List[np.ndarray],
                             bg: np.ndarray) -> bytes:
        """
        Encode a list of frames as residuals vs the background plate.
        Returns a single bytes blob: [4B frame_count][frame0_bytes][frame1_bytes]...
        Each frame block is prefixed by a 4B length.
        """
        parts = [struct.pack(">I", len(frames))]
        for frame in frames:
            fg_mask  = self._bg_model.get_foreground_mask(frame)
            residual = frame.astype(np.int16) - bg.astype(np.int16)
            encoded  = encode_residual(residual, fg_mask, quality=self.quality)
            parts.append(struct.pack(">I", len(encoded)))
            parts.append(encoded)
        return b"".join(parts)

    def _encode_cycle_delta(self, frames: List[np.ndarray],
                             canon_frames: np.ndarray,
                             bg: np.ndarray,
                             canon_idx: int = 0) -> bytes:
        """
        Encode per-frame delta: (current_frame - canonical_frame) as residual.
        Frame count mismatch handled by clamping to shorter length.
        Layout: [2B canon_idx][4B frame_count][frame blobs...]
        """
        n     = min(len(frames), len(canon_frames))
        parts = [struct.pack(">HI", canon_idx, n)]
        for i in range(n):
            delta    = frames[i].astype(np.int16) - canon_frames[i].astype(np.int16)
            fg_mask  = self._bg_model.get_foreground_mask(frames[i])
            encoded  = encode_residual(delta, fg_mask, quality=self.quality)
            parts.append(struct.pack(">I", len(encoded)))
            parts.append(encoded)
        return b"".join(parts)
