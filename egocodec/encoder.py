"""
EgoEncoder: top-level encoder.

Encoding modes
--------------
upload   (default)
    Maximises compression.  Repeat cycles that are near-identical to a
    canonical are stored as 4-byte clone records.  Best for transmission.
    ⚠ Clone records discard per-cycle pixel variation — models trained
      on reconstructed clones see N copies of the same canonical frames.

training
    Preserves per-cycle variation.  Near-identical cycles are stored as
    small temporal deltas vs the canonical instead of clone pointers.
    The deltas capture real-world variation (tool slip, speed jitter,
    illumination drift) that is signal for physical AI models.
    Larger files but every frame is uniquely reconstructable.

Pipeline
--------
  Frame source (mp4 / camera / numpy array)
      │
      ▼
  IMU stabilisation  ─────────────── IMU stream saved to bitstream
      │
      ▼
  Background model warmup (first ~300 frames)
      │  → background JPEG keyframe (written once)
      ▼
  Motion-energy cycle detection
      │
      ▼
  For each canonical cycle:
      └──► temporal I/P coding vs background → CYCLE_CANON chunk
  For each non-canonical cycle:
      ├── phase-align with best canonical
      ├── if upload mode and MSE < threshold → FRAME_SKIP (4-byte clone)
      └── else → temporal delta vs canonical → CYCLE_DELTA chunk
"""

import json
import struct
import numpy as np
from typing import Optional, List
try:
    import cv2
except ImportError:
    cv2 = None  # optional; only needed for from_video() classmethod

from .bitstream      import BitstreamWriter, ChunkType
from .background     import BackgroundModel, encode_background_jpeg
from .cycle_detector import CycleDetector, CycleSegmentation
from .residual_codec import cycle_residual_mse
from .temporal_codec import (encode_cycle_temporal, find_best_phase_offset,
                              encode_frame, FLAG_TEMPORAL, FLAG_BBOX)
from .imu            import IMUIntegrator, FrameStabilizer, pack_imu_quats


class EgoEncoder:
    """
    Parameters
    ----------
    output_path     : path for the .dfy file
    fps             : camera frame rate
    width, height   : frame dimensions
    quality         : residual codec quality (1-100; lower = more compression)
    warmup_frames   : background model warmup length
    has_imu         : whether IMU data will be supplied
    K               : camera intrinsics (3×3); None = auto-estimate
    use_temporal    : use inter-frame temporal prediction within cycles
    use_bbox        : encode only the foreground bounding box per frame
    use_vq          : train a VQ codebook on the first canonical cycle and use it
                      to quantise all canonical cycle DCT blocks (~27× extra reduction)
    """

    def __init__(self,
                 output_path: str,
                 fps: float = 30.0,
                 width: int = 1920,
                 height: int = 1080,
                 quality: int = 25,
                 warmup_frames: int = 300,
                 has_imu: bool = False,
                 K: Optional[np.ndarray] = None,
                 use_temporal: bool = True,
                 use_bbox: bool = True,
                 use_vq: bool = False):
        self.fps          = fps
        self.width        = width
        self.height       = height
        self.quality      = quality
        self.has_imu      = has_imu
        self.use_temporal = use_temporal
        self.use_bbox     = use_bbox
        self.use_vq       = use_vq

        self._bg_model   = BackgroundModel(warmup_frames=warmup_frames)
        self._cycle_det  = CycleDetector(fps=fps)
        self._imu_int    = IMUIntegrator(camera_hz=fps) if has_imu else None
        self._stabilizer = FrameStabilizer(K=K, width=width, height=height) if has_imu else None

        self._frame_buffer: List[np.ndarray] = []
        self._imu_quats:    List[np.ndarray] = []
        self._prev_frame:   Optional[np.ndarray] = None

        self._writer: Optional[BitstreamWriter] = None
        self._output_path = output_path
        self._total_frames = 0

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray,
                   imu_quat: Optional[np.ndarray] = None) -> None:
        """Push one uint8 H×W×C frame."""
        if self.has_imu and imu_quat is not None and self._stabilizer is not None:
            frame = self._stabilizer.warp_frame(frame, imu_quat)
            self._imu_quats.append(imu_quat)

        self._bg_model.update(frame)

        # Motion-energy cycle detection: frame-to-frame mean absolute diff.
        # Produces clear valleys when the worker pauses at the home posture.
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
        if self._imu_int is not None:
            self._imu_int.push_gyro(gyro_xyz, dt)

    # ------------------------------------------------------------------
    # Two-pass encoding
    # ------------------------------------------------------------------

    def encode(self) -> None:
        """Segment cycles, encode, write .dfy file.  Call after all frames pushed."""
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

        # Build temporal cycle order manifest so decoder can reconstruct
        # the original frame sequence (encoder writes canonicals then deltas,
        # but the original video interleaves them).
        non_canon_count = 0
        cycle_map = []
        for i, cycle in enumerate(seg.cycles):
            if cycle.is_canonical:
                cycle_map.append([0, seg.canonical_indices.index(i)])
            else:
                cycle_map.append([1, non_canon_count])
                non_canon_count += 1

        meta = dict(
            total_frames = self._total_frames,
            fps          = self.fps,
            n_cycles     = len(seg.cycles),
            n_canonicals = len(seg.canonical_indices),
            quality      = self.quality,
            use_temporal = self.use_temporal,
            use_bbox     = self.use_bbox,
            cycle_map    = cycle_map,
        )
        self._writer.write_chunk(ChunkType.METADATA,
                                  json.dumps(meta).encode(), compress=False)

        bg_jpeg = encode_background_jpeg(bg, quality=85)
        self._writer.write_chunk(ChunkType.BACKGROUND, bg_jpeg, compress=False)

        # ── VQ codebook training ─────────────────────────────────────────────
        vq_codebook = None
        if self.use_vq and seg.canonical_indices:
            from .vq_codec import VQCodebook, collect_dct_blocks
            first_cycle  = seg.cycles[seg.canonical_indices[0]]
            first_frames = self._frame_buffer[first_cycle.start_frame:first_cycle.end_frame]
            all_blocks   = []
            for frame in first_frames:
                fg_mask  = self._bg_model.get_foreground_mask(frame)
                residual = frame.astype(np.int16) - bg.astype(np.int16)
                blks     = collect_dct_blocks(residual, fg_mask, quality=self.quality)
                if len(blks) > 0:
                    all_blocks.append(blks)
            if all_blocks:
                vq_codebook = VQCodebook(n_codewords=256)
                vq_codebook.train(np.vstack(all_blocks))
                self._writer.write_chunk(ChunkType.CODEBOOK,
                                         vq_codebook.to_bytes(), compress=False)

        if self.has_imu and self._imu_quats:
            self._writer.write_chunk(ChunkType.IMU_BLOCK,
                                      __import__('egocodec.imu', fromlist=['pack_imu_quats'])
                                      .pack_imu_quats(np.array(self._imu_quats)))

        # ── Canonical cycles ────────────────────────────────────────────────
        canonical_frame_seqs: List[np.ndarray] = []

        for canon_list_idx, cycles_idx in enumerate(seg.canonical_indices):
            cycle  = seg.cycles[cycles_idx]
            frames = self._frame_buffer[cycle.start_frame:cycle.end_frame]
            if self.use_temporal:
                encoded = encode_cycle_temporal(
                    frames, bg, self._bg_model,
                    quality=self.quality, use_bbox=self.use_bbox,
                    vq_codebook=vq_codebook)
            else:
                encoded = self._encode_cycle_vs_bg_legacy(frames, bg)
            canonical_frame_seqs.append(np.array(frames))
            self._writer.write_chunk(ChunkType.CYCLE_CANON, encoded)

        # ── Non-canonical cycles ──────────────────────────────────────────────
        bg_f32 = bg.astype(np.float32)

        for cycle in seg.cycles:
            if cycle.is_canonical:
                continue

            frames      = self._frame_buffer[cycle.start_frame:cycle.end_frame]
            canon_idx   = cycle.canonical_idx or 0
            canon_frames = canonical_frame_seqs[canon_idx]

            # Phase-align before delta encoding
            frames_arr  = np.array(frames)
            offset, _   = find_best_phase_offset(
                frames_arr, canon_frames, bg, max_offset=20)

            if offset > 0:
                aligned_canon = canon_frames[offset:]
            elif offset < 0:
                aligned_canon = np.concatenate([
                    canon_frames[-offset:],
                    np.tile(canon_frames[[-1]], (-offset, 1, 1, 1)),
                ])
            else:
                aligned_canon = canon_frames

            delta_bytes = self._encode_cycle_delta(
                frames, aligned_canon, bg, canon_idx)
            self._writer.write_chunk(ChunkType.CYCLE_DELTA, delta_bytes)

        self._writer.close()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_video(cls, input_path: str, output_path: str,
                   quality: int = 25, **kwargs) -> "EgoEncoder":
        if cv2 is None:
            raise ImportError("cv2 (opencv-python) is required for from_video()")
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        enc    = cls(output_path, fps=fps, width=width, height=height,
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_cycle_vs_bg_legacy(self, frames: List[np.ndarray],
                                    bg: np.ndarray) -> bytes:
        """Legacy frame-by-frame bg residual (no temporal prediction)."""
        from .residual_codec import encode_residual
        parts = [struct.pack(">IB", len(frames), 0x00)]   # 0 flags = legacy
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
        Encode per-frame temporal delta vs canonical cycle.
        Uses temporal prediction: delta[k] = frame[k] - canon[k].
        Layout: [2B canon_idx][4B n][per-frame: [4B size][payload]]
        """
        n     = min(len(frames), len(canon_frames))
        parts = [struct.pack(">HI", canon_idx, n)]
        for i in range(n):
            delta   = frames[i].astype(np.int16) - canon_frames[i].astype(np.int16)
            fg_mask = self._bg_model.get_foreground_mask(frames[i])
            encoded = encode_frame(delta, fg_mask,
                                   quality=self.quality, use_bbox=self.use_bbox)
            parts.append(struct.pack(">I", len(encoded)))
            parts.append(encoded)
        return b"".join(parts)
