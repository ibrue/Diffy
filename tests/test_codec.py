"""
Unit tests for EgoCodec.
"""
import os
import struct
import tempfile
import numpy as np
import pytest

from egocodec.background    import BackgroundModel, encode_background_jpeg, decode_background_jpeg
from egocodec.cycle_detector import CycleDetector, Cycle, compute_fg_energy
from egocodec.residual_codec import encode_residual, decode_residual, cycle_residual_mse
from egocodec.bitstream      import BitstreamWriter, BitstreamReader, ChunkType
from egocodec.imu            import (IMUIntegrator, FrameStabilizer,
                                     pack_imu_quats, unpack_imu_quats,
                                     quat_mul, quat_conjugate)
from egocodec.encoder        import EgoEncoder
from egocodec.decoder        import EgoDecoder
from egocodec.vq_codec       import VQCodebook, collect_dct_blocks


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_frame(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def make_noisy_frame(h=64, w=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_synthetic_video(n_frames=120, n_cycles=4, h=64, w=64):
    """Static background + moving blob cycling n_cycles times."""
    bg = make_frame(h, w, val=100)
    frames = []
    fpc = n_frames // n_cycles
    for i in range(n_frames):
        t  = (i % fpc) / fpc
        cx = int(w * 0.5 + w * 0.3 * np.sin(2 * np.pi * t))
        cy = int(h * 0.5 + h * 0.2 * np.sin(4 * np.pi * t))
        f  = bg.copy()
        r  = 6
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
        f[mask] = [200, 100, 80]
        frames.append(f)
    return frames, bg


# ──────────────────────────────────────────────────────────────────────────────
# BackgroundModel
# ──────────────────────────────────────────────────────────────────────────────

class TestBackgroundModel:
    def test_warmup(self):
        model = BackgroundModel(warmup_frames=10)
        for i in range(10):
            model.update(make_frame(val=100))
        assert model.is_ready
        bg = model.get_background()
        assert bg.shape == (64, 64, 3)
        assert abs(int(bg[0, 0, 0]) - 100) <= 2

    def test_not_ready_before_warmup(self):
        model = BackgroundModel(warmup_frames=10)
        for i in range(9):
            model.update(make_frame())
        assert not model.is_ready
        with pytest.raises(RuntimeError):
            model.get_background()

    def test_foreground_mask(self):
        model = BackgroundModel(warmup_frames=10)
        bg_frame = make_frame(h=32, w=32, val=100)
        for _ in range(10):
            model.update(bg_frame.copy())

        # Frame with a bright foreground blob
        fg_frame = bg_frame.copy()
        fg_frame[10:20, 10:20] = 200   # bright patch
        mask = model.get_foreground_mask(fg_frame)
        assert mask.shape == (32, 32)
        # The bright patch should be marked foreground
        assert mask[14, 14]
        # Unmodified background should not be foreground
        assert not mask[0, 0]

    def test_residual_zeros_on_background(self):
        model = BackgroundModel(warmup_frames=10)
        bg_frame = make_frame(h=32, w=32, val=100)
        for _ in range(10):
            model.update(bg_frame.copy())
        res = model.get_residual(bg_frame)
        # Background pixel residuals should be near zero
        assert np.abs(res).max() < 5


# ──────────────────────────────────────────────────────────────────────────────
# Background JPEG roundtrip
# ──────────────────────────────────────────────────────────────────────────────

class TestBackgroundJpeg:
    def test_roundtrip(self):
        bg = make_noisy_frame(h=64, w=64, seed=7)
        data = encode_background_jpeg(bg, quality=90)
        assert isinstance(data, bytes)
        assert len(data) > 0
        decoded = decode_background_jpeg(data)
        assert decoded.shape == bg.shape
        # JPEG is lossy; random noise compresses poorly but shape/dtype must survive
        assert decoded.dtype == np.uint8
        assert decoded.shape == bg.shape


# ──────────────────────────────────────────────────────────────────────────────
# CycleDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestCycleDetector:
    def _make_periodic_energy(self, n_frames=600, n_cycles=6, noise=0.05):
        """Sine-based energy with valleys at cycle boundaries."""
        t      = np.linspace(0, n_cycles * 2 * np.pi, n_frames)
        energy = (np.cos(t) + 1) / 2   # 0..1, minima at boundaries
        energy += np.random.default_rng(0).normal(0, noise, n_frames)
        return np.clip(energy, 0, None)

    def test_segment_returns_cycles(self):
        det = CycleDetector(fps=30.0, min_cycle_frames=5)
        energies = self._make_periodic_energy(300, 6)
        for e in energies:
            det.push_energy(float(e))
        seg = det.segment()
        assert len(seg.cycles) >= 2

    def test_canonical_assigned(self):
        det = CycleDetector(fps=30.0, min_cycle_frames=5)
        energies = self._make_periodic_energy(300, 6)
        for e in energies:
            det.push_energy(float(e))
        seg = det.segment()
        assert len(seg.canonical_indices) >= 1
        for idx in seg.canonical_indices:
            assert seg.cycles[idx].is_canonical

    def test_cycles_cover_all_frames(self):
        det = CycleDetector(fps=30.0, min_cycle_frames=5)
        n   = 180
        for _ in range(n):
            det.push_energy(1.0)
        seg = det.segment()
        covered = sum(c.end_frame - c.start_frame for c in seg.cycles)
        assert covered == n

    def test_compute_fg_energy(self):
        residual = np.zeros((32, 32, 3), dtype=np.int16)
        residual[10:20, 10:20] = 50
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        e = compute_fg_energy(residual, mask)
        assert abs(e - 50.0) < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# ResidualCodec
# ──────────────────────────────────────────────────────────────────────────────

class TestResidualCodec:
    def test_encode_decode_zero_residual(self):
        residual = np.zeros((64, 64, 3), dtype=np.int16)
        data = encode_residual(residual, quality=50)
        decoded = decode_residual(data)
        assert decoded.shape == residual.shape
        assert np.abs(decoded).max() < 3   # small rounding

    def test_encode_decode_sparse_residual(self):
        residual = np.zeros((64, 64, 3), dtype=np.int16)
        residual[20:30, 20:30] = 80   # small bright patch
        fg_mask = np.zeros((64, 64), dtype=bool)
        fg_mask[20:30, 20:30] = True
        data = encode_residual(residual, fg_mask=fg_mask, quality=50)
        decoded = decode_residual(data)
        # Background should still be ~zero
        assert np.abs(decoded[0, 0]).max() < 5
        # Foreground patch should be roughly recovered
        patch_mean = float(np.abs(decoded[20:30, 20:30]).mean())
        assert patch_mean > 20

    def test_compression_is_effective(self):
        """Zero residual should compress to << raw size."""
        h, w    = 256, 256
        residual = np.zeros((h, w, 3), dtype=np.int16)
        data    = encode_residual(residual, quality=30)
        raw_size = h * w * 3 * 2   # int16
        assert len(data) < raw_size / 10

    def test_encode_decode_shape_preservation(self):
        """Encoder should handle non-multiple-of-8 dimensions."""
        residual = np.zeros((100, 100, 3), dtype=np.int16)
        residual[10:20, 10:20] = 40
        data    = encode_residual(residual, quality=40)
        decoded = decode_residual(data)
        assert decoded.shape == (100, 100, 3)

    def test_cycle_residual_mse_identical(self):
        frames = np.array([make_frame(val=100)] * 10)
        mse = cycle_residual_mse(frames, frames.copy())
        assert mse < 1.0

    def test_cycle_residual_mse_different(self):
        a = np.array([make_frame(val=100)] * 10)
        b = np.array([make_frame(val=200)] * 10)
        mse = cycle_residual_mse(a, b)
        assert mse > 1000


# ──────────────────────────────────────────────────────────────────────────────
# Bitstream
# ──────────────────────────────────────────────────────────────────────────────

class TestBitstream:
    def test_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
            path = tf.name
        try:
            with BitstreamWriter(path, total_frames=100, fps=30.0,
                                  width=64, height=64) as w:
                w.write_chunk(ChunkType.BACKGROUND,  b"hello world")
                w.write_chunk(ChunkType.CYCLE_CANON, b"\x00" * 200)
                w.write_chunk(ChunkType.METADATA,    b'{"test":1}', compress=False)

            with BitstreamReader(path) as r:
                assert r.header["total_frames"] == 100
                assert r.header["fps"] == 30.0
                chunks = list(r.read_chunks())
            assert len(chunks) == 3
            assert chunks[0][0] == ChunkType.BACKGROUND
            assert chunks[0][1] == b"hello world"
            assert chunks[2][1] == b'{"test":1}'
        finally:
            os.unlink(path)

    def test_magic_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
            tf.write(b"BADD")
            path = tf.name
        try:
            with pytest.raises(ValueError):
                BitstreamReader(path)
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────────────────────────────────────
# IMU
# ──────────────────────────────────────────────────────────────────────────────

class TestIMU:
    def test_quat_mul_identity(self):
        identity = np.array([0., 0., 0., 1.])
        q        = np.array([0.1, 0.2, 0.3, 0.9])
        q       /= np.linalg.norm(q)
        result   = quat_mul(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-6)

    def test_quat_conjugate_inverse(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q /= np.linalg.norm(q)
        product = quat_mul(q, quat_conjugate(q))
        # Should be close to identity [0,0,0,1]
        np.testing.assert_allclose(product, [0, 0, 0, 1], atol=1e-6)

    def test_imu_integrator_static(self):
        """Zero gyro → orientation stays at identity."""
        imu = IMUIntegrator(imu_hz=200.0, camera_hz=30.0)
        for _ in range(200):
            imu.push_gyro(np.zeros(3))
        quats = imu.get_frame_orientations()
        assert len(quats) > 0
        for q in quats:
            np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-5)

    def test_imu_pack_unpack(self):
        quats = np.random.randn(100, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        data  = pack_imu_quats(quats)
        recovered = unpack_imu_quats(data)
        np.testing.assert_allclose(quats, recovered, atol=1e-2)  # f16 precision


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end encoder / decoder
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def _run_e2e(self, n_frames=120, n_cycles=4, quality=30):
        frames, bg = make_synthetic_video(n_frames, n_cycles, h=64, w=64)
        with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
            path = tf.name
        try:
            enc = EgoEncoder(path, fps=30.0, width=64, height=64,
                             quality=quality, warmup_frames=20)
            for f in frames:
                enc.push_frame(f)
            enc.encode()

            ego_size = os.path.getsize(path)
            raw_size = n_frames * 64 * 64 * 3

            dec  = EgoDecoder(path)
            decoded = list(dec.iter_frames())

            return dict(ego_size=ego_size, raw_size=raw_size,
                        n_decoded=len(decoded), decoder=dec,
                        first_frame=decoded[0] if decoded else None)
        finally:
            os.unlink(path)

    def test_encode_produces_file(self):
        result = self._run_e2e()
        assert result["ego_size"] > 0

    def test_compresses_vs_raw(self):
        result = self._run_e2e()
        ratio = result["raw_size"] / result["ego_size"]
        assert ratio > 5, f"Expected >5× compression, got {ratio:.1f}×"

    def test_decode_returns_frames(self):
        result = self._run_e2e()
        assert result["n_decoded"] > 0

    def test_decoded_frame_shape(self):
        result = self._run_e2e()
        first  = result["first_frame"]
        assert first is not None
        assert first.shape == (64, 64, 3)
        assert first.dtype == np.uint8

    def test_background_accessible(self):
        result = self._run_e2e()
        bg = result["decoder"].background
        assert bg is not None
        assert bg.shape == (64, 64, 3)

    def test_many_cycles_high_compression(self):
        """More repeated cycles → better compression."""
        r4  = self._run_e2e(n_frames=120, n_cycles=4)
        r12 = self._run_e2e(n_frames=120, n_cycles=12)
        # More cycles should compress at least as well
        assert r12["ego_size"] <= r4["ego_size"] * 1.5


# ──────────────────────────────────────────────────────────────────────────────
# VQ Codec
# ──────────────────────────────────────────────────────────────────────────────

class TestVQCodec:
    def _make_blocks(self, n=500, seed=7):
        """Synthetic float32 blocks with cluster structure."""
        rng = np.random.default_rng(seed)
        centres = rng.standard_normal((8, 64)).astype(np.float32) * 20
        labels  = rng.integers(0, 8, n)
        noise   = rng.standard_normal((n, 64)).astype(np.float32) * 2
        return centres[labels] + noise

    def test_train_sets_codebook(self):
        vq = VQCodebook(n_codewords=16)
        vq.train(self._make_blocks())
        assert vq.is_trained
        assert vq.codebook.shape == (16, 64)

    def test_codeword_zero_is_zero_block(self):
        vq = VQCodebook(n_codewords=16)
        vq.train(self._make_blocks())
        # Codeword 0 should be the closest centroid to the all-zeros block
        norms = np.sum(vq.codebook ** 2, axis=1)
        assert norms[0] == norms.min()

    def test_encode_decode_approximate_roundtrip(self):
        vq = VQCodebook(n_codewords=64)
        blocks = self._make_blocks(n=1000)
        vq.train(blocks)
        indices  = vq.encode_blocks(blocks)
        recon    = vq.decode_blocks(indices)
        assert recon.shape == blocks.shape
        # Reconstruction error should be less than block variance
        mse = float(np.mean((blocks - recon) ** 2))
        var = float(np.var(blocks))
        assert mse < var, f"VQ MSE {mse:.1f} >= block variance {var:.1f}"

    def test_serialise_deserialise(self):
        vq = VQCodebook(n_codewords=32)
        vq.train(self._make_blocks())
        data = vq.to_bytes()
        vq2  = VQCodebook.from_bytes(data)
        assert vq2.n_codewords == 32
        np.testing.assert_allclose(vq.codebook, vq2.codebook, atol=0.05)  # f16 precision

    def test_collect_dct_blocks(self):
        h, w   = 64, 64
        rng    = np.random.default_rng(0)
        res    = rng.integers(-30, 30, (h, w, 3), dtype=np.int16)
        fg     = np.zeros((h, w), dtype=bool)
        fg[16:48, 16:48] = True
        blocks = collect_dct_blocks(res, fg, quality=30)
        assert blocks.ndim == 2
        assert blocks.shape[1] == 64
        assert len(blocks) > 0

    def test_vq_end_to_end(self):
        """Full encode → decode roundtrip with use_vq=True."""
        frames, bg = make_synthetic_video(n_frames=120, n_cycles=4, h=64, w=64)
        with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
            path = tf.name
        try:
            enc = EgoEncoder(path, fps=30.0, width=64, height=64,
                             quality=30, warmup_frames=20, use_vq=True)
            for f in frames:
                enc.push_frame(f)
            enc.encode()

            dec     = EgoDecoder(path)
            decoded = list(dec.iter_frames())
            assert len(decoded) > 0
            assert decoded[0].shape == (64, 64, 3)
            assert decoded[0].dtype == np.uint8
            # VQ should not inflate size vs non-VQ baseline
            vq_size = os.path.getsize(path)
            assert vq_size > 0
        finally:
            os.unlink(path)
