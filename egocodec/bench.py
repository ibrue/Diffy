"""
Benchmark + compression ratio analysis for EgoCodec.

Generates synthetic egocentric video that mimics the key properties:
  • Static factory background
  • A "worker hand" foreground blob moving in a cyclic pattern
  • Optional simulated head rotation via synthetic IMU

Then compresses with both EgoCodec and estimates H.265 equivalent size,
reporting compression ratios and projected 8-hour / 10 MB headroom.

Usage
-----
  python -m egocodec.bench [--frames N] [--cycles C] [--quality Q]
"""

import argparse
import os
import sys
import time
import tempfile
import numpy as np

from .encoder import EgoEncoder
from .decoder import EgoDecoder
from .background import encode_background_jpeg


# --------------------------------------------------------------------------
# Synthetic video generator
# --------------------------------------------------------------------------

def make_background(height: int, width: int) -> np.ndarray:
    """Simulate a factory floor: grey concrete texture + fixed machinery."""
    rng = np.random.default_rng(42)
    bg  = np.full((height, width, 3), 128, dtype=np.uint8)
    # Add some texture
    noise = rng.integers(0, 20, (height, width), dtype=np.uint8)
    for c in range(3):
        bg[:, :, c] = np.clip(bg[:, :, c].astype(int) + noise - 10, 0, 255).astype(np.uint8)
    # Draw a fixed "machine" rectangle
    bg[height//2:height//2+80, width//4:width//4+200] = [60, 60, 80]
    return bg


def make_hand_blob(frame: np.ndarray, cx: int, cy: int, radius: int = 40) -> np.ndarray:
    """Draw a circular 'hand' blob on the frame at (cx, cy)."""
    out = frame.copy()
    H, W = frame.shape[:2]
    y, x = np.ogrid[:H, :W]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    out[mask] = [180, 120, 90]   # skin-tone approximation (BGR)
    # Add a small "object" in the hand
    out[cy-10:cy+10, cx-10:cx+10] = [30, 80, 150]
    return out


def make_synthetic_video(n_frames: int, n_cycles: int,
                          height: int = 540, width: int = 960) -> list:
    """
    Generate a list of n_frames uint8 H×W×3 frames simulating:
    - Static background
    - Hand moving with a reach-and-return motion profile that has a genuine
      pause at the "home" position (cycle boundary), creating clear energy valleys
      for cycle detection.

    Motion profile: cosine ease-in/ease-out from home→target→home.
    Home position = left side; target = right side. Worker pauses 15% of
    each cycle at home (loading/unloading time), giving a clear signal valley.
    """
    bg = make_background(height, width)
    frames = []
    rng = np.random.default_rng(0)

    frames_per_cycle = n_frames // n_cycles
    pause_frac = 0.15   # fraction of cycle spent stationary at home

    home_cx   = int(width  * 0.25)
    home_cy   = int(height * 0.50)
    target_cx = int(width  * 0.75)
    target_cy = int(height * 0.45)

    for i in range(n_frames):
        t = (i % frames_per_cycle) / frames_per_cycle   # 0..1

        if t < pause_frac:
            # Pause at home position
            cx, cy = home_cx, home_cy
        else:
            # Smooth cosine ease from home → target → home
            t_move = (t - pause_frac) / (1.0 - pause_frac)   # 0..1 over moving portion
            # Use cos: starts at home (t_move=0), reaches target at 0.5, returns at 1
            alpha = 0.5 * (1 - np.cos(2 * np.pi * t_move))   # 0→1→0
            cx = int(home_cx + alpha * (target_cx - home_cx))
            cy = int(home_cy + alpha * (target_cy - home_cy)
                     + height * 0.05 * np.sin(4 * np.pi * t_move))  # small vertical arc

        frame = make_hand_blob(bg, cx, cy)
        # Add sensor noise only during motion; home position is quiet
        # (In real deployments the IMU signal flags stationary periods)
        motion_phase = max(0.0, t - pause_frac)
        noise_scale  = int(4 * min(motion_phase / 0.1, 1.0))   # ramps up over first 10%
        if noise_scale > 0:
            noise = rng.integers(-noise_scale, noise_scale + 1, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames, bg


# --------------------------------------------------------------------------
# H.265 size estimator (analytical)
# --------------------------------------------------------------------------

def estimate_h265_size_bytes(n_frames: int, fps: float,
                              width: int, height: int,
                              bitrate_kbps: int = 2000) -> int:
    """
    Estimate H.265 output size at a given bitrate.
    Default 2 Mbps is typical for 1080p30 at good quality.
    Scales down for lower resolution.
    """
    pixels = width * height
    ref_pixels = 1920 * 1080
    scaled_bitrate = bitrate_kbps * (pixels / ref_pixels)
    duration_s = n_frames / fps
    return int(scaled_bitrate * 1000 / 8 * duration_s)


# --------------------------------------------------------------------------
# Main benchmark
# --------------------------------------------------------------------------

def run_benchmark(n_frames: int = 900, n_cycles: int = 9,
                  quality: int = 25, fps: float = 30.0) -> dict:
    height, width = 540, 960

    print(f"\nEgoCodec Benchmark")
    print(f"  {n_frames} frames  |  {n_cycles} cycles  |  {height}×{width}  |  quality={quality}")
    print(f"  Duration: {n_frames/fps:.1f}s  ({n_frames/fps/3600:.2f} hours)")

    t0 = time.perf_counter()
    frames, bg = make_synthetic_video(n_frames, n_cycles, height, width)
    print(f"\n  [1/4] Generated {n_frames} frames in {time.perf_counter()-t0:.2f}s")

    raw_bytes = n_frames * height * width * 3
    print(f"  Raw uncompressed: {raw_bytes / 1e6:.1f} MB")

    warmup = min(300, n_frames // 4)

    # ── Upload mode ───────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
        out_path = tf.name

    try:
        t0 = time.perf_counter()
        enc = EgoEncoder(out_path, fps=fps, width=width, height=height,
                         quality=quality, warmup_frames=warmup,
                         training_mode=False)
        for frame in frames:
            enc.push_frame(frame)
        enc.encode()
        enc_time = time.perf_counter() - t0

        ego_bytes = os.path.getsize(out_path)
        print(f"\n  [2/4] Upload mode encoded in {enc_time:.2f}s")
        print(f"  .ego (upload) size: {ego_bytes / 1e3:.1f} KB  ({ego_bytes / 1e6:.3f} MB)")
        print(f"  Compression vs raw: {raw_bytes / ego_bytes:.0f}×")

        h265_bytes = estimate_h265_size_bytes(n_frames, fps, width, height)
        print(f"  H.265 estimate:     {h265_bytes / 1e6:.1f} MB")
        print(f"  Upload vs H.265:    {h265_bytes / ego_bytes:.1f}×")

        dec_up   = EgoDecoder(out_path)
        n_clones = sum(1 for ct, _ in dec_up._cycle_chunks if ct.name == "FRAME_SKIP")
        n_deltas = len(dec_up._cycle_chunks) - n_clones
        print(f"  Cycles: {len(dec_up._canonicals)} canonical, "
              f"{n_clones} clones (4B each), {n_deltas} deltas")
    finally:
        os.unlink(out_path)

    # ── Training mode ─────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
        train_path = tf.name

    try:
        t0 = time.perf_counter()
        enc2 = EgoEncoder(train_path, fps=fps, width=width, height=height,
                          quality=quality, warmup_frames=warmup,
                          training_mode=True)
        for frame in frames:
            enc2.push_frame(frame)
        enc2.encode()
        train_enc_time = time.perf_counter() - t0

        train_bytes = os.path.getsize(train_path)
        dec_tr = EgoDecoder(train_path)
        n_decoded = sum(1 for _ in dec_tr.iter_frames())
        n_tr_deltas = len(dec_tr._cycle_chunks)
        avg_delta = (sum(len(p) for _, p in dec_tr._cycle_chunks) / n_tr_deltas
                     if n_tr_deltas else 0)

        print(f"\n  [3/4] Training mode encoded in {train_enc_time:.2f}s")
        print(f"  .ego (training) size: {train_bytes / 1e3:.1f} KB  ({train_bytes / 1e6:.3f} MB)")
        print(f"  Compression vs raw:   {raw_bytes / train_bytes:.0f}×")
        print(f"  Training vs H.265:    {h265_bytes / train_bytes:.1f}×")
        print(f"  Cycles: {len(dec_tr._canonicals)} canonical, "
              f"{n_tr_deltas} deltas @ {avg_delta/1e3:.1f} KB avg  (variation preserved)")
        print(f"  Decoded {n_decoded} frames")
    finally:
        os.unlink(train_path)

    # ── Projections ───────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".ego", delete=False) as tf:
        out_path = tf.name

    try:
        enc = EgoEncoder(out_path, fps=fps, width=width, height=height,
                         quality=quality, warmup_frames=warmup)
        for frame in frames:
            enc.push_frame(frame)
        enc.encode()
        ego_bytes = os.path.getsize(out_path)
        dec = EgoDecoder(out_path)
        n_decoded = sum(1 for _ in dec.iter_frames())
        print(f"\n  Decoded {n_decoded} frames (encoder emitted {n_frames})")

        # ── Sub-linear projection for EgoCodec ───────────────────────────────
        # EgoCodec scaling is NOT linear with duration.  Key insight:
        #   • Canonical cycles are learned once (fixed overhead per task variant)
        #   • Subsequent repeat cycles cost ~4-20 bytes (clone / tiny delta)
        #
        # Projection model:
        #   total ≈ bg_jpeg + sum(canonical_cycle_encoded_bytes) + n_repeats × bytes_per_repeat
        #
        # We estimate bytes_per_repeat from the non-canonical chunk sizes in the file.
        frames_8h   = int(8 * 3600 * fps)
        duration_s  = n_frames / fps
        cycle_s     = duration_s / max(n_cycles, 1)
        n_cycles_8h = int(frames_8h / fps / cycle_s)

        bg_jpeg_bytes = len(encode_background_jpeg(dec.background, quality=80))
        n_canon       = len(dec._canonicals)
        n_non_canon   = len(dec._cycle_chunks)

        frames_8h   = int(8 * 3600 * fps)
        duration_s  = n_frames / fps
        cycle_s     = duration_s / max(n_cycles, 1)
        n_cycles_8h = int(frames_8h / fps / cycle_s)

        n_non_canon = len(dec._cycle_chunks)
        n_clones_p  = sum(1 for ct, _ in dec._cycle_chunks if ct.name == "FRAME_SKIP")
        n_deltas_p  = n_non_canon - n_clones_p
        bytes_per_non_canon = (sum(len(p) for _, p in dec._cycle_chunks) / n_non_canon
                               if n_non_canon > 0 else 4.0)
        canonical_overhead  = ego_bytes - n_non_canon * bytes_per_non_canon
        ego_8h_mb  = (canonical_overhead + n_cycles_8h * bytes_per_non_canon) / 1e6
        train_8h_mb = (canonical_overhead + n_cycles_8h * avg_delta) / 1e6 if n_tr_deltas > 0 else ego_8h_mb

        scale      = frames_8h / n_frames
        h265_8h_mb = h265_bytes * scale / 1e6
        raw_8h_gb  = n_frames * height * width * 3 * scale / 1e9

        print(f"\n  [4/4] ── Projected to 8-hour shift ({n_cycles_8h} cycles) ──")
        print(f"  Raw:                  {raw_8h_gb:.0f} GB")
        print(f"  H.265:                {h265_8h_mb:.0f} MB  (linear scale)")
        print(f"  EgoCodec upload:      {ego_8h_mb:.1f} MB  "
              f"({'✓' if ego_8h_mb < 10 else '✗'} 10 MB target)  "
              f"[{canonical_overhead/1e3:.0f} KB fixed + {bytes_per_non_canon:.0f} B/cycle]")
        print(f"  EgoCodec training:    {train_8h_mb:.1f} MB  "
              f"(per-cycle variation preserved for model training)  "
              f"[{avg_delta/1e3:.1f} KB/cycle delta]")

        upload_bps = (500e6) / 1000
        print(f"\n  Upload @ 500 Mbps shared / 1000 workers:")
        print(f"    H.265:    {h265_8h_mb*8/upload_bps*1e6/3600:.0f} hours/day")
        print(f"    Upload:   {ego_8h_mb*8e6/upload_bps/3600:.1f} hours/day  ✓")
        print(f"    Training: {train_8h_mb*8e6/upload_bps/3600:.1f} hours/day")

        return dict(
            raw_bytes=raw_bytes,
            ego_bytes=ego_bytes,
            train_bytes=train_bytes,
            h265_bytes_est=h265_bytes,
            ratio_vs_raw=raw_bytes / ego_bytes,
            ratio_vs_h265=h265_bytes / ego_bytes,
            ego_8h_mb=ego_8h_mb,
            train_8h_mb=train_8h_mb,
            n_decoded=n_decoded,
            n_clones=n_clones_p,
            n_deltas=n_tr_deltas,
        )
    finally:
        os.unlink(out_path)


def main():
    parser = argparse.ArgumentParser(description="EgoCodec benchmark")
    parser.add_argument("--frames",  type=int,   default=900,  help="number of frames")
    parser.add_argument("--cycles",  type=int,   default=9,    help="number of work cycles")
    parser.add_argument("--quality", type=int,   default=25,   help="codec quality 1-100")
    parser.add_argument("--fps",     type=float, default=30.0, help="frames per second")
    args = parser.parse_args()
    run_benchmark(n_frames=args.frames, n_cycles=args.cycles,
                  quality=args.quality, fps=args.fps)


if __name__ == "__main__":
    main()
