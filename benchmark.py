#!/usr/bin/env python3
"""
Diffy codec quality benchmark.

Downloads a real cyclic video, encodes with the Python codec (ground truth),
and measures PSNR/SSIM to establish a quality baseline.

Usage:
    python benchmark.py                        # uses built-in synthetic video
    python benchmark.py --video path/to.mp4   # use a real video
    python benchmark.py --download             # download a test clip from web
    python benchmark.py --quality 75 80 85 90  # sweep qualities

Output: PSNR per frame, mean/min/max, compression ratio, wall-clock encode/decode time.
"""

import argparse
import os
import sys
import time
import tempfile
import struct
import urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from diffycodec.encoder import DiffyEncoder
from diffycodec.decoder import DiffyDecoder


# ── Public-domain test clips (short, Creative Commons or public domain) ───────
TEST_CLIPS = [
    # Big Buck Bunny — highly repetitive scene good for cycle detection
    ("https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
     "ForBiggerBlazes.mp4"),
    # Fallback synthetic
    (None, None),
]


def download_video(url: str, dest: str) -> bool:
    """Download a video file. Returns True on success."""
    if os.path.exists(dest):
        print(f"  [cache] {dest} already downloaded")
        return True
    print(f"  Downloading {url} …", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f" {size // 1024} KB ok")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def load_video_frames(path: str, max_frames: int = 600, scale: float = 1.0):
    """Load frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV (cv2) required to load video files")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if scale != 1.0:
        w = int(w * scale)
        h = int(h * scale)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = frame[:, :, ::-1]  # BGR → RGB
        if scale != 1.0:
            frame_rgb = np.array(Image.fromarray(frame_rgb).resize((w, h), Image.BILINEAR))
        frames.append(frame_rgb.astype(np.uint8))
    cap.release()
    print(f"  Loaded {len(frames)} frames at {w}×{h} @ {fps:.1f} fps")
    return frames, fps, w, h


def make_synthetic_video(n_frames: int = 300, fps: float = 30.0,
                          width: int = 320, height: int = 240, n_cycles: int = 5):
    """Generate a synthetic cyclic video (background + moving foreground)."""
    rng = np.random.default_rng(42)

    # Background: static gradient + texture
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            bg[y, x] = [
                int(160 + 20 * np.sin(x / 20.0)),
                int(140 + 15 * np.cos(y / 15.0)),
                int(120 + 10 * np.sin((x + y) / 30.0)),
            ]

    frames_per_cycle = n_frames // n_cycles
    frames = []
    for c in range(n_cycles):
        for f in range(frames_per_cycle):
            t = f / frames_per_cycle  # 0..1 within cycle
            frame = bg.copy()
            # Simulated moving arm: a colored rectangle
            arm_x = int(width * 0.3 + width * 0.2 * np.sin(2 * np.pi * t))
            arm_y = int(height * 0.4 + height * 0.1 * np.cos(2 * np.pi * t))
            arm_w, arm_h = 40, 20
            x1, x2 = max(0, arm_x), min(width, arm_x + arm_w)
            y1, y2 = max(0, arm_y), min(height, arm_y + arm_h)
            frame[y1:y2, x1:x2] = [200, 80, 60]  # red-ish arm
            # Add small noise per cycle (simulates natural variation)
            noise = rng.integers(-3, 4, size=(y2 - y1, x2 - x1, 3))
            frame[y1:y2, x1:x2] = np.clip(
                frame[y1:y2, x1:x2].astype(int) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)

    print(f"  Synthetic: {len(frames)} frames at {width}×{height} @ {fps:.0f} fps, {n_cycles} cycles")
    return frames, fps, width, height


def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute PSNR in dB between two uint8 frames."""
    mse = float(np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def run_benchmark(frames, fps, width, height, quality: int,
                  label: str = "", warmup_frames: int = 150) -> dict:
    """Encode → decode → measure PSNR. Returns stats dict."""
    with tempfile.NamedTemporaryFile(suffix=".dfy", delete=False) as f:
        dfy_path = f.name

    # ── Encode ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    enc = DiffyEncoder(dfy_path, fps=fps, width=width, height=height,
                       quality=quality, warmup_frames=min(warmup_frames, len(frames) // 3))
    for frame in frames:
        enc.push_frame(frame)
    enc.encode()
    encode_time = time.perf_counter() - t0
    dfy_size = os.path.getsize(dfy_path)

    raw_size = len(frames) * height * width * 3
    ratio = raw_size / max(dfy_size, 1)

    # ── Decode ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    dec = DiffyDecoder(dfy_path)
    decoded_frames = list(dec.iter_frames())
    decode_time = time.perf_counter() - t0

    os.unlink(dfy_path)

    # ── PSNR ──────────────────────────────────────────────────────────────────
    # Only measure on the "warmed-up" portion (skip warmup background frames)
    skip = min(warmup_frames, len(frames) // 3, len(decoded_frames) // 4)
    n_compare = min(len(frames), len(decoded_frames)) - skip
    if n_compare <= 0:
        return {"quality": quality, "psnr_mean": 0, "psnr_min": 0, "error": "no frames to compare"}

    psnrs = []
    for i in range(skip, skip + n_compare):
        orig  = frames[i]
        recon = decoded_frames[i]
        if orig.shape[:2] != recon.shape[:2]:
            continue
        # Handle BGR vs RGB: decoder returns same channel order as input
        psnrs.append(psnr(orig, recon))

    result = {
        "label":       label or f"q={quality}",
        "quality":     quality,
        "n_frames":    len(frames),
        "n_decoded":   len(decoded_frames),
        "dfy_kb":      dfy_size // 1024,
        "raw_kb":      raw_size // 1024,
        "ratio":       ratio,
        "encode_s":    encode_time,
        "decode_s":    decode_time,
        "psnr_mean":   float(np.mean(psnrs)) if psnrs else 0,
        "psnr_min":    float(np.min(psnrs))  if psnrs else 0,
        "psnr_max":    float(np.max(psnrs))  if psnrs else 0,
        "psnr_std":    float(np.std(psnrs))  if psnrs else 0,
    }
    return result


def print_result(r: dict):
    print(f"\n  [{r['label']}]  q={r['quality']}")
    print(f"    PSNR  mean={r['psnr_mean']:.1f} dB  min={r['psnr_min']:.1f}  max={r['psnr_max']:.1f}  std={r['psnr_std']:.1f}")
    print(f"    Size  {r['dfy_kb']} KB  (raw {r['raw_kb']} KB,  {r['ratio']:.0f}× compression)")
    print(f"    Time  encode={r['encode_s']:.1f}s  decode={r['decode_s']:.1f}s")
    print(f"    Frames {r['n_frames']} → decoded {r['n_decoded']}")


def main():
    parser = argparse.ArgumentParser(description="Diffy codec quality benchmark")
    parser.add_argument("--video",    default=None, help="Path to a video file")
    parser.add_argument("--download", action="store_true", help="Download a test clip")
    parser.add_argument("--quality",  nargs="+", type=int, default=[50, 75, 90],
                        help="Quality value(s) to test")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to encode")
    parser.add_argument("--scale",    type=float, default=0.5, help="Resolution scale factor")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic video")
    args = parser.parse_args()

    print("\n===  Diffy Codec Benchmark  ===\n")

    # ── Load video ────────────────────────────────────────────────────────────
    frames, fps, width, height = None, 30.0, 0, 0

    if args.synthetic:
        frames, fps, width, height = make_synthetic_video(
            n_frames=args.max_frames, fps=30.0, width=320, height=240)

    elif args.video:
        print(f"Loading {args.video} …")
        frames, fps, width, height = load_video_frames(
            args.video, max_frames=args.max_frames, scale=args.scale)

    elif args.download:
        clip_path = "/tmp/diffy_bench_clip.mp4"
        url = TEST_CLIPS[0][0]
        if download_video(url, clip_path):
            print("Loading downloaded clip …")
            frames, fps, width, height = load_video_frames(
                clip_path, max_frames=args.max_frames, scale=args.scale)
        else:
            print("Download failed, falling back to synthetic")

    if frames is None:
        frames, fps, width, height = make_synthetic_video(
            n_frames=args.max_frames, fps=30.0, width=320, height=240)

    # ── Run quality sweep ─────────────────────────────────────────────────────
    results = []
    for q in args.quality:
        print(f"\nBenchmarking quality={q} on {len(frames)} frames ({width}×{height}) …")
        try:
            r = run_benchmark(frames, fps, width, height, quality=q,
                              label=f"python q={q}")
            print_result(r)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        print("\n=== Summary ===")
        print(f"  {'quality':>8}  {'PSNR mean':>10}  {'PSNR min':>9}  {'ratio':>7}  {'dfy KB':>7}")
        for r in results:
            print(f"  {r['quality']:>8}  {r['psnr_mean']:>9.1f} dB  {r['psnr_min']:>8.1f} dB  "
                  f"  {r['ratio']:>5.0f}×  {r['dfy_kb']:>7}")

        best = max(results, key=lambda x: x["psnr_mean"])
        print(f"\n  Best quality: {best['psnr_mean']:.1f} dB PSNR at quality={best['quality']}")
        if best["psnr_mean"] < 30:
            print("  ⚠ PSNR < 30 dB — codec quality issue detected")
        elif best["psnr_mean"] < 35:
            print("  ✓ Acceptable quality (30–35 dB)")
        else:
            print("  ✓ Good quality (>35 dB)")


if __name__ == "__main__":
    main()
