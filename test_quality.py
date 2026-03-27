"""
Round-trip quality test for the Diffy codec.

Generates a synthetic cyclic video, encodes to .dfy, decodes, then
produces a side-by-side comparison image and PSNR/SSIM metrics so
quality problems are immediately visible without needing a browser.

Usage:
    python test_quality.py                  # run with defaults
    python test_quality.py --quality 75     # test specific quality
    python test_quality.py --cycles 6       # more cycles
    python test_quality.py --compare a.dfy  # decode existing file
"""

import argparse
import os
import struct
import sys
import time
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Make sure local egocodec is importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from egocodec.encoder import EgoEncoder
from egocodec.decoder import EgoDecoder


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic video generator
# ══════════════════════════════════════════════════════════════════════════════

def make_test_video(width=480, height=270, fps=30,
                    n_cycles=5, frames_per_cycle=100,
                    noise_sigma=2.0):
    """
    Returns (frames, true_background).

    Scene: gray factory floor + wall.
    Foreground: a robot arm that sweeps in from off-screen left, works,
    and retreats back off-screen — giving clean background frames at
    cycle start/end for reliable cycle detection and background estimation.

    Cycle structure (frames_per_cycle=100):
      0- 9   : robot off-screen (pure background) — cycle boundary
     10-50   : robot enters, sweeps right, works
     51-90   : robot retreats back left
     91-99   : robot off-screen again

    The cycle detector's valley (low energy) lands on the off-screen
    frames, giving clean boundaries.
    """
    # Background: gradient floor / wall
    bg = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        t = y / height
        if t < 0.4:
            v = 200 + (t / 0.4) * 20   # wall: light
        else:
            v = 130 + ((t - 0.4) / 0.6) * 30  # floor: medium gray
        bg[y] = v
    for y in range(int(height * 0.4), height, 20):
        bg[y] = np.clip(bg[y] - 15, 0, 255)
    for x in range(0, width, 30):
        bg[int(height*0.4):, x] = np.clip(bg[int(height*0.4):, x] - 10, 0, 255)
    bg = bg.astype(np.uint8)

    frames = []

    for frame_idx in range(n_cycles * frames_per_cycle):
        cycle_idx = frame_idx // frames_per_cycle
        phase     = (frame_idx % frames_per_cycle) / frames_per_cycle  # 0→1

        img = bg.copy().astype(np.float32)

        # Slight per-cycle variation
        rng         = np.random.default_rng(seed=cycle_idx * 13 + 7)
        speed       = 1.0 + rng.uniform(-0.08, 0.08)
        color_shift = rng.uniform(-6, 6)

        # Smooth in/out envelope: robot enters at phase=0.1, exits at phase=0.9
        # cx goes: off-screen-left → centre → off-screen-left
        entry = 0.10
        exit_ = 0.90
        if phase < entry:
            t_norm = 0.0                           # fully off-screen
        elif phase > exit_:
            t_norm = 0.0
        else:
            t_norm = (phase - entry) / (exit_ - entry)  # 0→1→0

        sweep = np.sin(t_norm * np.pi) * speed     # 0 → peak → 0

        # cx starts off the left edge when sweep=0
        cx = int(-50 + sweep * (width * 0.75 + 50))
        cy = int(height * 0.58)

        # Only draw if at least partially on-screen
        bw, bh = 60, 45
        x0, y0 = cx - bw//2, cy - bh//2
        x1, y1 = cx + bw//2, cy + bh//2
        if x1 > 0 and x0 < width:
            x0c, x1c = max(0, x0), min(width, x1)
            y0c, y1c = max(0, y0), min(height, y1)
            body_color = np.array([40 + color_shift, 45 + color_shift, 52])
            img[y0c:y1c, x0c:x1c] = np.clip(body_color, 0, 255)

            # Arm
            arm_phase = t_norm * np.pi
            ax = int(cx + np.sin(arm_phase - np.pi/2) * 28)
            ay = int(cy - 38 - np.cos(arm_phase - np.pi/2) * 8)
            aw, ah = 10, 32
            ax0c = max(0, ax - aw//2); ax1c = min(width, ax + aw//2)
            ay0c = max(0, ay - ah//2); ay1c = min(height, ay + ah//2)
            img[ay0c:ay1c, ax0c:ax1c] = np.clip(
                [175 + color_shift, 155, 135], 0, 255)

            # LED tip
            gx = np.clip(ax, 3, width-3)
            gy = np.clip(ay - ah//2, 3, height-3)
            img[gy-2:gy+2, gx-2:gx+2] = [245, 215, 75]

        if noise_sigma > 0:
            noise = np.random.default_rng(frame_idx).normal(
                0, noise_sigma, img.shape)
            img = np.clip(img + noise, 0, 255)

        frames.append(img.astype(np.uint8))

    return frames, bg


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (higher = better)."""
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def ssim_channel(a: np.ndarray, b: np.ndarray, k1=0.01, k2=0.03, L=255) -> float:
    """Simplified single-channel SSIM."""
    from scipy.ndimage import uniform_filter
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a  = uniform_filter(a, 11)
    mu_b  = uniform_filter(b, 11)
    mu_ab = mu_a * mu_b
    sig_a = uniform_filter(a * a, 11) - mu_a ** 2
    sig_b = uniform_filter(b * b, 11) - mu_b ** 2
    sig_ab= uniform_filter(a * b, 11) - mu_ab
    num   = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den   = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
    return float(np.mean(num / (den + 1e-10)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return np.mean([ssim_channel(a[:,:,c], b[:,:,c]) for c in range(3)])


# ══════════════════════════════════════════════════════════════════════════════
# Comparison image builder
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_image(orig_frames, dec_frames, metrics, out_path,
                           sample_indices=None):
    """
    Saves a grid: for each sampled frame, two rows:
        top:    original  |  decoded  |  diff×4 amplified
        bottom: PSNR / SSIM labels
    """
    n_total = min(len(orig_frames), len(dec_frames))
    if sample_indices is None:
        # Pick 6 evenly-spaced frames
        sample_indices = [int(i * (n_total - 1) / 5) for i in range(6)]
    sample_indices = [i for i in sample_indices if i < n_total]

    H, W = orig_frames[0].shape[:2]
    thumb_w, thumb_h = min(W, 320), min(H, 180)
    scale_w, scale_h = thumb_w / W, thumb_h / H

    cols      = len(sample_indices)
    panel_w   = thumb_w * 3 + 6          # orig | decoded | diff
    panel_h   = thumb_h + 22             # image + label row
    img_w     = cols * panel_w + (cols + 1) * 4
    img_h     = panel_h + 8 + 80         # panels + summary row
    canvas    = Image.new("RGB", (img_w, img_h), (18, 18, 18))
    draw      = ImageDraw.Draw(canvas)

    def thumb(arr):
        return Image.fromarray(arr).resize((thumb_w, thumb_h), Image.BILINEAR)

    def diff_thumb(a, b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
        d = np.clip(d * 4, 0, 255).astype(np.uint8)
        return Image.fromarray(d).resize((thumb_w, thumb_h), Image.BILINEAR)

    for col, fi in enumerate(sample_indices):
        orig = orig_frames[fi]
        dec  = dec_frames[fi] if fi < len(dec_frames) else np.zeros_like(orig)
        m    = metrics[fi] if fi < len(metrics) else {}

        px = col * (panel_w + 4) + 4
        py = 4

        canvas.paste(thumb(orig), (px,              py))
        canvas.paste(thumb(dec),  (px + thumb_w + 2, py))
        canvas.paste(diff_thumb(orig, dec), (px + (thumb_w + 2) * 2, py))

        label = f"f{fi}  PSNR {m.get('psnr', 0):.1f}dB  SSIM {m.get('ssim', 0):.3f}"
        draw.text((px, py + thumb_h + 4), label, fill=(160, 160, 160))

    # Column headers
    draw.text((8,  img_h - 76), "original", fill=(100, 100, 100))
    draw.text((8 + thumb_w + 2, img_h - 76), "decoded",   fill=(100, 100, 100))
    draw.text((8 + (thumb_w + 2)*2, img_h - 76), "diff ×4",   fill=(100, 100, 100))

    # Summary bar
    valid = [m for m in metrics if m.get('psnr') is not None]
    if valid:
        avg_psnr = np.mean([m['psnr'] for m in valid])
        avg_ssim = np.mean([m['ssim'] for m in valid])
        min_psnr = np.min([m['psnr'] for m in valid])
        summary  = (f"avg PSNR {avg_psnr:.2f} dB   min PSNR {min_psnr:.2f} dB"
                    f"   avg SSIM {avg_ssim:.4f}   {n_total} frames decoded")
        color = (0x7a, 0xaa, 0x7a) if avg_psnr > 35 else \
                (0xaa, 0xaa, 0x7a) if avg_psnr > 30 else (0xaa, 0x60, 0x60)
        draw.text((8, img_h - 56), summary, fill=color)

    canvas.save(out_path)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main round-trip test
# ══════════════════════════════════════════════════════════════════════════════

def run_roundtrip(quality=75, n_cycles=5, frames_per_cycle=100,
                  width=480, height=270, fps=30,
                  out_dir="/tmp/diffy_test", verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    # ── Generate test video ──────────────────────────────────────────────────
    if verbose:
        print(f"[gen]  {n_cycles} cycles × {frames_per_cycle} frames = "
              f"{n_cycles * frames_per_cycle} total  ({width}×{height} @ {fps}fps)")
    orig_frames, bg = make_test_video(width, height, fps, n_cycles, frames_per_cycle)

    # Save background for inspection
    Image.fromarray(bg).save(os.path.join(out_dir, "background_gt.png"))

    # ── Encode ───────────────────────────────────────────────────────────────
    dfy_path = os.path.join(out_dir, f"test_q{quality}.dfy")
    warmup   = max(30, min(200, (n_cycles * frames_per_cycle) // 4))
    enc      = EgoEncoder(dfy_path, fps=fps, width=width, height=height,
                          quality=quality, warmup_frames=warmup)

    t0 = time.time()
    for frame in orig_frames:
        enc.push_frame(frame)
    enc.encode()
    enc_time = time.time() - t0

    raw_bytes  = n_cycles * frames_per_cycle * width * height * 3
    dfy_bytes  = os.path.getsize(dfy_path)
    ratio      = raw_bytes / dfy_bytes

    if verbose:
        print(f"[enc]  {enc_time:.1f}s  →  {dfy_bytes/1024:.0f} KB  "
              f"({ratio:.1f}:1 vs raw)  quality={quality}")

    # ── Decode ───────────────────────────────────────────────────────────────
    t1      = time.time()
    decoder = EgoDecoder(dfy_path)
    dec_frames = list(decoder.iter_frames())
    dec_time = time.time() - t1

    if verbose:
        print(f"[dec]  {dec_time:.1f}s  →  {len(dec_frames)} frames decoded")

    # ── Metrics ──────────────────────────────────────────────────────────────
    n = min(len(orig_frames), len(dec_frames))
    metrics = []
    for i in range(n):
        p = psnr(orig_frames[i], dec_frames[i])
        s = ssim(orig_frames[i], dec_frames[i])
        metrics.append({'frame': i, 'psnr': p, 'ssim': s})

    avg_psnr = np.mean([m['psnr'] for m in metrics])
    avg_ssim = np.mean([m['ssim'] for m in metrics])
    min_psnr = np.min( [m['psnr'] for m in metrics])

    if verbose:
        print(f"[qual] avg PSNR {avg_psnr:.2f} dB  "
              f"min PSNR {min_psnr:.2f} dB  "
              f"avg SSIM {avg_ssim:.4f}")
        # Show per-cycle worst frame
        fpc = frames_per_cycle
        for c in range(n_cycles):
            chunk = metrics[c*fpc:(c+1)*fpc]
            if chunk:
                worst = min(chunk, key=lambda m: m['psnr'])
                print(f"       cycle {c}: worst frame {worst['frame']}  "
                      f"PSNR {worst['psnr']:.1f} dB")

    # ── Comparison image ─────────────────────────────────────────────────────
    img_path = os.path.join(out_dir, f"compare_q{quality}.png")
    make_comparison_image(orig_frames, dec_frames, metrics, img_path)
    if verbose:
        print(f"[img]  comparison saved → {img_path}")

    # ── PSNR-over-time plot (text sparkline) ─────────────────────────────────
    if verbose and n > 0:
        step   = max(1, n // 60)
        values = [metrics[i]['psnr'] for i in range(0, n, step)]
        lo, hi = min(values), max(values)
        span   = hi - lo if hi > lo else 1
        bars   = " ▁▂▃▄▅▆▇█"
        spark  = ''.join(bars[int((v - lo) / span * 8)] for v in values)
        print(f"[psnr] {lo:.0f}dB {spark} {hi:.0f}dB  (per-frame over time)")

    return {
        'avg_psnr': avg_psnr, 'min_psnr': min_psnr, 'avg_ssim': avg_ssim,
        'dfy_bytes': dfy_bytes, 'ratio': ratio,
        'n_decoded': len(dec_frames), 'n_original': len(orig_frames),
        'comparison_img': img_path, 'dfy_path': dfy_path,
    }


def sweep_quality(qualities=(50, 65, 75, 85, 90), **kwargs):
    """Run round-trip at multiple quality levels and compare."""
    print("quality  PSNR(avg)  PSNR(min)  SSIM    size(KB)  ratio")
    print("──────── ───────── ───────── ──────  ──────── ──────")
    for q in qualities:
        r = run_roundtrip(quality=q, verbose=False, **kwargs)
        flag = ""
        if r['n_decoded'] != r['n_original']:
            flag = f"  ⚠ decoded {r['n_decoded']} ≠ {r['n_original']} original"
        print(f"  q={q:3d}   {r['avg_psnr']:6.2f} dB  {r['min_psnr']:6.2f} dB"
              f"  {r['avg_ssim']:.4f}  {r['dfy_bytes']//1024:6d} KB"
              f"  {r['ratio']:5.1f}:1{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality",   type=int, default=75)
    ap.add_argument("--cycles",    type=int, default=5)
    ap.add_argument("--fpc",       type=int, default=100,
                    help="frames per cycle (must be >90 for cycle detector)")
    ap.add_argument("--width",     type=int, default=480)
    ap.add_argument("--height",    type=int, default=270)
    ap.add_argument("--sweep",     action="store_true",
                    help="test multiple quality levels")
    ap.add_argument("--out",       default="/tmp/diffy_test")
    args = ap.parse_args()

    if args.sweep:
        sweep_quality(n_cycles=args.cycles, frames_per_cycle=args.fpc,
                      width=args.width, height=args.height, out_dir=args.out)
    else:
        result = run_roundtrip(quality=args.quality, n_cycles=args.cycles,
                               frames_per_cycle=args.fpc,
                               width=args.width, height=args.height,
                               out_dir=args.out)
        if result['n_decoded'] != result['n_original']:
            print(f"\n⚠  FRAME COUNT MISMATCH: decoded {result['n_decoded']} "
                  f"but original had {result['n_original']}")
