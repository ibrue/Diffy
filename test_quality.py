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

# ── Make sure local diffycodec is importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from diffycodec.encoder import DiffyEncoder
from diffycodec.decoder import DiffyDecoder


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic video generator
# ══════════════════════════════════════════════════════════════════════════════

def _make_pov_background(width, height, rng):
    """
    Build a static first-person factory workbench background using PIL.

    Scene layout (top → bottom):
      0  – 35%: concrete wall with mounted tool rail + shadow stripe
      35 – 55%: wall/bench transition zone — metal edging, shadow
      55 – 100%: wooden workbench surface with bolts and a PCB tray

    Returns uint8 H×W×3 BGR array.
    """
    from PIL import Image as _Image, ImageDraw

    img = _Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    wall_h  = int(height * 0.40)
    bench_y = int(height * 0.55)

    # ── Concrete wall: slightly warm gray with subtle vertical streaks ─────
    for x in range(width):
        streak = int(rng.integers(-6, 7))
        for y in range(wall_h):
            v = 178 + streak + int(rng.integers(-2, 3))
            draw.point((x, y), fill=(v, v - 2, v - 4))

    # Horizontal mortar lines
    for y in range(0, wall_h, 28):
        for x in range(0, width, 2):
            draw.point((x, y), fill=(148, 145, 140))

    # Vertical mortar lines (staggered)
    for row in range(wall_h // 28 + 1):
        offset = (row % 2) * 48
        for x in range(offset, width, 96):
            for y in range(row * 28, min((row + 1) * 28, wall_h)):
                draw.point((x, y), fill=(148, 145, 140))

    # ── Pegboard / tool rail on wall ──────────────────────────────────────
    rail_y = int(height * 0.18)
    draw.rectangle([0, rail_y - 6, width - 1, rail_y + 6], fill=(90, 85, 80))
    # Pegboard holes
    for px in range(20, width - 20, 22):
        for py in range(rail_y - 3, rail_y + 4, 6):
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(55, 50, 45))
    # Hanging screwdriver handle
    sd_x = int(width * 0.72)
    draw.rectangle([sd_x - 5, rail_y + 7, sd_x + 5, rail_y + 36],
                   fill=(180, 60, 40))   # red handle
    draw.rectangle([sd_x - 2, rail_y + 37, sd_x + 2, rail_y + 52],
                   fill=(160, 155, 150))  # metal shaft

    # ── Wall-bench transition shadow ──────────────────────────────────────
    for y in range(wall_h, bench_y):
        alpha = (y - wall_h) / (bench_y - wall_h)
        v = int(178 * (1 - alpha) + 110 * alpha)
        draw.line([(0, y), (width - 1, y)], fill=(v, v - 3, v - 6))

    # Bench edge metal strip
    draw.rectangle([0, bench_y - 4, width - 1, bench_y + 2],
                   fill=(120, 118, 112))
    draw.line([(0, bench_y + 3), (width - 1, bench_y + 3)], fill=(80, 78, 74))

    # ── Wooden workbench surface: warm wood grain ─────────────────────────
    for y in range(bench_y + 4, height):
        depth = (y - bench_y) / (height - bench_y)
        base  = int(148 + depth * 12)
        for x in range(width):
            grain_offset = int(8 * np.sin((x + y * 0.3) / 18.0))
            r = min(255, base + grain_offset + int(rng.integers(-3, 4)))
            g = min(255, int(base * 0.78) + grain_offset // 2 + int(rng.integers(-3, 4)))
            b = min(255, int(base * 0.55) + int(rng.integers(-2, 3)))
            draw.point((x, y), fill=(r, g, b))

    # Wood grain lines
    for i in range(12):
        gx = int(rng.integers(0, width))
        gy_start = bench_y + 4
        for y in range(gy_start, height):
            gx += int(rng.integers(-1, 2))
            gx = max(0, min(width - 1, gx))
            v  = 118 + int(rng.integers(-5, 6))
            draw.point((gx, y), fill=(v, int(v * 0.72), int(v * 0.50)))

    # ── Green PCB tray (static component, visible throughout) ─────────────
    tray_x, tray_y = int(width * 0.10), int(height * 0.65)
    tray_w, tray_h = int(width * 0.22), int(height * 0.18)
    draw.rectangle([tray_x, tray_y, tray_x + tray_w, tray_y + tray_h],
                   fill=(34, 85, 34))
    # PCB traces
    for row in range(3):
        for col in range(4):
            tx = tray_x + 8 + col * (tray_w // 4)
            ty = tray_y + 8 + row * (tray_h // 3)
            draw.rectangle([tx, ty, tx + 10, ty + 6], fill=(180, 160, 0))
    # Mounting bolts on bench
    for bx in [int(width * 0.40), int(width * 0.55), int(width * 0.68)]:
        by = int(height * 0.72)
        draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(120, 115, 105))
        draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=(80, 78, 72))

    arr = np.array(img, dtype=np.uint8)
    # Convert RGB→BGR
    return arr[:, :, ::-1]


def make_test_video(width=480, height=270, fps=30,
                    n_cycles=5, frames_per_cycle=100,
                    noise_sigma=2.0):
    """
    Returns (frames, true_background).

    Scene: first-person (POV) view of a factory assembly workbench.
    The camera is head-mounted — it bobs gently with breathing/movement.
    The worker's hands enter from below, perform a cyclic assembly task
    (pick up a component from the PCB tray, move it to a mounting bolt,
    press it down, release, retreat), then exit off the bottom.

    Cycle structure (frames_per_cycle=100):
      0- 9  : hands off-screen, camera settling — pure background
     10-45  : right hand reaches in, grasps component, moves to bolt
     46-70  : press-fit operation (hand presses down, slight camera dip)
     71-90  : hand releases, retreats off bottom edge
     91-99  : both hands gone, camera resettles — cycle boundary

    Camera bob: sinusoidal vertical offset ±3px + lateral ±1.5px
    simulating normal walking/breathing movement.
    """
    from PIL import Image as _Image

    rng_bg = np.random.default_rng(seed=0xD1FF)
    bg_bgr = _make_pov_background(width, height, rng_bg)

    # ── Pre-render component (small orange capacitor block) ────────────────
    comp_w, comp_h = 20, 14
    comp_arr = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
    comp_arr[:, :] = [0, 100, 200]   # orange in BGR
    comp_arr[1:-1, 1:-1] = [0, 120, 230]
    # Silver leads
    comp_arr[:, :3] = [180, 180, 175]
    comp_arr[:, -3:] = [180, 180, 175]

    # Target bolt position (where component gets mounted)
    bolt_cx = int(width * 0.55)
    bolt_cy = int(height * 0.72)

    frames = []

    for frame_idx in range(n_cycles * frames_per_cycle):
        cycle_idx = frame_idx // frames_per_cycle
        phase     = (frame_idx % frames_per_cycle) / frames_per_cycle

        # Camera is head-mounted but mostly static (tripod-assisted rig).
        # Tiny per-frame jitter simulates micro-vibrations without large offsets
        # that would confuse the background model's Welford accumulation.
        img = bg_bgr.copy().astype(np.float32)

        # ── Per-cycle tiny colour variation (lighting shift) ─────────────────
        rng_cyc     = np.random.default_rng(seed=cycle_idx * 31 + 17)
        light_delta = rng_cyc.uniform(-4.0, 4.0)

        # ── Hand / arm animation ──────────────────────────────────────────────
        # phase windows
        entry_start = 0.10
        reach_end   = 0.45
        press_end   = 0.70
        retract_end = 0.90

        # Component start: PCB tray position
        pcb_cx = int(width * 0.21)
        pcb_cy = int(height * 0.74)

        def _lerp(a, b, t):
            return a + (b - a) * np.clip(t, 0, 1)

        def _ease(t):
            return t * t * (3 - 2 * t)  # smoothstep

        # Determine hand position & whether component is being held
        if phase < entry_start or phase >= retract_end:
            hand_visible = False
            comp_held    = False
            comp_visible = True   # resting in tray
            hand_cx = width // 2
            hand_cy = height + 80
        elif phase < reach_end:
            hand_visible = True
            t = _ease((phase - entry_start) / (reach_end - entry_start))
            hand_cx = int(_lerp(width // 2, pcb_cx, t))
            hand_cy = int(_lerp(height + 60, pcb_cy + 10, t))
            comp_held    = t > 0.85   # grab near end of reach
            comp_visible = not comp_held
        elif phase < press_end:
            hand_visible = True
            t = _ease((phase - reach_end) / (press_end - reach_end))
            hand_cx = int(_lerp(pcb_cx, bolt_cx, t))
            hand_cy = int(_lerp(pcb_cy + 10, bolt_cy + 5, t))
            comp_held    = True
            comp_visible = False
            comp_held = True  # keep holding through press
        else:
            hand_visible = True
            t = _ease((phase - press_end) / (retract_end - press_end))
            hand_cx = int(_lerp(bolt_cx, width // 2, t))
            hand_cy = int(_lerp(bolt_cy + 5, height + 60, t))
            comp_held    = False
            comp_visible = False   # component now mounted (permanently placed)

        # Draw component in tray (if not held or placed)
        if comp_visible:
            cy0 = max(0, pcb_cy - comp_h // 2)
            cy1 = min(height, pcb_cy + comp_h // 2)
            cx0 = max(0, pcb_cx - comp_w // 2)
            cx1 = min(width, pcb_cx + comp_w // 2)
            sh0 = cy1 - cy0; sw0 = cx1 - cx0
            if sh0 > 0 and sw0 > 0:
                img[cy0:cy1, cx0:cx1] = comp_arr[:sh0, :sw0].astype(np.float32)

        # Draw held component (moves with hand)
        if comp_held:
            cy0 = max(0, hand_cy - 28 - comp_h // 2)
            cy1 = min(height, cy0 + comp_h)
            cx0 = max(0, hand_cx - comp_w // 2)
            cx1 = min(width, hand_cx + comp_w // 2)
            sh0 = cy1 - cy0; sw0 = cx1 - cx0
            if sh0 > 0 and sw0 > 0:
                img[cy0:cy1, cx0:cx1] = comp_arr[:sh0, :sw0].astype(np.float32)

        # Draw hand (skin-toned tapered shape)
        if hand_visible and hand_cy < height + 20:
            hw, hh = 38, 55   # hand width, height
            # Forearm (thicker, entering from bottom)
            fa_x0 = max(0, hand_cx - 22)
            fa_x1 = min(width, hand_cx + 22)
            fa_y0 = max(0, hand_cy)
            fa_y1 = min(height, hand_cy + hh + 30)
            skin  = np.array([85, 125, 185], dtype=np.float32)  # BGR skin tone
            if fa_y1 > fa_y0:
                img[fa_y0:fa_y1, fa_x0:fa_x1] = skin

            # Hand palm
            hx0 = max(0, hand_cx - hw // 2)
            hx1 = min(width, hand_cx + hw // 2)
            hy0 = max(0, hand_cy - hh // 2)
            hy1 = min(height, hand_cy + hh // 2)
            if hy1 > hy0 and hx1 > hx0:
                img[hy0:hy1, hx0:hx1] = skin * 0.93

            # Knuckle highlights
            for kx_off in [-12, -4, 4, 12]:
                kx = hand_cx + kx_off
                ky = hand_cy - hh // 2 + 5
                if 2 <= kx < width - 2 and 2 <= ky < height - 2:
                    img[ky-2:ky+2, kx-2:kx+2] = skin * 1.15

        # Lighting variation
        img = img + light_delta

        # Per-frame sensor noise
        if noise_sigma > 0:
            noise = np.random.default_rng(frame_idx).normal(0, noise_sigma, img.shape)
            img   = img + noise

        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    return frames, bg_bgr


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def temporal_smoothness(orig_frames, dec_frames, max_p_run=25):
    """
    Measure temporal banding: detect frames where the decoded temporal jump
    is much larger than the original, which causes visible flickers.

    Returns dict with:
      seam_frames   — list of (frame_index, orig_jump, dec_jump, ratio)
      banding_score — fraction of transitions where dec_jump/orig_jump > 2
      worst_seam    — (frame_index, ratio) of the worst frame
      expected_seams— which frames are I-frame positions
    """
    n = min(len(orig_frames), len(dec_frames))
    if n < 2:
        return {'seam_frames': [], 'banding_score': 0.0, 'worst_seam': None}

    seams = []
    for i in range(1, n):
        orig_jump = float(np.mean(np.abs(orig_frames[i].astype(np.float32)
                                        - orig_frames[i-1].astype(np.float32))))
        dec_jump  = float(np.mean(np.abs(dec_frames[i].astype(np.float32)
                                        - dec_frames[i-1].astype(np.float32))))
        ratio = dec_jump / (orig_jump + 0.5)   # +0.5 to avoid div-by-zero on static frames
        seams.append((i, orig_jump, dec_jump, ratio))

    # Expected I-frame positions (every max_p_run+1 frames starting at 0)
    expected = set()
    pos = 0
    while pos < n:
        if pos > 0:
            expected.add(pos)
        pos += max_p_run + 1

    bad = [s for s in seams if s[3] > 2.0]
    banding_score = len(bad) / len(seams) if seams else 0.0
    worst = max(seams, key=lambda s: s[3]) if seams else None

    return {
        'seam_frames':    sorted(bad, key=lambda s: -s[3])[:10],
        'banding_score':  banding_score,
        'worst_seam':     worst,
        'expected_seams': expected,
        'all_seams':      seams,
    }


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
                  out_dir="/tmp/diffy_test", verbose=True,
                  max_p_run=None):
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
    enc_kwargs = {}
    if max_p_run is not None:
        enc_kwargs['max_p_run'] = max_p_run
    enc      = DiffyEncoder(dfy_path, fps=fps, width=width, height=height,
                          quality=quality, warmup_frames=warmup, **enc_kwargs)

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
    decoder = DiffyDecoder(dfy_path)
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

    # ── Temporal smoothness (banding) ─────────────────────────────────────────
    from diffycodec.temporal_codec import encode_cycle_temporal  # just to get max_p_run default
    import inspect
    sig = inspect.signature(encode_cycle_temporal)
    default_p_run = sig.parameters.get('max_p_run')
    default_p_run = default_p_run.default if default_p_run else 25

    ts = temporal_smoothness(orig_frames, dec_frames, max_p_run=default_p_run)
    if verbose:
        worst = ts['worst_seam']
        score = ts['banding_score']
        flag  = ' ← BANDING' if score > 0.05 else ' ✓'
        print(f"[band] banding_score {score:.3f}{flag}  "
              f"(fraction of transitions with dec_jump > 2× orig_jump)")
        if worst:
            expected = 'I-frame' if worst[0] in ts['expected_seams'] else 'natural'
            print(f"       worst seam: frame {worst[0]}  ratio {worst[3]:.1f}×  "
                  f"orig_jump={worst[1]:.1f}  dec_jump={worst[2]:.1f}  [{expected}]")
        if ts['seam_frames']:
            print(f"       top bad frames: {[s[0] for s in ts['seam_frames'][:5]]}")
        # Cycle boundary analysis — check if transitions between cycles are smooth
        fpc = frames_per_cycle
        boundaries = [c * fpc for c in range(1, n_cycles)]
        print(f"[cyc]  cycle boundary jumps (orig → decoded, lower = smoother):")
        for b in boundaries:
            if b < len(ts['all_seams']):
                s = ts['all_seams'][b - 1]   # transition into frame b
                flag2 = ' ← SEAM' if s[3] > 1.5 else ''
                print(f"       frame {b:4d}  orig {s[1]:.1f}  dec {s[2]:.1f}  "
                      f"ratio {s[3]:.2f}×{flag2}")

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


def sweep_prun(n_cycles=5, frames_per_cycle=100, quality=75,
               width=480, height=270, out_dir="/tmp/diffy_test", **kwargs):
    """
    Test different max_p_run values to find the best banding/quality trade-off.
    Lower  → more I-frames → less drift but more banding risk
    Higher → fewer I-frames → more drift but smoother transitions
    """
    prun_values = [10, 15, 25, 40, 60, 90]
    print(f"{'max_p_run':>10}  {'PSNR avg':>9}  {'PSNR min':>9}  "
          f"{'band_score':>10}  {'size KB':>8}")
    print("─" * 60)
    best = None
    for prun in prun_values:
        r = run_roundtrip(quality=quality, n_cycles=n_cycles,
                          frames_per_cycle=frames_per_cycle,
                          width=width, height=height,
                          out_dir=out_dir, verbose=False,
                          max_p_run=prun)
        # Re-decode to measure banding
        from diffycodec.decoder import DiffyDecoder
        orig_frames, _ = make_test_video(width, height, 30, n_cycles, frames_per_cycle)
        dec_frames = list(DiffyDecoder(r['dfy_path']).iter_frames())
        ts = temporal_smoothness(orig_frames, dec_frames, max_p_run=prun)
        band = ts['banding_score']
        print(f"  p_run={prun:3d}   {r['avg_psnr']:7.2f} dB  {r['min_psnr']:7.2f} dB  "
              f"{band:10.4f}  {r['dfy_bytes']//1024:8d} KB"
              + (" ← BANDING" if band > 0.05 else " ✓"))
        if best is None or (band < 0.03 and r['avg_psnr'] > best['avg_psnr']):
            best = {'prun': prun, **r, 'band': band}
    if best:
        print(f"\n  → recommended max_p_run = {best['prun']}  "
              f"(PSNR {best['avg_psnr']:.2f} dB, banding {best['band']:.4f})")


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
    ap.add_argument("--sweep-prun", action="store_true",
                    help="sweep max_p_run values to find best banding/quality trade-off")
    ap.add_argument("--out",       default="/tmp/diffy_test")
    args = ap.parse_args()

    if args.sweep:
        sweep_quality(n_cycles=args.cycles, frames_per_cycle=args.fpc,
                      width=args.width, height=args.height, out_dir=args.out)
    elif args.sweep_prun:
        sweep_prun(n_cycles=args.cycles, frames_per_cycle=args.fpc,
                   width=args.width, height=args.height,
                   quality=args.quality, out_dir=args.out)
    else:
        result = run_roundtrip(quality=args.quality, n_cycles=args.cycles,
                               frames_per_cycle=args.fpc,
                               width=args.width, height=args.height,
                               out_dir=args.out)
        if result['n_decoded'] != result['n_original']:
            print(f"\n⚠  FRAME COUNT MISMATCH: decoded {result['n_decoded']} "
                  f"but original had {result['n_original']}")
