//! 2D Gaussian Splatting for Diffy background representation.
//!
//! Fits a set of 2D Gaussians to an RGB image, producing a compact
//! representation that can be rendered back to pixels.
//!
//! Serialization format (matches Python `GaussianSplatModel.to_bytes`):
//!   [2B u16 BE] width
//!   [2B u16 BE] height
//!   [4B u32 BE] n_splats
//!   [N * 9 * 4B] float32 array: x, y, sx, sy, rot, r, g, b, opacity
//!                 (native / little-endian floats, matching numpy `.tobytes()`)

/// A 2D Gaussian Splatting model.
pub struct GaussianSplatModel {
    pub width: u16,
    pub height: u16,
    pub n_splats: usize,
    pub positions: Vec<[f32; 2]>,  // (x, y)
    pub scales: Vec<[f32; 2]>,    // (sx, sy)
    pub rotations: Vec<f32>,      // rotation angle in radians
    pub colours: Vec<[f32; 3]>,   // (r, g, b) in 0–255
    pub opacities: Vec<f32>,      // 0–1
}

impl GaussianSplatModel {
    /// Fit Gaussian splats to an RGB image.
    ///
    /// `background` — flat RGB bytes (H * W * 3).
    /// `width`, `height` — image dimensions.
    /// `quality` — 0–100 knob controlling iteration count.
    pub fn fit(background: &[u8], width: usize, height: usize, quality: u8) -> Self {
        assert_eq!(background.len(), width * height * 3);

        // ── Determine hyper-parameters ──────────────────────────────────
        let area = height * width;
        let n_splats = (area / 200).max(50).min(2000);
        let iterations = ((40 * quality as usize) / 50).max(20).min(200);

        // ── Downsample factor ───────────────────────────────────────────
        let ds = (width.min(height) / 64).max(1);
        let ds_w = width / ds;
        let ds_h = height / ds;

        // Build downsampled target (simple box average).
        let target = downsample_rgb(background, width, height, ds, ds_w, ds_h);

        // ── Grid-initialise splats ──────────────────────────────────────
        let cols = (n_splats as f32).sqrt().ceil() as usize;
        let rows = ((n_splats as f32) / cols as f32).ceil() as usize;

        let mut positions: Vec<[f32; 2]> = Vec::with_capacity(n_splats);
        let mut scales: Vec<[f32; 2]> = Vec::with_capacity(n_splats);
        let mut rotations: Vec<f32> = Vec::with_capacity(n_splats);
        let mut colours: Vec<[f32; 3]> = Vec::with_capacity(n_splats);
        let mut opacities: Vec<f32> = Vec::with_capacity(n_splats);

        let cell_w = width as f32 / cols as f32;
        let cell_h = height as f32 / rows as f32;

        let mut count = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                if count >= n_splats {
                    break;
                }
                let x = (c as f32 + 0.5) * cell_w;
                let y = (r as f32 + 0.5) * cell_h;

                // Sample colour from the original image at this position.
                let ix = (x as usize).min(width - 1);
                let iy = (y as usize).min(height - 1);
                let idx = (iy * width + ix) * 3;
                let cr = background[idx] as f32;
                let cg = background[idx + 1] as f32;
                let cb = background[idx + 2] as f32;

                positions.push([x, y]);
                scales.push([cell_w * 0.6, cell_h * 0.6]);
                rotations.push(0.0);
                colours.push([cr, cg, cb]);
                opacities.push(0.8);
                count += 1;
            }
        }

        // ── Optimisation loop ───────────────────────────────────────────
        let lr: f32 = 0.01;

        for _iter in 0..iterations {
            // Render at downsampled resolution.
            let rendered = render_to_f32(&positions, &scales, &rotations, &colours, &opacities,
                                         ds_w, ds_h, ds as f32);

            // Compute per-pixel error (target - rendered) and magnitude.
            let npix = ds_w * ds_h;
            let mut err = vec![0.0f32; npix * 3];
            let mut err_mag = vec![0.0f32; npix];
            for i in 0..npix {
                let dr = target[i * 3] - rendered[i * 3];
                let dg = target[i * 3 + 1] - rendered[i * 3 + 1];
                let db = target[i * 3 + 2] - rendered[i * 3 + 2];
                err[i * 3] = dr;
                err[i * 3 + 1] = dg;
                err[i * 3 + 2] = db;
                err_mag[i] = (dr * dr + dg * dg + db * db).sqrt();
            }

            // Update each splat.
            let n = positions.len();
            for s in 0..n {
                // Project position to downsampled coordinates.
                let px = (positions[s][0] / ds as f32).min((ds_w - 1) as f32).max(0.0);
                let py = (positions[s][1] / ds as f32).min((ds_h - 1) as f32).max(0.0);
                let ix = px as usize;
                let iy = py as usize;
                let pi = iy * ds_w + ix;

                let er = err[pi * 3];
                let eg = err[pi * 3 + 1];
                let eb = err[pi * 3 + 2];
                let em = err_mag[pi];
                let opa = opacities[s];

                // Update colour toward error.
                let colour_lr = lr * 40.0 * opa;
                colours[s][0] = (colours[s][0] + colour_lr * er).clamp(0.0, 255.0);
                colours[s][1] = (colours[s][1] + colour_lr * eg).clamp(0.0, 255.0);
                colours[s][2] = (colours[s][2] + colour_lr * eb).clamp(0.0, 255.0);

                // Nudge opacity based on error magnitude.
                let opa_delta = lr * (em / 255.0 - 0.1);
                opacities[s] = (opa + opa_delta).clamp(0.0, 1.0);

                // Nudge position toward higher-error neighbour.
                nudge_position(&mut positions[s], &err_mag, ds_w, ds_h, ds as f32, lr);
            }
        }

        // ── Prune low-opacity splats ────────────────────────────────────
        let mut keep = Vec::new();
        for i in 0..positions.len() {
            if opacities[i] >= 0.01 {
                keep.push(i);
            }
        }
        let positions: Vec<[f32; 2]> = keep.iter().map(|&i| positions[i]).collect();
        let scales: Vec<[f32; 2]> = keep.iter().map(|&i| scales[i]).collect();
        let rotations: Vec<f32> = keep.iter().map(|&i| rotations[i]).collect();
        let colours: Vec<[f32; 3]> = keep.iter().map(|&i| colours[i]).collect();
        let opacities: Vec<f32> = keep.iter().map(|&i| opacities[i]).collect();

        GaussianSplatModel {
            width: width as u16,
            height: height as u16,
            n_splats: positions.len(),
            positions,
            scales,
            rotations,
            colours,
            opacities,
        }
    }

    /// Render the splat model to an RGB image (H * W * 3 bytes).
    pub fn render(&self, width: usize, height: usize) -> Vec<u8> {
        let canvas_f = render_to_f32(
            &self.positions,
            &self.scales,
            &self.rotations,
            &self.colours,
            &self.opacities,
            width,
            height,
            1.0, // no downsampling
        );
        let npix = width * height;
        let mut out = vec![0u8; npix * 3];
        for i in 0..npix * 3 {
            out[i] = canvas_f[i].clamp(0.0, 255.0) as u8;
        }
        out
    }

    /// Serialize to the Diffy splat wire format.
    ///
    /// Header: big-endian width(u16), height(u16), n_splats(u32).
    /// Body:   N * 9 float32 values in native (little-endian on WASM) byte order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_size = 2 + 2 + 4;
        let body_size = self.n_splats * 9 * 4;
        let mut buf = Vec::with_capacity(header_size + body_size);

        // Header — big-endian.
        buf.extend_from_slice(&self.width.to_be_bytes());
        buf.extend_from_slice(&self.height.to_be_bytes());
        buf.extend_from_slice(&(self.n_splats as u32).to_be_bytes());

        // Body — native-endian floats (LE on WASM, matching numpy .tobytes()).
        for i in 0..self.n_splats {
            let floats: [f32; 9] = [
                self.positions[i][0],
                self.positions[i][1],
                self.scales[i][0],
                self.scales[i][1],
                self.rotations[i],
                self.colours[i][0],
                self.colours[i][1],
                self.colours[i][2],
                self.opacities[i],
            ];
            for f in &floats {
                buf.extend_from_slice(&f.to_ne_bytes());
            }
        }

        buf
    }

    /// Compute PSNR against a reference RGB image.
    pub fn psnr(&self, reference: &[u8]) -> f32 {
        let rendered = self.render(self.width as usize, self.height as usize);
        assert_eq!(rendered.len(), reference.len());

        let n = reference.len() as f64;
        let mut mse: f64 = 0.0;
        for i in 0..reference.len() {
            let d = rendered[i] as f64 - reference[i] as f64;
            mse += d * d;
        }
        mse /= n;

        if mse < 1e-10 {
            return 100.0;
        }
        (10.0 * (255.0_f64 * 255.0 / mse).log10()) as f32
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Downsample an RGB image by factor `ds` using box averaging.
fn downsample_rgb(
    src: &[u8],
    w: usize,
    h: usize,
    ds: usize,
    ds_w: usize,
    ds_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ds_w * ds_h * 3];
    let mut counts = vec![0u32; ds_w * ds_h];

    for y in 0..h {
        let dy = y / ds;
        if dy >= ds_h {
            continue;
        }
        for x in 0..w {
            let dx = x / ds;
            if dx >= ds_w {
                continue;
            }
            let si = (y * w + x) * 3;
            let di = dy * ds_w + dx;
            out[di * 3] += src[si] as f32;
            out[di * 3 + 1] += src[si + 1] as f32;
            out[di * 3 + 2] += src[si + 2] as f32;
            counts[di] += 1;
        }
    }

    for i in 0..ds_w * ds_h {
        if counts[i] > 0 {
            let c = counts[i] as f32;
            out[i * 3] /= c;
            out[i * 3 + 1] /= c;
            out[i * 3 + 2] /= c;
        }
    }
    out
}

/// Render splats to a float32 RGB canvas.
///
/// `scale_div` lets us work in downsampled coordinates:
/// splat positions are divided by `scale_div`, scales are divided by `scale_div`.
fn render_to_f32(
    positions: &[[f32; 2]],
    scales: &[[f32; 2]],
    _rotations: &[f32],
    colours: &[[f32; 3]],
    opacities: &[f32],
    canvas_w: usize,
    canvas_h: usize,
    scale_div: f32,
) -> Vec<f32> {
    let npix = canvas_w * canvas_h;
    let mut canvas = vec![0.0f32; npix * 3];
    let mut weights = vec![0.0f32; npix];

    let n = positions.len();
    for i in 0..n {
        let cx = positions[i][0] / scale_div;
        let cy = positions[i][1] / scale_div;
        let sx = (scales[i][0] / scale_div).max(0.5);
        let sy = (scales[i][1] / scale_div).max(0.5);
        let opa = opacities[i];

        if opa < 1e-4 {
            continue;
        }

        // Bounding box: 3 sigma.
        let x0 = ((cx - 3.0 * sx).floor() as isize).max(0) as usize;
        let x1 = ((cx + 3.0 * sx).ceil() as usize + 1).min(canvas_w);
        let y0 = ((cy - 3.0 * sy).floor() as isize).max(0) as usize;
        let y1 = ((cy + 3.0 * sy).ceil() as usize + 1).min(canvas_h);

        let inv_sx2 = 1.0 / (sx * sx);
        let inv_sy2 = 1.0 / (sy * sy);

        let cr = colours[i][0];
        let cg = colours[i][1];
        let cb = colours[i][2];

        for py in y0..y1 {
            let dy = py as f32 - cy;
            let dy2 = dy * dy * inv_sy2;
            let row_off = py * canvas_w;
            for px in x0..x1 {
                let dx = px as f32 - cx;
                let exponent = -0.5 * (dx * dx * inv_sx2 + dy2);
                // Fast reject.
                if exponent < -4.5 {
                    continue;
                }
                let w = fast_exp(exponent) * opa;
                let idx = row_off + px;
                canvas[idx * 3] += w * cr;
                canvas[idx * 3 + 1] += w * cg;
                canvas[idx * 3 + 2] += w * cb;
                weights[idx] += w;
            }
        }
    }

    // Normalise.
    let eps = 1e-6;
    for i in 0..npix {
        if weights[i] > eps {
            let inv_w = 1.0 / weights[i];
            canvas[i * 3] *= inv_w;
            canvas[i * 3 + 1] *= inv_w;
            canvas[i * 3 + 2] *= inv_w;
        }
    }

    canvas
}

/// Nudge a splat position toward the highest-error neighbour in the
/// downsampled error magnitude map.
fn nudge_position(
    pos: &mut [f32; 2],
    err_mag: &[f32],
    ds_w: usize,
    ds_h: usize,
    ds: f32,
    lr: f32,
) {
    let px = (pos[0] / ds).min((ds_w - 1) as f32).max(0.0) as usize;
    let py = (pos[1] / ds).min((ds_h - 1) as f32).max(0.0) as usize;

    let mut best_dx: f32 = 0.0;
    let mut best_dy: f32 = 0.0;
    let mut best_err: f32 = 0.0;

    // Check 3x3 neighbourhood.
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = px as i32 + dx;
            let ny = py as i32 + dy;
            if nx < 0 || ny < 0 || nx >= ds_w as i32 || ny >= ds_h as i32 {
                continue;
            }
            let ni = ny as usize * ds_w + nx as usize;
            if err_mag[ni] > best_err {
                best_err = err_mag[ni];
                best_dx = dx as f32;
                best_dy = dy as f32;
            }
        }
    }

    // Move toward higher error.
    let step = lr * ds * 2.0;
    pos[0] += best_dx * step;
    pos[1] += best_dy * step;
}

/// Fast approximate exp() for small negative exponents.
/// Uses the classic (1 + x/256)^256 trick — good enough for Gaussian weights.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    if x < -10.0 {
        return 0.0;
    }
    // 6th-order polynomial is overkill; use the standard fast path.
    let mut v = 1.0 + x / 256.0;
    // Square 8 times: (1 + x/256)^256.
    v *= v; // ^2
    v *= v; // ^4
    v *= v; // ^8
    v *= v; // ^16
    v *= v; // ^32
    v *= v; // ^64
    v *= v; // ^128
    v *= v; // ^256
    v.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_serialization() {
        // Small solid-red image.
        let w = 32usize;
        let h = 32usize;
        let mut img = vec![0u8; w * h * 3];
        for i in 0..w * h {
            img[i * 3] = 200;
            img[i * 3 + 1] = 50;
            img[i * 3 + 2] = 50;
        }

        let model = GaussianSplatModel::fit(&img, w, h, 30);
        let bytes = model.to_bytes();

        // Check header.
        let rw = u16::from_be_bytes([bytes[0], bytes[1]]);
        let rh = u16::from_be_bytes([bytes[2], bytes[3]]);
        let rn = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(rw, w as u16);
        assert_eq!(rh, h as u16);
        assert_eq!(rn as usize, model.n_splats);

        // Check body length.
        assert_eq!(bytes.len(), 8 + model.n_splats * 9 * 4);
    }

    #[test]
    fn test_render_size() {
        let w = 16usize;
        let h = 16usize;
        let img = vec![128u8; w * h * 3];

        let model = GaussianSplatModel::fit(&img, w, h, 20);
        let rendered = model.render(w, h);
        assert_eq!(rendered.len(), w * h * 3);
    }

    #[test]
    fn test_psnr_perfect() {
        // An image that is all zeros — model should get decent PSNR.
        let w = 8usize;
        let h = 8usize;
        let img = vec![100u8; w * h * 3];
        let model = GaussianSplatModel::fit(&img, w, h, 50);
        let p = model.psnr(&img);
        // Should be at least a reasonable PSNR on a solid colour image.
        assert!(p > 10.0, "PSNR too low: {}", p);
    }

    #[test]
    fn test_fast_exp() {
        // Verify fast_exp is reasonably close to std exp.
        let vals: &[f32] = &[-0.5, -1.0, -2.0, -4.0, 0.0];
        for &x in vals {
            let exact = x.exp();
            let approx = fast_exp(x);
            let rel_err = ((approx - exact) / exact).abs();
            assert!(rel_err < 0.05, "fast_exp({}) = {}, expected ~{}, rel_err={}", x, approx, exact, rel_err);
        }
    }
}
