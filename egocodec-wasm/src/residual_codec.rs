/// DCT + quantization + RLE + zlib residual codec.
/// Normalization matches scipy dctn(norm='ortho') — verified against diffy-native.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;

// Standard JPEG luminance quantization table
const JPEG_LUMA_QT: [f32; 64] = [
    16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0,
    12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0,
    14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0,
    14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0,
    18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0,
    24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0, 92.0,
    49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0,
    72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0,
];

const JPEG_CHROMA_QT: [f32; 64] = [
    17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0,
    18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
    24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
];

pub fn make_qt(quality: u8, luma: bool) -> [f32; 64] {
    let base = if luma { &JPEG_LUMA_QT } else { &JPEG_CHROMA_QT };
    let scale = if quality < 50 {
        5000.0 / quality as f32
    } else {
        200.0 - 2.0 * quality as f32
    };
    let mut qt = [0.0f32; 64];
    for i in 0..64 {
        qt[i] = ((base[i] * scale + 50.0) / 100.0).floor().clamp(1.0, 255.0);
    }
    qt
}

// ── DCT normalization ─────────────────────────────────────────────────────────
//
// rustdct plan_dct2: X[k] = Σ x[n]·cos(π·k·(2n+1)/(2N))  (no leading 2×)
// rustdct plan_dct3: y[n] = X[0]/2 + Σ_{k>0} X[k]·cos(...)  (DC halved)
//
// scipy ortho factors: w(0) = 1/√N, w(k>0) = 1/2  per axis.
// IDCT scale: two DCT-III passes each halve DC → total scale = 4/N² = 1/16 for N=8.
fn ortho_norm_factor(k: usize) -> f32 {
    const N: f32 = 8.0;
    if k == 0 { 1.0 / N.sqrt() } else { 0.5 }
}

/// Forward 2D DCT-II, ortho-normalised to match scipy dctn(norm='ortho').
pub fn dct8x8(block: &mut [f32; 64]) {
    let mut planner = rustdct::DctPlanner::<f32>::new();
    let dct8 = planner.plan_dct2(8);
    let mut scratch = vec![0.0f32; dct8.get_scratch_len()];

    // Row DCTs
    for row in 0..8 {
        dct8.process_dct2_with_scratch(&mut block[row * 8..row * 8 + 8], &mut scratch);
    }
    // Transpose
    let mut tmp = [0.0f32; 64];
    for r in 0..8 { for c in 0..8 { tmp[c * 8 + r] = block[r * 8 + c]; } }
    // Column DCTs (on transposed)
    for row in 0..8 {
        dct8.process_dct2_with_scratch(&mut tmp[row * 8..row * 8 + 8], &mut scratch);
    }
    // Transpose back + apply per-k ortho normalisation
    for r in 0..8 {
        let wr = ortho_norm_factor(r);
        for c in 0..8 {
            block[r * 8 + c] = tmp[c * 8 + r] * wr * ortho_norm_factor(c);
        }
    }
}

/// Inverse 2D DCT-III, inverse of dct8x8.
pub fn idct8x8(block: &mut [f32; 64]) {
    // Un-apply normalisation (transpose into tmp for column-first pass)
    let mut tmp = [0.0f32; 64];
    for r in 0..8 {
        let wr = ortho_norm_factor(r);
        for c in 0..8 {
            tmp[c * 8 + r] = block[r * 8 + c] / (wr * ortho_norm_factor(c));
        }
    }

    let mut planner = rustdct::DctPlanner::<f32>::new();
    let idct8 = planner.plan_dct3(8);
    let mut scratch = vec![0.0f32; idct8.get_scratch_len()];

    // Row IDCTs (on transposed)
    for row in 0..8 {
        idct8.process_dct3_with_scratch(&mut tmp[row * 8..row * 8 + 8], &mut scratch);
    }
    // Transpose
    let mut tmp2 = [0.0f32; 64];
    for r in 0..8 { for c in 0..8 { tmp2[c * 8 + r] = tmp[r * 8 + c]; } }
    // Column IDCTs
    for row in 0..8 {
        idct8.process_dct3_with_scratch(&mut tmp2[row * 8..row * 8 + 8], &mut scratch);
    }
    // Scale: 4/N² = 1/16 for N=8  (rustdct DCT-III halves DC each pass)
    let scale = 4.0 / 64.0;
    for r in 0..8 {
        for c in 0..8 {
            block[r * 8 + c] = tmp2[r * 8 + c] * scale;
        }
    }
}

// ── YCbCr conversion ──────────────────────────────────────────────────────────
// BT.601 full-swing (no offset — residuals are signed and centred).
// Input is RGB (from canvas), not BGR.

#[inline]
fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y  =  0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.16874 * r - 0.33126 * g + 0.5 * b;
    let cr =  0.5 * r - 0.41869 * g - 0.08131 * b;
    (y, cb, cr)
}

// ── RLE encode (i16 big-endian, matches Python decoder) ───────────────────────
// Format: [value:>i16][run_length:u8] triplets.
// Zero runs: value=0, run_length=count (≤255, chain for longer).
// Non-zero: value=v, run_length=1.
pub fn rle_encode(data: &[i16]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len() * 3);
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0 {
            let mut run: u8 = 0;
            while i < data.len() && data[i] == 0 && run < 255 {
                run += 1;
                i += 1;
            }
            out.push(0); out.push(0); // 0i16 big-endian
            out.push(run);
        } else {
            let b = data[i].to_be_bytes();
            out.push(b[0]); out.push(b[1]);
            out.push(1u8);
            i += 1;
        }
    }
    out
}

/// Compress bytes with zlib level 9.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(9));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// Encode one channel (H8×W8 f32, multiples of 8) → quantized i16 coefficients.
pub fn encode_channel(channel: &[f32], h8: usize, w8: usize, qt: &[f32; 64]) -> Vec<i16> {
    let mut planner = rustdct::DctPlanner::<f32>::new();
    let bh = h8 / 8;
    let bw = w8 / 8;
    let mut result = vec![0i16; h8 * w8];

    for by in 0..bh {
        for bx in 0..bw {
            let mut block = [0.0f32; 64];
            for r in 0..8 {
                for c in 0..8 {
                    block[r * 8 + c] = channel[(by * 8 + r) * w8 + bx * 8 + c];
                }
            }
            dct8x8(&mut block);
            for i in 0..64 {
                result[(by * 8 + (i / 8)) * w8 + bx * 8 + (i % 8)] =
                    (block[i] / qt[i]).round().clamp(-32767.0, 32767.0) as i16;
            }
        }
    }
    result
}

/// Encode a full H×W×3 residual (RGB f32, interleaved) to compressed bytes.
/// Output format is byte-compatible with the Python `encode_residual` / `decode_residual`.
pub fn encode_residual(
    residual: &[f32],        // H*W*3 RGB row-major
    fg_mask: Option<&[bool]>, // H*W
    h: usize,
    w: usize,
    quality: u8,
) -> Vec<u8> {
    let h8 = ((h + 7) / 8) * 8;
    let w8 = ((w + 7) / 8) * 8;

    let qt_luma   = make_qt(quality, true);
    let qt_chroma = make_qt(quality, false);

    // Convert RGB → YCbCr and separate into three padded channels
    let mut y_ch  = vec![0.0f32; h8 * w8];
    let mut cb_ch = vec![0.0f32; h8 * w8];
    let mut cr_ch = vec![0.0f32; h8 * w8];

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 3;
            let masked = match fg_mask {
                Some(mask) if !mask[py * w + px] => (0.0, 0.0, 0.0),
                _ => {
                    let r = residual[idx];
                    let g = residual[idx + 1];
                    let b = residual[idx + 2];
                    rgb_to_ycbcr(r, g, b)
                }
            };
            let dst = py * w8 + px;
            y_ch[dst]  = masked.0;
            cb_ch[dst] = masked.1;
            cr_ch[dst] = masked.2;
        }
    }

    // Encode each YCbCr channel
    let y_coef  = encode_channel(&y_ch,  h8, w8, &qt_luma);
    let cb_coef = encode_channel(&cb_ch, h8, w8, &qt_chroma);
    let cr_coef = encode_channel(&cr_ch, h8, w8, &qt_chroma);

    // Interleave H8*W8*3 i16 (Y, Cb, Cr order → matches Python ch0=Y, ch1=Cb, ch2=Cr)
    let mut coefs = vec![0i16; h8 * w8 * 3];
    for i in 0..h8 * w8 {
        coefs[i * 3]     = y_coef[i];
        coefs[i * 3 + 1] = cb_coef[i];
        coefs[i * 3 + 2] = cr_coef[i];
    }

    let rle        = rle_encode(&coefs);
    let compressed = zlib_compress(&rle);

    // Header: [2B H][2B W][1B quality]  — same as Python
    let mut out = Vec::with_capacity(5 + compressed.len());
    out.extend_from_slice(&(h as u16).to_be_bytes());
    out.extend_from_slice(&(w as u16).to_be_bytes());
    out.push(quality);
    out.extend_from_slice(&compressed);
    out
}
