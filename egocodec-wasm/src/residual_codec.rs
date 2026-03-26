/// DCT + quantization + RLE + zlib residual codec.

use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;
use flate2::Compression;
use std::io::{Write, Read};

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

/// 2D DCT-II on an 8x8 block using rustdct.
pub fn dct8x8(block: &mut [f32; 64], dct: &mut rustdct::DctPlanner<f32>) {
    // Row DCTs
    let dct8 = dct.plan_dct2(8);
    let mut scratch = vec![0.0f32; dct8.get_scratch_len()];
    for row in 0..8 {
        let start = row * 8;
        dct8.process_dct2_with_scratch(&mut block[start..start + 8], &mut scratch);
    }
    // Transpose
    let mut tmp = [0.0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            tmp[c * 8 + r] = block[r * 8 + c];
        }
    }
    // Column DCTs (on transposed = rows)
    for row in 0..8 {
        let start = row * 8;
        dct8.process_dct2_with_scratch(&mut tmp[start..start + 8], &mut scratch);
    }
    // Transpose back
    for r in 0..8 {
        for c in 0..8 {
            block[r * 8 + c] = tmp[c * 8 + r];
        }
    }
    // Ortho normalization
    let norm = 1.0 / 4.0; // 1/sqrt(8) * 1/sqrt(8) = 1/8, but DCT-II includes factor of 2
    for v in block.iter_mut() {
        *v *= norm;
    }
}

/// 2D IDCT-III on an 8x8 block.
pub fn idct8x8(block: &mut [f32; 64], dct: &mut rustdct::DctPlanner<f32>) {
    let idct8 = dct.plan_dct3(8);
    let mut scratch = vec![0.0f32; idct8.get_scratch_len()];
    // Row IDCTs
    for row in 0..8 {
        let start = row * 8;
        idct8.process_dct3_with_scratch(&mut block[start..start + 8], &mut scratch);
    }
    // Transpose
    let mut tmp = [0.0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            tmp[c * 8 + r] = block[r * 8 + c];
        }
    }
    // Column IDCTs
    for row in 0..8 {
        let start = row * 8;
        idct8.process_dct3_with_scratch(&mut tmp[start..start + 8], &mut scratch);
    }
    // Transpose back
    for r in 0..8 {
        for c in 0..8 {
            block[r * 8 + c] = tmp[c * 8 + r];
        }
    }
    let norm = 1.0 / 4.0;
    for v in block.iter_mut() {
        *v *= norm;
    }
}

/// RLE encode i8 data. Format: [value:i8][run:u8] pairs.
pub fn rle_encode(data: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0 {
            let mut run: u8 = 0;
            while i < data.len() && data[i] == 0 && run < 255 {
                run += 1;
                i += 1;
            }
            out.push(0u8); // value as u8 (0 for zero)
            out.push(run);
        } else {
            out.push(data[i] as u8);
            out.push(1u8);
            i += 1;
        }
    }
    out
}

/// Compress bytes with zlib.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

/// Decompress zlib bytes.
pub fn zlib_decompress(data: &[u8]) -> Vec<u8> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).unwrap();
    out
}

/// Encode a single-channel residual (H8 x W8 float, multiples of 8) to quantized i8 coefficients.
pub fn encode_channel(
    channel: &[f32], // H8*W8 row-major
    h8: usize,
    w8: usize,
    qt: &[f32; 64],
) -> Vec<i8> {
    let mut planner = rustdct::DctPlanner::new();
    let bh = h8 / 8;
    let bw = w8 / 8;
    let mut result = vec![0i8; h8 * w8];

    for by in 0..bh {
        for bx in 0..bw {
            let mut block = [0.0f32; 64];
            for r in 0..8 {
                for c in 0..8 {
                    block[r * 8 + c] = channel[(by * 8 + r) * w8 + bx * 8 + c];
                }
            }
            dct8x8(&mut block, &mut planner);
            // Quantize
            for i in 0..64 {
                block[i] = (block[i] / qt[i]).round().clamp(-127.0, 127.0);
            }
            // Write back
            for r in 0..8 {
                for c in 0..8 {
                    result[(by * 8 + r) * w8 + bx * 8 + c] = block[r * 8 + c] as i8;
                }
            }
        }
    }
    result
}

/// Encode a full H×W×3 residual frame to compressed bytes.
/// residual: i16 values as f32, H*W*3 row-major, channels interleaved.
pub fn encode_residual(
    residual: &[f32],  // H*W*3
    fg_mask: Option<&[bool]>,  // H*W
    h: usize, w: usize,
    quality: u8,
) -> Vec<u8> {
    let h8 = ((h + 7) / 8) * 8;
    let w8 = ((w + 7) / 8) * 8;

    let qt_luma = make_qt(quality, true);
    let qt_chroma = make_qt(quality, false);

    // Separate channels, apply mask, pad to 8x multiples
    let mut channels_i8 = Vec::with_capacity(3);
    for ch in 0..3 {
        let mut padded = vec![0.0f32; h8 * w8];
        for y in 0..h {
            for x in 0..w {
                let val = residual[(y * w + x) * 3 + ch];
                let masked = match fg_mask {
                    Some(mask) if !mask[y * w + x] => 0.0,
                    _ => val,
                };
                padded[y * w8 + x] = masked;
            }
        }
        let qt = if ch == 0 { &qt_luma } else { &qt_chroma };
        let encoded = encode_channel(&padded, h8, w8, qt);
        channels_i8.push(encoded);
    }

    // Interleave H8*W8*3
    let total = h8 * w8 * 3;
    let mut coefs = vec![0i8; total];
    for y in 0..h8 {
        for x in 0..w8 {
            for ch in 0..3 {
                coefs[(y * w8 + x) * 3 + ch] = channels_i8[ch][y * w8 + x];
            }
        }
    }

    let rle = rle_encode(&coefs);
    let compressed = zlib_compress(&rle);

    // Header: [2B H][2B W][1B quality]
    let mut out = Vec::with_capacity(5 + compressed.len());
    out.extend_from_slice(&(h as u16).to_be_bytes());
    out.extend_from_slice(&(w as u16).to_be_bytes());
    out.push(quality);
    out.extend_from_slice(&compressed);
    out
}
