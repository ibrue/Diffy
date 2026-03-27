/// Codec round-trip tests — verify encode → decode recovers input within
/// acceptable quantization error.
///
/// Run with: cargo test --lib

#[cfg(test)]
mod tests {
    use crate::residual_codec::{
        dct8x8, idct8x8, make_qt, encode_channel, decode_channel,
        rle_encode, rle_decode, zlib_compress, zlib_decompress,
        decode_residual_payload, test_encode_residual, test_ycbcr_roundtrip,
    };

    // ── DCT round-trip ────────────────────────────────────────────────────────

    #[test]
    fn dct_roundtrip_constant() {
        let val = 42.0f32;
        let mut block = [val; 64];
        let orig = block;
        dct8x8(&mut block);
        idct8x8(&mut block);
        let max_err = orig.iter().zip(block.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 0.01, "DCT constant round-trip error: {:.6}", max_err);
    }

    #[test]
    fn dct_roundtrip_random() {
        let block_orig: [f32; 64] = core::array::from_fn(|i| {
            ((i as f32 * 2.7183 + 1.4142) * 31.0).sin() * 50.0
        });
        let mut block = block_orig;
        dct8x8(&mut block);
        idct8x8(&mut block);
        let mse: f32 = block_orig.iter().zip(block.iter())
            .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / 64.0;
        let psnr = 10.0 * (255.0f32.powi(2) / mse.max(1e-12)).log10();
        println!("dct_roundtrip_random: PSNR = {:.1} dB (want >80)", psnr);
        assert!(psnr > 80.0, "DCT lossless round-trip PSNR too low: {:.1} dB", psnr);
    }

    // ── Quantize + IDCT round-trip ────────────────────────────────────────────

    #[test]
    fn channel_roundtrip_quality80() {
        let qt = make_qt(80, true);
        let channel_orig: Vec<f32> = (0..64).map(|i| {
            ((i as f32 * 1.234) * 17.0).sin() * 30.0
        }).collect();
        let coefs = encode_channel(&channel_orig, 8, 8, &qt);
        let decoded = decode_channel(&coefs, 8, 8, &qt);
        let mse: f32 = channel_orig.iter().zip(decoded.iter())
            .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / 64.0;
        let psnr = 10.0 * (255.0f32.powi(2) / mse.max(1e-12)).log10();
        println!("channel_roundtrip q=80: PSNR = {:.1} dB (want >30)", psnr);
        assert!(psnr > 30.0, "channel round-trip PSNR too low: {:.1} dB", psnr);
    }

    #[test]
    fn channel_roundtrip_multi_block() {
        // 4 blocks: 16×8
        let qt = make_qt(75, true);
        let channel_orig: Vec<f32> = (0..128).map(|i| {
            ((i as f32 * 0.7) * 11.0).sin() * 40.0
        }).collect();
        let coefs = encode_channel(&channel_orig, 8, 16, &qt);
        let decoded = decode_channel(&coefs, 8, 16, &qt);
        let mse: f32 = channel_orig.iter().zip(decoded.iter())
            .map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / 128.0;
        let psnr = 10.0 * (255.0f32.powi(2) / mse.max(1e-12)).log10();
        println!("channel_roundtrip multi_block q=75: PSNR = {:.1} dB (want >30)", psnr);
        assert!(psnr > 30.0, "multi-block channel PSNR too low: {:.1} dB", psnr);
    }

    // ── RLE round-trip ────────────────────────────────────────────────────────

    #[test]
    fn rle_roundtrip_sparse() {
        let mut data = vec![0i16; 512];
        data[3] = 127; data[100] = -300; data[200] = 1; data[511] = -1;
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded, data.len());
        assert_eq!(data, decoded, "RLE sparse round-trip failed");
    }

    #[test]
    fn rle_roundtrip_dense() {
        let data: Vec<i16> = (0..200).map(|i| (i as i16 - 100) * 3).collect();
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded, data.len());
        assert_eq!(data, decoded, "RLE dense round-trip failed");
    }

    #[test]
    fn rle_long_zero_run() {
        let data = vec![0i16; 500]; // spans multiple 255-length chains
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded, data.len());
        assert_eq!(data, decoded, "RLE long zero run failed");
    }

    // ── Zlib round-trip ───────────────────────────────────────────────────────

    #[test]
    fn zlib_roundtrip() {
        let data: Vec<u8> = (0..1000).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = zlib_compress(&data);
        let decompressed = zlib_decompress(&compressed);
        assert_eq!(data, decompressed);
        println!("zlib: {} → {} bytes ({:.0}%)", data.len(), compressed.len(),
            compressed.len() as f32 / data.len() as f32 * 100.0);
    }

    // ── YCbCr round-trip ──────────────────────────────────────────────────────

    #[test]
    fn ycbcr_roundtrip() {
        let cases = [
            (255.0f32, 0.0, 0.0),
            (0.0, 255.0, 0.0),
            (0.0, 0.0, 255.0),
            (128.0, 64.0, 192.0),
            (-50.0, 30.0, -10.0),
            (0.0, 0.0, 0.0),
        ];
        for (r, g, b) in cases {
            let (rr, gg, bb) = test_ycbcr_roundtrip(r, g, b);
            let err = ((r - rr).abs() + (g - gg).abs() + (b - bb).abs()) / 3.0;
            assert!(err < 0.01, "YCbCr round-trip error {:.5} for ({}, {}, {})", err, r, g, b);
        }
    }

    // ── Full residual encode/decode round-trip ────────────────────────────────

    #[test]
    fn residual_roundtrip_q50() {
        let h = 16usize; let w = 16usize;
        let residual_orig: Vec<f32> = (0..h * w * 3).map(|i| {
            ((i as f32 * 1.618) * 13.0).sin() * 40.0
        }).collect();
        let encoded = test_encode_residual(&residual_orig, h, w, 50);
        let (decoded, dh, dw) = decode_residual_payload(&encoded).expect("decode failed");
        assert_eq!((dh, dw), (h, w));
        let mse: f32 = residual_orig.iter().zip(decoded.iter())
            .map(|(&o, &d)| (o - d as f32).powi(2)).sum::<f32>() / (h * w * 3) as f32;
        let psnr = 10.0 * (255.0f32.powi(2) / mse.max(1e-12)).log10();
        println!("residual_roundtrip q=50: PSNR = {:.1} dB", psnr);
        assert!(psnr > 25.0, "residual q=50 PSNR too low: {:.1} dB", psnr);
    }

    #[test]
    fn residual_roundtrip_q80() {
        let h = 16usize; let w = 16usize;
        let residual_orig: Vec<f32> = (0..h * w * 3).map(|i| {
            ((i as f32 * 0.9) * 7.0).sin() * 30.0
        }).collect();
        let encoded = test_encode_residual(&residual_orig, h, w, 80);
        let (decoded, _, _) = decode_residual_payload(&encoded).expect("decode failed");
        let mse: f32 = residual_orig.iter().zip(decoded.iter())
            .map(|(&o, &d)| (o - d as f32).powi(2)).sum::<f32>() / (h * w * 3) as f32;
        let psnr = 10.0 * (255.0f32.powi(2) / mse.max(1e-12)).log10();
        println!("residual_roundtrip q=80: PSNR = {:.1} dB (want >35)", psnr);
        assert!(psnr > 35.0, "residual q=80 PSNR too low: {:.1} dB", psnr);
    }

    #[test]
    fn residual_roundtrip_larger() {
        // 64×64 residual to exercise multi-block paths
        let h = 64usize; let w = 64usize;
        let residual_orig: Vec<f32> = (0..h * w * 3).map(|i| {
            ((i as f32 * 0.123) * 9.0).sin() * 25.0
        }).collect();
        let encoded = test_encode_residual(&residual_orig, h, w, 75);
        let (decoded, dh, dw) = decode_residual_payload(&encoded).expect("decode failed");
        assert_eq!((dh, dw), (h, w));
        let mse: f32 = residual_orig.iter().zip(decoded.iter())
            .map(|(&o, &d)| (o - d as f32).powi(2)).sum::<f32>() / (h * w * 3) as f32;
        let psnr = 10.0 * (255.0f32.powi(2) / mse.max(1e-12)).log10();
        println!("residual_roundtrip 64×64 q=75: PSNR = {:.1} dB (want >25)", psnr);
        assert!(psnr > 25.0, "64×64 residual PSNR too low: {:.1} dB", psnr);
    }
}
