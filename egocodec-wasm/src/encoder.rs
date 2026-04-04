/// EgoEncoder: top-level encode pipeline (Rust WASM version).

extern crate jpeg_encoder;

use crate::background::BackgroundModel;
use crate::bitstream::{BitstreamWriter, ChunkType};
use crate::cycle_detector::{CycleDetector, CycleSegmentation};
use crate::residual_codec::{encode_residual, decode_residual_payload};
use crate::decoder::decode_jpeg_rgb;
use crate::slam::{VisualSLAM, pack_slam_poses, pack_camera_k};
use crate::gaussian_splatting::GaussianSplatModel;

pub struct EgoEncoder {
    fps: f32,
    width: usize,
    height: usize,
    quality: u8,
    bg_model: BackgroundModel,
    cycle_det: CycleDetector,
    slam: VisualSLAM,
    frame_buffer: Vec<Vec<u8>>,  // u8 H*W*3 per frame
    prev_frame: Option<Vec<f32>>,
    total_frames: usize,
}

impl EgoEncoder {
    pub fn new(
        fps: f32,
        width: usize,
        height: usize,
        quality: u8,
        warmup_frames: usize,
    ) -> Self {
        let kf_interval = (warmup_frames / 30).max(1);
        Self {
            fps,
            width,
            height,
            quality,
            bg_model: BackgroundModel::new(warmup_frames, width, height),
            cycle_det: CycleDetector::new(fps),
            slam: VisualSLAM::new(None, width, height, 300, kf_interval),
            frame_buffer: Vec::new(),
            prev_frame: None,
            total_frames: 0,
        }
    }

    /// Push one RGB frame (u8 H*W*3).
    pub fn push_frame(&mut self, frame: &[u8]) {
        self.bg_model.update(frame);
        self.slam.process_frame(frame);

        // Compute motion energy
        let energy = match &self.prev_frame {
            Some(prev) => {
                let mut sum = 0.0f32;
                let len = frame.len();
                for i in 0..len {
                    sum += (frame[i] as f32 - prev[i]).abs();
                }
                sum / len as f32
            }
            None => 0.0,
        };

        self.prev_frame = Some(frame.iter().map(|&v| v as f32).collect());
        self.cycle_det.push_energy(energy);
        self.frame_buffer.push(frame.to_vec());
        self.total_frames += 1;
    }

    /// Encode all frames and return .dfy bytes.
    pub fn encode(&self) -> Vec<u8> {
        let seg = self.cycle_det.segment();
        let bg_raw = self.bg_model.get_background();

        let mut writer = BitstreamWriter::new(
            self.total_frames as u64,
            self.fps,
            self.width as u16,
            self.height as u16,
            false,
        );

        // Background JPEG — round-trip through JPEG so encoder and decoder
        // use the identical background for residual computation.
        let bg_jpeg = encode_background_jpeg(&bg_raw, self.width, self.height);
        // CRITICAL: use the JPEG-decoded background for all residuals.
        // The decoder reconstructs frames as bg_jpeg_decoded + residual, so the
        // encoder must compute residual = frame - bg_jpeg_decoded (not frame - bg_raw).
        let bg = decode_jpeg_rgb(&bg_jpeg)
            .unwrap_or_else(|_| bg_raw.clone());

        // Build cycle_map so the decoder knows playback order
        let mut noncanon_count = 0usize;
        let mut cycle_map: Vec<[usize; 2]> = Vec::with_capacity(seg.cycles.len());
        for (i, cycle) in seg.cycles.iter().enumerate() {
            if cycle.is_canonical {
                let canon_pos = seg.canonical_indices.iter().position(|&c| c == i).unwrap_or(0);
                cycle_map.push([0, canon_pos]);
            } else {
                cycle_map.push([1, noncanon_count]);
                noncanon_count += 1;
            }
        }

        // Check if SLAM succeeded
        let slam_result = self.slam.get_result();
        let use_3d = slam_result.success;

        let meta = serde_json::json!({
            "total_frames": self.total_frames,
            "fps": self.fps,
            "n_cycles": seg.cycles.len(),
            "n_canonicals": seg.canonical_indices.len(),
            "quality": self.quality,
            "use_temporal": true,
            "use_bbox": true,
            "use_3d": use_3d,
            "cycle_map": cycle_map,
        });
        let meta_bytes = serde_json::to_vec(&meta).unwrap();
        writer.write_chunk(ChunkType::Metadata, &meta_bytes, false);

        // Write SLAM data if tracking succeeded
        if use_3d {
            writer.write_chunk(ChunkType::CameraK,
                &pack_camera_k(self.slam.intrinsics()), false);
            writer.write_chunk(ChunkType::SlamPoses,
                &pack_slam_poses(self.slam.poses()), true);
        }

        // Fit 2D Gaussian Splat model for the interactive viewer
        let splat_model = GaussianSplatModel::fit(&bg_raw, self.width, self.height, self.quality);
        writer.write_chunk(ChunkType::SplatModel, &splat_model.to_bytes(), true);

        writer.write_chunk(ChunkType::Background, &bg_jpeg, false);

        // CRITICAL: Use the JPEG-decoded background for all residuals.
        // Both encoder and decoder decode the same JPEG, so residuals stay in sync.
        let bg = decode_jpeg_rgb(&bg_jpeg).unwrap_or(bg);

        // Canonical cycles — encode and immediately decode to get the exact
        // reconstructed frames that the decoder will produce. These decoded frames
        // are used as reference for delta encoding so errors don't stack.
        let mut canonical_decoded_list: Vec<Vec<Vec<u8>>> = Vec::new();
        for &cycle_idx in &seg.canonical_indices {
            let cycle = &seg.cycles[cycle_idx];
            let frames = &self.frame_buffer[cycle.start_frame..cycle.end_frame];
            let (encoded, decoded_frames) = self.encode_cycle_vs_bg_decoded(frames, &bg);
            canonical_decoded_list.push(decoded_frames);
            writer.write_chunk(ChunkType::CycleCanon, &encoded, true);
        }

        // Non-canonical cycles as deltas vs decoded canonical (with phase alignment)
        for cycle in &seg.cycles {
            if cycle.is_canonical { continue; }
            let frames = &self.frame_buffer[cycle.start_frame..cycle.end_frame];
            let decoded_canon = &canonical_decoded_list[cycle.canonical_idx];

            // Phase alignment: find best offset to minimize residual energy
            let offset = Self::find_phase_offset(frames, decoded_canon, &bg, 20);
            let aligned_canon: Vec<&Vec<u8>> = if offset >= 0 {
                decoded_canon.iter().skip(offset as usize).collect()
            } else {
                // Negative offset: pad from end
                let abs_off = (-offset) as usize;
                let mut v: Vec<&Vec<u8>> = decoded_canon.iter().skip(0).collect();
                // Rotate left by abs_off
                if abs_off < v.len() {
                    let (a, b) = decoded_canon.split_at(abs_off);
                    v = b.iter().chain(a.iter()).collect();
                }
                v
            };

            let delta = self.encode_cycle_delta_aligned(frames, &aligned_canon,
                                                         &bg, cycle.canonical_idx);
            writer.write_chunk(ChunkType::CycleDelta, &delta, true);
        }

        writer.finish()
    }

    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Compute foreground bounding box with margin, aligned to 8-pixel blocks.
    fn fg_bbox(fg_mask: &[bool], w: usize, h: usize, margin: usize) -> (usize, usize, usize, usize) {
        let mut y0 = h; let mut y1 = 0usize;
        let mut x0 = w; let mut x1 = 0usize;
        for y in 0..h {
            for x in 0..w {
                if fg_mask[y * w + x] {
                    if y < y0 { y0 = y; }
                    if y > y1 { y1 = y; }
                    if x < x0 { x0 = x; }
                    if x > x1 { x1 = x; }
                }
            }
        }
        if y0 > y1 || x0 > x1 { return (0, 0, h, w); }
        y0 = y0.saturating_sub(margin);
        x0 = x0.saturating_sub(margin);
        y1 = (y1 + margin + 1).min(h);
        x1 = (x1 + margin + 1).min(w);
        // Align to 8-pixel blocks
        y0 = y0 / 8 * 8;
        x0 = x0 / 8 * 8;
        y1 = ((y1 + 7) / 8 * 8).min(h);
        x1 = ((x1 + 7) / 8 * 8).min(w);
        (y0, x0, y1, x1)
    }

    /// Crop a residual to a bounding box.
    fn crop_residual(residual: &[f32], w: usize, y0: usize, x0: usize, y1: usize, x1: usize) -> Vec<f32> {
        let cw = x1 - x0;
        let ch = y1 - y0;
        let mut cropped = vec![0.0f32; ch * cw * 3];
        for cy in 0..ch {
            for cx in 0..cw {
                let src = ((y0 + cy) * w + (x0 + cx)) * 3;
                let dst = (cy * cw + cx) * 3;
                for c in 0..3 {
                    cropped[dst + c] = residual[src + c];
                }
            }
        }
        cropped
    }

    /// Encode a frame with optional bounding box cropping.
    fn encode_frame_bbox(&self, residual: &[f32], fg_mask: Option<&[bool]>,
                          h: usize, w: usize) -> Vec<u8> {
        if let Some(mask) = fg_mask {
            let (y0, x0, y1, x1) = Self::fg_bbox(mask, w, h, 8);
            let ch = y1 - y0;
            let cw = x1 - x0;
            // Only use bbox if it's significantly smaller than full frame
            if ch * cw < h * w * 3 / 4 {
                let cropped = Self::crop_residual(residual, w, y0, x0, y1, x1);
                let encoded = encode_residual(&cropped, None, ch, cw, self.quality);
                // Bbox header: [2B H][2B W][2B y0][2B x0][2B cropH][2B cropW] = 12 bytes
                let mut out = Vec::with_capacity(12 + encoded.len());
                out.extend_from_slice(&(h as u16).to_be_bytes());
                out.extend_from_slice(&(w as u16).to_be_bytes());
                out.extend_from_slice(&(y0 as u16).to_be_bytes());
                out.extend_from_slice(&(x0 as u16).to_be_bytes());
                out.extend_from_slice(&(ch as u16).to_be_bytes());
                out.extend_from_slice(&(cw as u16).to_be_bytes());
                out.extend_from_slice(&encoded);
                return out;
            }
        }
        // Full frame encoding
        encode_residual(residual, fg_mask, h, w, self.quality)
    }

    /// Encode a canonical cycle with temporal prediction (I/P frames).
    /// Also decodes each frame to produce the exact reconstructed frames.
    fn encode_cycle_vs_bg_decoded(
        &self,
        frames: &[Vec<u8>],
        bg: &[u8],
    ) -> (Vec<u8>, Vec<Vec<u8>>) {
        let n = frames.len() as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&n.to_be_bytes());
        // flags: bit0=temporal, bit1=bbox
        let use_temporal = frames.len() > 2;
        let use_bbox = self.bg_model.is_ready();
        let flags: u8 = if use_temporal { 0x01 } else { 0x00 }
                       | if use_bbox { 0x02 } else { 0x00 };
        out.push(flags);

        let mut decoded_frames = Vec::with_capacity(frames.len());
        let mut prev_decoded: Option<Vec<u8>> = None;

        for (fi, frame) in frames.iter().enumerate() {
            let fg_mask_opt: Option<Vec<bool>> = if self.bg_model.is_ready() {
                Some(self.bg_model.get_foreground_mask(frame))
            } else {
                None
            };

            // Decide I-frame vs P-frame
            let is_iframe = !use_temporal || fi == 0 || fi % 25 == 0;
            let ref_frame: &[u8] = if is_iframe {
                bg
            } else {
                prev_decoded.as_ref().map(|v| v.as_slice()).unwrap_or(bg)
            };

            let residual: Vec<f32> = frame.iter().zip(ref_frame.iter())
                .map(|(&f, &b)| f as f32 - b as f32)
                .collect();

            // Zero out background pixels
            let mut masked_residual = residual.clone();
            if let Some(ref mask) = fg_mask_opt {
                for px in 0..(self.height * self.width) {
                    if !mask[px] {
                        masked_residual[px * 3] = 0.0;
                        masked_residual[px * 3 + 1] = 0.0;
                        masked_residual[px * 3 + 2] = 0.0;
                    }
                }
            }

            let encoded = if use_bbox {
                self.encode_frame_bbox(&masked_residual, fg_mask_opt.as_deref(),
                                       self.height, self.width)
            } else {
                encode_residual(&masked_residual, fg_mask_opt.as_deref(),
                                self.height, self.width, self.quality)
            };

            // Decode to get exact decoder output (prevents error stacking)
            let decoded = match decode_residual_payload(&encoded) {
                Ok((res, rh, rw)) => {
                    let recon: Vec<u8> = ref_frame.iter().zip(res.iter())
                        .map(|(&b, &r)| (b as i32 + r as i32).clamp(0, 255) as u8)
                        .collect();
                    let frame_size = self.height * self.width * 3;
                    if recon.len() > frame_size { recon[..frame_size].to_vec() } else { recon }
                }
                Err(_) => frame.to_vec(),
            };

            // Write: [1B frame_type][4B size][payload]
            out.push(if is_iframe { 0x00 } else { 0x01 });
            out.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
            out.extend_from_slice(&encoded);

            prev_decoded = Some(decoded.clone());
            decoded_frames.push(decoded);
        }
        (out, decoded_frames)
    }

    /// Find the best phase offset between two cycle frame sequences.
    /// Returns offset in [-max_offset, max_offset] that minimizes MSE.
    fn find_phase_offset(frames: &[Vec<u8>], canon: &[Vec<u8>],
                          bg: &[u8], max_offset: usize) -> i32 {
        let n = frames.len().min(canon.len());
        if n < 3 { return 0; }
        let sample_count = n.min(5); // Sample a few frames for speed
        let step = n / sample_count;

        let mut best_offset: i32 = 0;
        let mut best_cost = f64::MAX;

        let range = (max_offset as i32).min(n as i32 / 2);
        for off in -range..=range {
            let mut total_cost = 0.0f64;
            for si in 0..sample_count {
                let fi = si * step;
                let ci = (fi as i32 + off).rem_euclid(n as i32) as usize;
                if fi >= frames.len() || ci >= canon.len() { continue; }
                // Compute MSE of foreground pixels only (subsample for speed)
                let mut mse = 0.0f64;
                let mut count = 0u32;
                let stride = 8; // subsample every 8th pixel
                let len = frames[fi].len().min(canon[ci].len());
                for i in (0..len).step_by(stride) {
                    let diff = frames[fi][i] as f64 - canon[ci][i] as f64;
                    let bg_diff = (frames[fi][i] as f64 - bg[i.min(bg.len()-1)] as f64).abs();
                    if bg_diff > 15.0 { // only count foreground pixels
                        mse += diff * diff;
                        count += 1;
                    }
                }
                if count > 0 { total_cost += mse / count as f64; }
            }
            if total_cost < best_cost {
                best_cost = total_cost;
                best_offset = off;
            }
        }
        best_offset
    }

    fn encode_cycle_delta_aligned(
        &self,
        frames: &[Vec<u8>],
        canon_frames: &[&Vec<u8>],
        bg: &[u8],
        canon_idx: usize,
    ) -> Vec<u8> {
        let n = frames.len().min(canon_frames.len());
        let mut out = Vec::new();
        out.extend_from_slice(&(canon_idx as u16).to_be_bytes());
        out.extend_from_slice(&(n as u32).to_be_bytes());

        for i in 0..n {
            let fg_mask = self.bg_model.get_foreground_mask(&frames[i]);
            let residual: Vec<f32> = frames[i].iter().zip(canon_frames[i].iter())
                .map(|(&f, &c)| f as f32 - c as f32)
                .collect();
            // Zero out background pixels
            let mut masked = residual;
            for px in 0..(self.height * self.width) {
                if !fg_mask[px] {
                    masked[px * 3] = 0.0;
                    masked[px * 3 + 1] = 0.0;
                    masked[px * 3 + 2] = 0.0;
                }
            }
            let encoded = self.encode_frame_bbox(&masked, Some(&fg_mask),
                                                  self.height, self.width);
            out.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
            out.extend_from_slice(&encoded);
        }
        out
    }
}

/// JPEG-encode the background plate (RGB u8) using pure-Rust jpeg-encoder.
/// Quality 95 gives ~45 dB background fidelity, eliminating it as a bottleneck.
fn encode_background_jpeg(bg: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    jpeg_encoder::Encoder::new(&mut out, 95)
        .encode(bg, width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
        .expect("JPEG encode failed");
    out
}
