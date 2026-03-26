/// EgoEncoder: top-level encode pipeline (Rust WASM version).

use crate::background::BackgroundModel;
use crate::bitstream::{BitstreamWriter, ChunkType};
use crate::cycle_detector::{CycleDetector, CycleSegmentation};
use crate::residual_codec::encode_residual;

pub struct EgoEncoder {
    fps: f32,
    width: usize,
    height: usize,
    quality: u8,
    bg_model: BackgroundModel,
    cycle_det: CycleDetector,
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
        Self {
            fps,
            width,
            height,
            quality,
            bg_model: BackgroundModel::new(warmup_frames, width, height),
            cycle_det: CycleDetector::new(fps),
            frame_buffer: Vec::new(),
            prev_frame: None,
            total_frames: 0,
        }
    }

    /// Push one RGB frame (u8 H*W*3).
    pub fn push_frame(&mut self, frame: &[u8]) {
        self.bg_model.update(frame);

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

    /// Encode all frames and return .ego bytes.
    pub fn encode(&self) -> Vec<u8> {
        let seg = self.cycle_det.segment();
        let bg = self.bg_model.get_background();

        let mut writer = BitstreamWriter::new(
            self.total_frames as u64,
            self.fps,
            self.width as u16,
            self.height as u16,
            false,
        );

        // Metadata
        let meta = serde_json::json!({
            "total_frames": self.total_frames,
            "fps": self.fps,
            "n_cycles": seg.cycles.len(),
            "n_canonicals": seg.canonical_indices.len(),
            "quality": self.quality,
        });
        let meta_bytes = serde_json::to_vec(&meta).unwrap();
        writer.write_chunk(ChunkType::Metadata, &meta_bytes, false);

        // Background JPEG
        let bg_jpeg = encode_background_jpeg(&bg, self.width, self.height);
        writer.write_chunk(ChunkType::Background, &bg_jpeg, false);

        // Canonical cycles
        let mut canonical_frames_list: Vec<Vec<Vec<u8>>> = Vec::new();
        for &cycle_idx in &seg.canonical_indices {
            let cycle = &seg.cycles[cycle_idx];
            let frames = &self.frame_buffer[cycle.start_frame..cycle.end_frame];
            let encoded = self.encode_cycle_vs_bg(frames, &bg);
            canonical_frames_list.push(frames.to_vec());
            writer.write_chunk(ChunkType::CycleCanon, &encoded, true);
        }

        // Non-canonical cycles as deltas
        for cycle in &seg.cycles {
            if cycle.is_canonical { continue; }
            let frames = &self.frame_buffer[cycle.start_frame..cycle.end_frame];
            let canon_frames = &canonical_frames_list[cycle.canonical_idx];
            let delta = self.encode_cycle_delta(frames, canon_frames, &bg, cycle.canonical_idx);
            writer.write_chunk(ChunkType::CycleDelta, &delta, true);
        }

        writer.finish()
    }

    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    fn encode_cycle_vs_bg(&self, frames: &[Vec<u8>], bg: &[u8]) -> Vec<u8> {
        let n = frames.len() as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&n.to_be_bytes());
        out.push(0x00); // flags = legacy

        for frame in frames {
            let fg_mask = self.bg_model.get_foreground_mask(frame);
            let residual: Vec<f32> = frame.iter().zip(bg.iter())
                .map(|(&f, &b)| f as f32 - b as f32)
                .collect();
            let encoded = encode_residual(&residual, Some(&fg_mask),
                                          self.height, self.width, self.quality);
            out.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
            out.extend_from_slice(&encoded);
        }
        out
    }

    fn encode_cycle_delta(
        &self,
        frames: &[Vec<u8>],
        canon_frames: &[Vec<u8>],
        _bg: &[u8],
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
            let encoded = encode_residual(&residual, Some(&fg_mask),
                                          self.height, self.width, self.quality);
            out.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
            out.extend_from_slice(&encoded);
        }
        out
    }
}

/// Simple JPEG-like background encoding (just stores raw RGB for now,
/// real JPEG encoding would need a full JPEG encoder crate).
/// For WASM we can use the browser's Canvas API instead.
fn encode_background_jpeg(bg: &[u8], width: usize, height: usize) -> Vec<u8> {
    // Minimal: store as raw with a header so decoder knows format
    // In practice, the JS side will handle JPEG encoding via Canvas
    let mut out = Vec::with_capacity(4 + bg.len());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(bg);
    out
}
