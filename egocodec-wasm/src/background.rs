/// Background model using Welford's online algorithm.
/// Constant memory — no frame buffering.

pub struct BackgroundModel {
    pub warmup_frames: usize,
    update_alpha: f32,
    fg_sigma_thresh: f32,
    bg_mean: Option<Vec<f32>>,  // H*W*3
    bg_std: Option<Vec<f32>>,   // H*W*3
    bg_m2: Option<Vec<f32>>,    // Welford accumulator
    frame_idx: usize,
    warmed_up: bool,
    width: usize,
    height: usize,
}

impl BackgroundModel {
    pub fn new(warmup_frames: usize, width: usize, height: usize) -> Self {
        Self {
            warmup_frames,
            update_alpha: 0.002,
            fg_sigma_thresh: 3.5,
            bg_mean: None,
            bg_std: None,
            bg_m2: None,
            frame_idx: 0,
            warmed_up: false,
            width,
            height,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.warmed_up
    }

    /// Feed a frame (u8 H*W*3 in row-major order).
    pub fn update(&mut self, frame: &[u8]) {
        let n = self.frame_idx + 1;
        let len = self.width * self.height * 3;

        if !self.warmed_up {
            if self.bg_mean.is_none() {
                self.bg_mean = Some(frame.iter().map(|&v| v as f32).collect());
                self.bg_m2 = Some(vec![0.0f32; len]);
            } else {
                let mean = self.bg_mean.as_mut().unwrap();
                let m2 = self.bg_m2.as_mut().unwrap();
                let nf = n as f32;
                for i in 0..len {
                    let f = frame[i] as f32;
                    let delta = f - mean[i];
                    mean[i] += delta / nf;
                    let delta2 = f - mean[i];
                    m2[i] += delta * delta2;
                }
            }
            if n >= self.warmup_frames {
                let m2 = self.bg_m2.take().unwrap();
                let denom = (n - 1).max(1) as f32;
                let std: Vec<f32> = m2.into_iter()
                    .map(|v| (v / denom).sqrt().max(2.0))
                    .collect();
                self.bg_std = Some(std);
                self.warmed_up = true;
            }
        } else {
            // EMA update for background pixels only
            let alpha = self.update_alpha;
            let mean = self.bg_mean.as_mut().unwrap();
            let std = self.bg_std.as_ref().unwrap();
            let thresh = self.fg_sigma_thresh;

            for y in 0..self.height {
                for x in 0..self.width {
                    let base = (y * self.width + x) * 3;
                    // Check if any channel exceeds threshold (foreground)
                    let mut is_fg = false;
                    for c in 0..3 {
                        let diff = (frame[base + c] as f32 - mean[base + c]).abs();
                        if diff / std[base + c] > thresh {
                            is_fg = true;
                            break;
                        }
                    }
                    if !is_fg {
                        for c in 0..3 {
                            let idx = base + c;
                            mean[idx] += alpha * (frame[idx] as f32 - mean[idx]);
                        }
                    }
                }
            }
        }
        self.frame_idx += 1;
    }

    /// Get background as u8 slice.
    pub fn get_background(&self) -> Vec<u8> {
        self.bg_mean.as_ref().unwrap().iter()
            .map(|&v| v.clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Get foreground mask (true = foreground). Returns H*W bools.
    pub fn get_foreground_mask(&self, frame: &[u8]) -> Vec<bool> {
        let mean = self.bg_mean.as_ref().unwrap();
        let std = self.bg_std.as_ref().unwrap();
        let thresh = self.fg_sigma_thresh;
        let mut mask = vec![false; self.width * self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let base = (y * self.width + x) * 3;
                let px = y * self.width + x;
                for c in 0..3 {
                    let diff = (frame[base + c] as f32 - mean[base + c]).abs();
                    if diff / std[base + c] > thresh {
                        mask[px] = true;
                        break;
                    }
                }
            }
        }
        mask
    }
}
