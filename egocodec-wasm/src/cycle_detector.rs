/// Work-cycle detector using energy-valley segmentation.

pub struct Cycle {
    pub start_frame: usize,
    pub end_frame: usize,
    pub energy: f32,
    pub is_canonical: bool,
    pub canonical_idx: usize,
}

pub struct CycleSegmentation {
    pub cycles: Vec<Cycle>,
    pub canonical_indices: Vec<usize>,
}

pub struct CycleDetector {
    min_cycle_frames: usize,
    max_cycle_frames: usize,
    smoothing_window: usize,
    valley_threshold: f32,
    canonical_max_count: usize,
    energies: Vec<f32>,
}

impl CycleDetector {
    pub fn new(fps: f32) -> Self {
        Self {
            min_cycle_frames: 90,
            max_cycle_frames: 3600,
            smoothing_window: 15,
            valley_threshold: 0.25,
            canonical_max_count: 5,
            energies: Vec::new(),
        }
    }

    pub fn push_energy(&mut self, energy: f32) {
        self.energies.push(energy);
    }

    pub fn segment(&self) -> CycleSegmentation {
        let smoothed = self.smooth(&self.energies);
        let boundaries = self.find_boundaries(&smoothed);
        let mut cycles = self.boundaries_to_cycles(&boundaries);
        self.assign_canonicals(&mut cycles);
        let canonical_indices: Vec<usize> = cycles.iter().enumerate()
            .filter(|(_, c)| c.is_canonical)
            .map(|(i, _)| i)
            .collect();
        CycleSegmentation { cycles, canonical_indices }
    }

    fn smooth(&self, energies: &[f32]) -> Vec<f32> {
        let w = self.smoothing_window as isize;
        let n = energies.len();
        if n < (2 * w + 1) as usize {
            return energies.to_vec();
        }
        // Gaussian kernel
        let hw = w as f32 / 2.0;
        let kernel: Vec<f32> = (-w..=w)
            .map(|i| (-0.5 * (i as f32 / hw).powi(2)).exp())
            .collect();
        let ksum: f32 = kernel.iter().sum();
        let kernel: Vec<f32> = kernel.into_iter().map(|k| k / ksum).collect();

        let mut result = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for (j, &k) in kernel.iter().enumerate() {
                let idx = (i as isize + j as isize - w).clamp(0, n as isize - 1) as usize;
                sum += energies[idx] * k;
            }
            result[i] = sum;
        }
        result
    }

    fn find_boundaries(&self, smoothed: &[f32]) -> Vec<usize> {
        let n = smoothed.len();
        let mut rolling_max = vec![0.0f32; n];
        rolling_max[0] = smoothed[0];
        for i in 1..n {
            rolling_max[i] = rolling_max[i - 1].max(smoothed[i]);
        }

        let mut boundaries = vec![0usize];
        let mut last_b = 0;

        for i in 1..n.saturating_sub(1) {
            let gap = i - last_b;
            if gap < self.min_cycle_frames {
                continue;
            }
            if gap > self.max_cycle_frames {
                boundaries.push(i);
                last_b = i;
                continue;
            }
            let threshold = rolling_max[i] * self.valley_threshold;
            if smoothed[i] < threshold
                && smoothed[i] <= smoothed[i - 1]
                && smoothed[i] <= smoothed[i + 1]
            {
                boundaries.push(i);
                last_b = i;
            }
        }
        boundaries.push(n);
        boundaries
    }

    fn boundaries_to_cycles(&self, boundaries: &[usize]) -> Vec<Cycle> {
        boundaries.windows(2).map(|w| {
            let (start, end) = (w[0], w[1]);
            let energy: f32 = if end > start {
                self.energies[start..end].iter().sum::<f32>() / (end - start) as f32
            } else {
                0.0
            };
            Cycle {
                start_frame: start,
                end_frame: end,
                energy,
                is_canonical: false,
                canonical_idx: 0,
            }
        }).collect()
    }

    fn assign_canonicals(&self, cycles: &mut [Cycle]) {
        if cycles.is_empty() { return; }

        let energies: Vec<f32> = cycles.iter().map(|c| c.energy).collect();
        let lengths: Vec<f32> = cycles.iter().map(|c| (c.end_frame - c.start_frame) as f32).collect();
        let e_max = energies.iter().cloned().fold(0.0f32, f32::max) + 1e-6;
        let l_max = lengths.iter().cloned().fold(0.0f32, f32::max) + 1e-6;

        let features: Vec<[f32; 2]> = energies.iter().zip(lengths.iter())
            .map(|(&e, &l)| [e / e_max, l / l_max])
            .collect();

        // Seed: closest to median energy
        let mut sorted_e = energies.clone();
        sorted_e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_e = sorted_e[sorted_e.len() / 2];
        let seed = energies.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - median_e).abs()).partial_cmp(&((**b - median_e).abs())).unwrap()
            })
            .map(|(i, _)| i).unwrap();

        let mut canonicals = vec![seed];

        // Expand by max-min distance
        while canonicals.len() < self.canonical_max_count && canonicals.len() < cycles.len() {
            let mut best_dist = -1.0f32;
            let mut best_idx = 0;
            for i in 0..cycles.len() {
                if canonicals.contains(&i) { continue; }
                let min_dist = canonicals.iter()
                    .map(|&c| {
                        let dx = features[i][0] - features[c][0];
                        let dy = features[i][1] - features[c][1];
                        (dx * dx + dy * dy).sqrt()
                    })
                    .fold(f32::INFINITY, f32::min);
                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = i;
                }
            }
            if best_dist < 0.05 { break; }
            canonicals.push(best_idx);
        }

        for &idx in &canonicals {
            cycles[idx].is_canonical = true;
        }

        // Assign each non-canonical to closest canonical
        for i in 0..cycles.len() {
            if cycles[i].is_canonical {
                cycles[i].canonical_idx = canonicals.iter().position(|&c| c == i).unwrap_or(0);
                continue;
            }
            let best = canonicals.iter().enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    let da = ((features[i][0] - features[a][0]).powi(2)
                        + (features[i][1] - features[a][1]).powi(2)).sqrt();
                    let db = ((features[i][0] - features[b][0]).powi(2)
                        + (features[i][1] - features[b][1]).powi(2)).sqrt();
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(j, _)| j).unwrap_or(0);
            cycles[i].canonical_idx = best;
        }
    }
}
