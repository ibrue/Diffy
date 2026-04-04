//! Lightweight monocular Visual SLAM for camera pose estimation.
//!
//! Pure Rust, no external linear algebra — runs in browser WASM.
//! Pipeline: Harris corners → BRIEF descriptors → Hamming matching →
//! 8-point essential matrix + RANSAC → pose recovery → DLT triangulation.

// ─── Data types ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CameraPose {
    pub r: [[f32; 3]; 3],
    pub t: [f32; 3],
    pub frame_idx: u32,
}

impl CameraPose {
    pub fn identity(idx: u32) -> Self {
        Self {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0, 0.0, 0.0],
            frame_idx: idx,
        }
    }
}

pub struct SlamResult {
    pub poses: Vec<CameraPose>,
    pub point_cloud: Vec<[f32; 3]>,
    pub intrinsics: [[f64; 3]; 3],
    pub success: bool,
}

// ─── Deterministic RNG (xorshift32) ──────────────────────────────────

struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self { Self(seed) }
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() as usize) % max
    }
}

// ─── 3x3 matrix helpers (f64) ────────────────────────────────────────

type Mat3 = [[f64; 3]; 3];
type Vec3 = [f64; 3];

fn mat3_identity() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3_transpose(a: &Mat3) -> Mat3 {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn mat3_vec(a: &Mat3, v: &Vec3) -> Vec3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn vec3_norm(v: &Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn vec3_normalize(v: &Vec3) -> Vec3 {
    let n = vec3_norm(v);
    if n < 1e-12 { return [0.0, 0.0, 1.0]; }
    [v[0] / n, v[1] / n, v[2] / n]
}

fn mat3_det(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}


// ─── Jacobi SVD for 3×3 matrices ─────────────────────────────────────

fn svd3x3(m: &Mat3) -> (Mat3, [f64; 3], Mat3) {
    // Compute M^T * M
    let mt = mat3_transpose(m);
    let ata = mat3_mul(&mt, m);

    // Jacobi eigendecomposition of A^T*A → V, sigma^2
    let mut v = mat3_identity();
    let mut d = ata;

    for _ in 0..50 {
        for p in 0..3 {
            for q in (p + 1)..3 {
                if d[p][q].abs() < 1e-12 { continue; }
                let theta = 0.5 * ((d[q][q] - d[p][p]) / (2.0 * d[p][q])).atan();
                let c = theta.cos();
                let s = theta.sin();

                // Givens rotation on d
                let mut new_d = d;
                new_d[p][p] = c * c * d[p][p] - 2.0 * s * c * d[p][q] + s * s * d[q][q];
                new_d[q][q] = s * s * d[p][p] + 2.0 * s * c * d[p][q] + c * c * d[q][q];
                new_d[p][q] = 0.0;
                new_d[q][p] = 0.0;
                for i in 0..3 {
                    if i != p && i != q {
                        let dip = c * d[i][p] - s * d[i][q];
                        let diq = s * d[i][p] + c * d[i][q];
                        new_d[i][p] = dip;
                        new_d[p][i] = dip;
                        new_d[i][q] = diq;
                        new_d[q][i] = diq;
                    }
                }
                d = new_d;

                // Accumulate V
                let mut new_v = v;
                for i in 0..3 {
                    new_v[i][p] = c * v[i][p] - s * v[i][q];
                    new_v[i][q] = s * v[i][p] + c * v[i][q];
                }
                v = new_v;
            }
        }
    }

    // Sort eigenvalues descending
    let mut idx = [0usize, 1, 2];
    if d[idx[0]][idx[0]] < d[idx[1]][idx[1]] { idx.swap(0, 1); }
    if d[idx[0]][idx[0]] < d[idx[2]][idx[2]] { idx.swap(0, 2); }
    if d[idx[1]][idx[1]] < d[idx[2]][idx[2]] { idx.swap(1, 2); }

    let sigma = [
        d[idx[0]][idx[0]].max(0.0).sqrt(),
        d[idx[1]][idx[1]].max(0.0).sqrt(),
        d[idx[2]][idx[2]].max(0.0).sqrt(),
    ];

    let mut vt = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            vt[i][j] = v[j][idx[i]];
        }
    }

    // U = M * V * Sigma^-1
    let v_sorted = mat3_transpose(&vt);
    let mv = mat3_mul(m, &v_sorted);
    let mut u = mat3_identity();
    for i in 0..3 {
        if sigma[i] > 1e-10 {
            for j in 0..3 {
                u[j][i] = mv[j][i] / sigma[i];
            }
        }
    }

    // Fix signs for proper rotations
    if mat3_det(&u) < 0.0 {
        for j in 0..3 { u[j][2] = -u[j][2]; }
    }
    if mat3_det(&vt) < 0.0 {
        for j in 0..3 { vt[2][j] = -vt[2][j]; }
    }

    (u, sigma, vt)
}


// ─── Image processing ────────────────────────────────────────────────

fn to_gray(frame: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut gray = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            if i + 2 < frame.len() {
                gray[y * w + x] = 0.299 * frame[i] as f32
                    + 0.587 * frame[i + 1] as f32
                    + 0.114 * frame[i + 2] as f32;
            }
        }
    }
    gray
}

fn downsample_gray(gray: &[f32], w: usize, h: usize, factor: usize) -> (Vec<f32>, usize, usize) {
    let nw = w / factor;
    let nh = h / factor;
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            out[y * nw + x] = gray[y * factor * w + x * factor];
        }
    }
    (out, nw, nh)
}

fn gaussian_blur(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Simple 5x5 Gaussian (sigma ~1)
    let kernel = [1.0, 4.0, 6.0, 4.0, 1.0]; // /16 per dim
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];
    // Horizontal pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut wt = 0.0f32;
            for k in 0..5i32 {
                let xx = (x as i32 + k - 2).clamp(0, w as i32 - 1) as usize;
                sum += img[y * w + xx] * kernel[k as usize];
                wt += kernel[k as usize];
            }
            tmp[y * w + x] = sum / wt;
        }
    }
    // Vertical pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut wt = 0.0f32;
            for k in 0..5i32 {
                let yy = (y as i32 + k - 2).clamp(0, h as i32 - 1) as usize;
                sum += tmp[yy * w + x] * kernel[k as usize];
                wt += kernel[k as usize];
            }
            out[y * w + x] = sum / wt;
        }
    }
    out
}

// ─── Harris corner detection ─────────────────────────────────────────

pub fn detect_features(gray: &[f32], w: usize, h: usize, max_features: usize) -> Vec<[f32; 2]> {
    if w < 16 || h < 16 { return Vec::new(); }

    // Sobel gradients
    let mut ix = vec![0.0f32; w * h];
    let mut iy = vec![0.0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            ix[idx] = -gray[(y - 1) * w + x - 1] + gray[(y - 1) * w + x + 1]
                - 2.0 * gray[y * w + x - 1] + 2.0 * gray[y * w + x + 1]
                - gray[(y + 1) * w + x - 1] + gray[(y + 1) * w + x + 1];
            iy[idx] = -gray[(y - 1) * w + x - 1] - 2.0 * gray[(y - 1) * w + x] - gray[(y - 1) * w + x + 1]
                + gray[(y + 1) * w + x - 1] + 2.0 * gray[(y + 1) * w + x] + gray[(y + 1) * w + x + 1];
        }
    }

    // Structure tensor components (Gaussian smoothed)
    let mut ixx: Vec<f32> = ix.iter().zip(ix.iter()).map(|(a, b)| a * b).collect();
    let mut iyy: Vec<f32> = iy.iter().zip(iy.iter()).map(|(a, b)| a * b).collect();
    let mut ixy: Vec<f32> = ix.iter().zip(iy.iter()).map(|(a, b)| a * b).collect();
    ixx = gaussian_blur(&ixx, w, h);
    iyy = gaussian_blur(&iyy, w, h);
    ixy = gaussian_blur(&ixy, w, h);

    // Harris response
    let k = 0.04f32;
    let mut response = vec![0.0f32; w * h];
    let mut max_r = 0.0f32;
    for i in 0..w * h {
        let det = ixx[i] * iyy[i] - ixy[i] * ixy[i];
        let trace = ixx[i] + iyy[i];
        response[i] = det - k * trace * trace;
        if response[i] > max_r { max_r = response[i]; }
    }

    let thresh = max_r * 0.01;
    if thresh <= 0.0 { return Vec::new(); }

    // Non-max suppression + border exclusion
    let border = 8;
    let mut corners: Vec<(f32, f32, f32)> = Vec::new(); // (strength, x, y)
    for y in border..h - border {
        for x in border..w - border {
            let val = response[y * w + x];
            if val <= thresh { continue; }
            let mut is_max = true;
            'nms: for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    if dy == 0 && dx == 0 { continue; }
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    if response[ny * w + nx] > val {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                corners.push((val, x as f32, y as f32));
            }
        }
    }

    // Sort by strength descending, take top N
    corners.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    corners.truncate(max_features);
    corners.iter().map(|c| [c.1, c.2]).collect()
}


// ─── BRIEF descriptors ───────────────────────────────────────────────

// Fixed sampling pattern (256 pairs), deterministic
const BRIEF_PAIRS: [[i8; 4]; 256] = {
    let mut pairs = [[0i8; 4]; 256];
    let mut rng: u32 = 0xB01EF;
    let mut i = 0;
    while i < 256 {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        let a = (rng % 25) as i8 - 12;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        let b = (rng % 25) as i8 - 12;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        let c = (rng % 25) as i8 - 12;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        let d = (rng % 25) as i8 - 12;
        pairs[i] = [a, b, c, d];
        i += 1;
    }
    pairs
};

pub fn compute_descriptors(gray: &[f32], w: usize, h: usize, keypoints: &[[f32; 2]]) -> Vec<[u8; 32]> {
    let smooth = gaussian_blur(gray, w, h);
    let mut descs = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let cx = kp[0] as i32;
        let cy = kp[1] as i32;
        let mut desc = [0u8; 32];

        for j in 0..256 {
            let [dy1, dx1, dy2, dx2] = BRIEF_PAIRS[j];
            let y1 = (cy + dy1 as i32).clamp(0, h as i32 - 1) as usize;
            let x1 = (cx + dx1 as i32).clamp(0, w as i32 - 1) as usize;
            let y2 = (cy + dy2 as i32).clamp(0, h as i32 - 1) as usize;
            let x2 = (cx + dx2 as i32).clamp(0, w as i32 - 1) as usize;

            if smooth[y1 * w + x1] < smooth[y2 * w + x2] {
                desc[j / 8] |= 1 << (j % 8);
            }
        }
        descs.push(desc);
    }
    descs
}

// ─── Feature matching ────────────────────────────────────────────────

fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut dist = 0u32;
    for i in 0..32 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

pub fn match_features(
    desc1: &[[u8; 32]], kp1: &[[f32; 2]],
    desc2: &[[u8; 32]], kp2: &[[f32; 2]],
) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let n1 = desc1.len();
    let n2 = desc2.len();
    if n1 == 0 || n2 == 0 { return (Vec::new(), Vec::new()); }

    let mut pts1 = Vec::new();
    let mut pts2 = Vec::new();

    for i in 0..n1 {
        let mut best = u32::MAX;
        let mut second = u32::MAX;
        let mut best_j = 0usize;

        for j in 0..n2 {
            let d = hamming_distance(&desc1[i], &desc2[j]);
            if d < best {
                second = best;
                best = d;
                best_j = j;
            } else if d < second {
                second = d;
            }
        }

        if best > 80 { continue; }
        if n2 > 1 && second > 0 && (best as f64 / second as f64) > 0.75 { continue; }

        pts1.push(kp1[i]);
        pts2.push(kp2[best_j]);
    }

    (pts1, pts2)
}


// ─── Essential matrix estimation ─────────────────────────────────────

fn normalize_point(px: f32, py: f32, k: &[[f64; 3]; 3]) -> (f64, f64) {
    let fx = k[0][0]; let fy = k[1][1];
    let cx = k[0][2]; let cy = k[1][2];
    ((px as f64 - cx) / fx, (py as f64 - cy) / fy)
}

pub fn estimate_essential(
    pts1: &[[f32; 2]], pts2: &[[f32; 2]],
    k: &[[f64; 3]; 3],
    ransac_iters: usize,
) -> Option<Mat3> {
    let n = pts1.len();
    if n < 8 { return None; }

    let n1: Vec<(f64, f64)> = pts1.iter().map(|p| normalize_point(p[0], p[1], k)).collect();
    let n2: Vec<(f64, f64)> = pts2.iter().map(|p| normalize_point(p[0], p[1], k)).collect();

    let mut rng = Rng::new(42);
    let mut best_e: Option<Mat3> = None;
    let mut best_inliers = 0usize;

    for _ in 0..ransac_iters {
        // Sample 8 unique indices
        let mut idx = [0usize; 8];
        for j in 0..8 {
            loop {
                let v = rng.next_usize(n);
                if !idx[..j].contains(&v) { idx[j] = v; break; }
            }
        }

        // Build 8x9 constraint matrix and solve via simplified approach
        // For 8 points: Ax = 0 where x = vec(E)
        // We use the cofactor/nullspace approach for the last row of V
        let mut ata = [[0.0f64; 9]; 9];
        for &j in &idx {
            let (x1, y1) = n1[j];
            let (x2, y2) = n2[j];
            let row = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0];
            for r in 0..9 {
                for c in 0..9 {
                    ata[r][c] += row[r] * row[c];
                }
            }
        }

        // Find eigenvector of A^T*A with smallest eigenvalue (power iteration on inverse)
        // Simpler: just do Jacobi on 9x9... but that's heavy.
        // Instead: use the 8-point direct solution — with 8 equations and 9 unknowns,
        // we can find the null space by Gaussian elimination.
        let e_vec = solve_null_8x9(&idx, &n1, &n2);
        if e_vec.is_none() { continue; }
        let ev = e_vec.unwrap();

        let mut e = [[0.0; 3]; 3];
        for i in 0..3 { for j in 0..3 { e[i][j] = ev[i * 3 + j]; } }

        // Enforce rank-2 via SVD
        let (u, mut s, vt) = svd3x3(&e);
        let avg = (s[0] + s[1]) / 2.0;
        s = [avg, avg, 0.0];
        let diag = [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]];
        let e = mat3_mul(&mat3_mul(&u, &diag), &vt);

        // Count inliers
        let mut inliers = 0;
        for i in 0..n {
            let (x1, y1) = n1[i];
            let (x2, y2) = n2[i];
            let p1 = [x1, y1, 1.0];
            let p2 = [x2, y2, 1.0];
            let ep1 = mat3_vec(&e, &p1);
            let err = (p2[0] * ep1[0] + p2[1] * ep1[1] + p2[2] * ep1[2]).abs();
            if err < 0.005 { inliers += 1; }
        }

        if inliers > best_inliers {
            best_inliers = inliers;
            best_e = Some(e);
        }
    }

    if best_inliers < 8 { return None; }
    best_e
}

/// Solve 8x9 homogeneous system via Gaussian elimination to find null space.
fn solve_null_8x9(
    idx: &[usize; 8],
    n1: &[(f64, f64)],
    n2: &[(f64, f64)],
) -> Option<[f64; 9]> {
    let mut a = [[0.0f64; 9]; 8];
    for j in 0..8 {
        let (x1, y1) = n1[idx[j]];
        let (x2, y2) = n2[idx[j]];
        a[j] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..8 {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = col;
        for row in col..8 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 { return None; }
        a.swap(col, max_row);

        let pivot = a[col][col];
        for c in col..9 { a[col][c] /= pivot; }

        for row in 0..8 {
            if row == col { continue; }
            let factor = a[row][col];
            for c in col..9 { a[row][c] -= factor * a[col][c]; }
        }
    }

    // Last column (8) is the free variable — set to 1
    let mut x = [0.0f64; 9];
    x[8] = 1.0;
    for i in 0..8 {
        x[i] = -a[i][8];
    }

    // Normalize
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-12 { return None; }
    for v in &mut x { *v /= norm; }
    Some(x)
}


// ─── Pose recovery from essential matrix ─────────────────────────────

pub fn recover_pose(
    e: &Mat3,
    pts1: &[[f32; 2]], pts2: &[[f32; 2]],
    k: &[[f64; 3]; 3],
) -> (Mat3, Vec3) {
    let (u, _, vt) = svd3x3(e);

    let w: Mat3 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let wt: Mat3 = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    let r1 = mat3_mul(&mat3_mul(&u, &w), &vt);
    let r2 = mat3_mul(&mat3_mul(&u, &wt), &vt);
    let t_pos = [u[0][2], u[1][2], u[2][2]];
    let t_neg = [-u[0][2], -u[1][2], -u[2][2]];

    let candidates: [(Mat3, Vec3); 4] = [
        (r1, t_pos), (r1, t_neg), (r2, t_pos), (r2, t_neg),
    ];

    let mut best_r = mat3_identity();
    let mut best_t = [0.0; 3];
    let mut best_count = 0i32;

    let check_n = pts1.len().min(15);

    for (mut r, mut t) in candidates {
        if mat3_det(&r) < 0.0 {
            for i in 0..3 { for j in 0..3 { r[i][j] = -r[i][j]; } }
            t = [-t[0], -t[1], -t[2]];
        }

        let mut count = 0i32;
        for j in 0..check_n {
            let (x1, y1) = normalize_point(pts1[j][0], pts1[j][1], k);
            let (x2, y2) = normalize_point(pts2[j][0], pts2[j][1], k);

            // DLT triangulation
            let p1 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
            let p2 = [
                [r[0][0], r[0][1], r[0][2], t[0]],
                [r[1][0], r[1][1], r[1][2], t[1]],
                [r[2][0], r[2][1], r[2][2], t[2]],
            ];

            // Build 4x4 system
            let a = [
                [x1 * p1[2][0] - p1[0][0], x1 * p1[2][1] - p1[0][1], x1 * p1[2][2] - p1[0][2], x1 * p1[2][3] - p1[0][3]],
                [y1 * p1[2][0] - p1[1][0], y1 * p1[2][1] - p1[1][1], y1 * p1[2][2] - p1[1][2], y1 * p1[2][3] - p1[1][3]],
                [x2 * p2[2][0] - p2[0][0], x2 * p2[2][1] - p2[0][1], x2 * p2[2][2] - p2[0][2], x2 * p2[2][3] - p2[0][3]],
                [y2 * p2[2][0] - p2[1][0], y2 * p2[2][1] - p2[1][1], y2 * p2[2][2] - p2[1][2], y2 * p2[2][3] - p2[1][3]],
            ];

            if let Some(pt) = solve_null_4x4(&a) {
                if pt[3].abs() < 1e-10 { continue; }
                let x3d = [pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3]];
                // Check positive depth in camera 1
                if x3d[2] <= 0.0 { continue; }
                // Check positive depth in camera 2
                let x_cam2 = mat3_vec(&r, &x3d);
                if x_cam2[0] + t[0] > 0.0 || x_cam2[2] + t[2] > 0.0 {
                    count += 1;
                }
            }
        }

        if count > best_count {
            best_count = count;
            best_r = r;
            best_t = t;
        }
    }

    // Normalize t
    let tn = vec3_norm(&best_t);
    if tn > 1e-10 {
        best_t = [best_t[0] / tn, best_t[1] / tn, best_t[2] / tn];
    }

    (best_r, best_t)
}

/// Solve 4x4 homogeneous system — find null vector via Gaussian elimination.
fn solve_null_4x4(a: &[[f64; 4]; 4]) -> Option<[f64; 4]> {
    let mut m = *a;
    for col in 0..3 {
        let mut max_val = 0.0;
        let mut max_row = col;
        for row in col..4 {
            if m[row][col].abs() > max_val { max_val = m[row][col].abs(); max_row = row; }
        }
        if max_val < 1e-14 { return None; }
        m.swap(col, max_row);
        let pivot = m[col][col];
        for c in col..4 { m[col][c] /= pivot; }
        for row in 0..4 {
            if row == col { continue; }
            let factor = m[row][col];
            for c in col..4 { m[row][c] -= factor * m[col][c]; }
        }
    }
    let mut x = [0.0f64; 4];
    x[3] = 1.0;
    for i in 0..3 { x[i] = -m[i][3]; }
    Some(x)
}

// ─── Point triangulation ─────────────────────────────────────────────

pub fn triangulate_points(
    pts1: &[[f32; 2]], pts2: &[[f32; 2]],
    k: &[[f64; 3]; 3],
    r: &Mat3, t: &Vec3,
) -> Vec<[f32; 3]> {
    let mut points = Vec::new();
    let p2 = [
        [r[0][0], r[0][1], r[0][2], t[0]],
        [r[1][0], r[1][1], r[1][2], t[1]],
        [r[2][0], r[2][1], r[2][2], t[2]],
    ];

    for i in 0..pts1.len() {
        let (x1, y1) = normalize_point(pts1[i][0], pts1[i][1], k);
        let (x2, y2) = normalize_point(pts2[i][0], pts2[i][1], k);

        let a: [[f64; 4]; 4] = [
            [x1 * 0.0 - 1.0, x1 * 0.0 - 0.0, x1 * 1.0 - 0.0, 0.0],
            [y1 * 0.0 - 0.0, y1 * 0.0 - 1.0, y1 * 1.0 - 0.0, 0.0],
            [x2 * p2[2][0] - p2[0][0], x2 * p2[2][1] - p2[0][1], x2 * p2[2][2] - p2[0][2], x2 * p2[2][3] - p2[0][3]],
            [y2 * p2[2][0] - p2[1][0], y2 * p2[2][1] - p2[1][1], y2 * p2[2][2] - p2[1][2], y2 * p2[2][3] - p2[1][3]],
        ];

        if let Some(pt) = solve_null_4x4(&a) {
            if pt[3].abs() < 1e-10 { continue; }
            let x = (pt[0] / pt[3]) as f32;
            let y = (pt[1] / pt[3]) as f32;
            let z = (pt[2] / pt[3]) as f32;
            if z > 0.0 && z < 100.0 {
                points.push([x, y, z]);
            }
        }
    }
    points
}


// ─── Visual SLAM pipeline ────────────────────────────────────────────

pub struct VisualSLAM {
    k: [[f64; 3]; 3],
    width: usize,
    height: usize,
    max_features: usize,
    keyframe_interval: usize,
    poses: Vec<CameraPose>,
    point_cloud_parts: Vec<Vec<[f32; 3]>>,
    frame_idx: usize,
    prev_kp: Option<Vec<[f32; 2]>>,
    prev_desc: Option<Vec<[u8; 32]>>,
    r_world: Mat3,
    t_world: Vec3,
}

impl VisualSLAM {
    pub fn new(
        k: Option<[[f64; 3]; 3]>,
        width: usize,
        height: usize,
        max_features: usize,
        keyframe_interval: usize,
    ) -> Self {
        let k = k.unwrap_or_else(|| {
            let f = (width.max(height) as f64) * 1.2;
            [[f, 0.0, width as f64 / 2.0],
             [0.0, f, height as f64 / 2.0],
             [0.0, 0.0, 1.0]]
        });
        Self {
            k, width, height, max_features, keyframe_interval,
            poses: Vec::new(),
            point_cloud_parts: Vec::new(),
            frame_idx: 0,
            prev_kp: None,
            prev_desc: None,
            r_world: mat3_identity(),
            t_world: [0.0; 3],
        }
    }

    pub fn process_frame(&mut self, frame: &[u8]) -> CameraPose {
        let is_keyframe = self.frame_idx % self.keyframe_interval == 0;

        if is_keyframe {
            let gray = to_gray(frame, self.width, self.height);
            // Downsample for speed
            let ds = (self.width.min(self.height) / 256).max(1);
            let (gray_ds, w_ds, h_ds) = if ds > 1 {
                downsample_gray(&gray, self.width, self.height, ds)
            } else {
                (gray.clone(), self.width, self.height)
            };

            let mut kp = detect_features(&gray_ds, w_ds, h_ds, self.max_features);
            if ds > 1 {
                for p in &mut kp { p[0] *= ds as f32; p[1] *= ds as f32; }
            }

            let desc = compute_descriptors(&gray, self.width, self.height, &kp);

            if let (Some(prev_kp), Some(prev_desc)) = (&self.prev_kp, &self.prev_desc) {
                if kp.len() >= 8 && prev_kp.len() >= 8 {
                    let (pts_prev, pts_curr) = match_features(prev_desc, prev_kp, &desc, &kp);

                    if pts_prev.len() >= 8 {
                        if let Some(e) = estimate_essential(&pts_prev, &pts_curr, &self.k, 200) {
                            let (r, t) = recover_pose(&e, &pts_prev, &pts_curr, &self.k);

                            // Accumulate world pose
                            let new_t = mat3_vec(&self.r_world, &t);
                            self.t_world[0] += new_t[0];
                            self.t_world[1] += new_t[1];
                            self.t_world[2] += new_t[2];
                            self.r_world = mat3_mul(&self.r_world, &r);

                            // Triangulate sparse points
                            let pts_3d = triangulate_points(&pts_prev, &pts_curr, &self.k, &r, &t);
                            if !pts_3d.is_empty() {
                                // Transform to world frame
                                let world_pts: Vec<[f32; 3]> = pts_3d.iter().map(|p| {
                                    let pd = [p[0] as f64, p[1] as f64, p[2] as f64];
                                    let wp = mat3_vec(&self.r_world, &pd);
                                    [(wp[0] + self.t_world[0]) as f32,
                                     (wp[1] + self.t_world[1]) as f32,
                                     (wp[2] + self.t_world[2]) as f32]
                                }).collect();
                                self.point_cloud_parts.push(world_pts);
                            }
                        }
                    }
                }
            }

            self.prev_kp = Some(kp);
            self.prev_desc = Some(desc);
        }

        let pose = CameraPose {
            r: [
                [self.r_world[0][0] as f32, self.r_world[0][1] as f32, self.r_world[0][2] as f32],
                [self.r_world[1][0] as f32, self.r_world[1][1] as f32, self.r_world[1][2] as f32],
                [self.r_world[2][0] as f32, self.r_world[2][1] as f32, self.r_world[2][2] as f32],
            ],
            t: [self.t_world[0] as f32, self.t_world[1] as f32, self.t_world[2] as f32],
            frame_idx: self.frame_idx as u32,
        };
        self.poses.push(pose.clone());
        self.frame_idx += 1;
        pose
    }

    pub fn get_point_cloud(&self) -> Vec<[f32; 3]> {
        let mut cloud: Vec<[f32; 3]> = self.point_cloud_parts.iter().flatten().cloned().collect();
        if cloud.len() > 10 {
            // Remove outliers: clip to 3x median distance from centroid
            let n = cloud.len() as f32;
            let cx: f32 = cloud.iter().map(|p| p[0]).sum::<f32>() / n;
            let cy: f32 = cloud.iter().map(|p| p[1]).sum::<f32>() / n;
            let cz: f32 = cloud.iter().map(|p| p[2]).sum::<f32>() / n;
            let mut dists: Vec<f32> = cloud.iter()
                .map(|p| ((p[0]-cx).powi(2) + (p[1]-cy).powi(2) + (p[2]-cz).powi(2)).sqrt())
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = dists[dists.len() / 2];
            if med > 0.0 {
                cloud.retain(|p| {
                    ((p[0]-cx).powi(2) + (p[1]-cy).powi(2) + (p[2]-cz).powi(2)).sqrt() < 3.0 * med
                });
            }
        }
        cloud
    }

    pub fn get_result(&self) -> SlamResult {
        let cloud = self.get_point_cloud();
        SlamResult {
            success: self.poses.len() > 1 && cloud.len() > 10,
            poses: self.poses.clone(),
            point_cloud: cloud,
            intrinsics: self.k,
        }
    }

    pub fn poses(&self) -> &[CameraPose] { &self.poses }
    pub fn intrinsics(&self) -> &[[f64; 3]; 3] { &self.k }
}

// ─── Serialisation ───────────────────────────────────────────────────

pub fn pack_slam_poses(poses: &[CameraPose]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + poses.len() * 52);
    out.extend_from_slice(&(poses.len() as u32).to_be_bytes());
    for pose in poses {
        out.extend_from_slice(&pose.frame_idx.to_be_bytes());
        for row in &pose.r {
            for v in row { out.extend_from_slice(&v.to_be_bytes()); }
        }
        for v in &pose.t { out.extend_from_slice(&v.to_be_bytes()); }
    }
    out
}

pub fn pack_camera_k(k: &[[f64; 3]; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(72);
    for row in k {
        for v in row { out.extend_from_slice(&v.to_be_bytes()); }
    }
    out
}

