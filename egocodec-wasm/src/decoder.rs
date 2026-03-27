/// DiffyDecoder: reconstruct frames from a .dfy bitstream.
///
/// Supports:
///   - CYCLE_CANON with FLAG_TEMPORAL=0 (legacy, YCbCr residual — WASM encoder output)
///   - CYCLE_DELTA (encode_residual format from WASM encoder)
///   - FRAME_SKIP (clone from canonical)
///
/// Returns frames one at a time via next_frame() to keep memory bounded.
/// Canonical cycles are pre-decoded and cached; delta cycles are decoded on demand.

use crate::bitstream::MAGIC;
use crate::residual_codec::{decode_residual_payload, zlib_decompress};

// Chunk type constants (match bitstream.py)
const CHUNK_BACKGROUND: u8 = 0x01;
const CHUNK_CYCLE_CANON: u8 = 0x02;
const CHUNK_CYCLE_DELTA: u8 = 0x03;
const CHUNK_FRAME_SKIP: u8 = 0x05;
const CHUNK_METADATA: u8 = 0x06;

/// A single cycle entry in decode order.
enum CycleEntry {
    /// Pre-decoded canonical cycle — frame_idx iterates over it.
    Canon(usize),
    /// Frame-skip: alias first N frames of a canonical.
    Skip(usize, usize), // (canon_idx, frame_count)
    /// Delta cycle payload; decoded on first use.
    Delta(Vec<u8>),
}

pub struct DiffyDecoder {
    pub fps: f32,
    pub width: usize,
    pub height: usize,
    pub total_frames: u64,

    background: Vec<u8>,                  // RGB u8 H*W*3
    canonicals: Vec<Vec<Vec<u8>>>,        // [canon_idx][frame_idx] = RGB u8 H*W*3

    entries: Vec<CycleEntry>,             // ordered playback entries
    entry_idx: usize,                     // current entry
    frame_in_entry: usize,               // current frame within that entry
    current_delta_frames: Vec<Vec<u8>>,  // decoded delta cycle (reused)
}

impl DiffyDecoder {
    pub fn new(data: &[u8]) -> Result<Self, String> {
        // ── Header ────────────────────────────────────────────────────────────
        if data.len() < 19 {
            return Err("file too short".to_string());
        }
        if &data[..4] != MAGIC {
            return Err(format!("bad magic: {:?}", &data[..4]));
        }
        let mut pos = 4;
        let total_frames = u64::from_be_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let fps = f32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let width  = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let height = u16::from_be_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let _flags = data[pos];
        pos += 1;

        // ── Parse chunks ──────────────────────────────────────────────────────
        let mut background: Vec<u8> = Vec::new();
        let mut canonicals: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut noncanon_chunks: Vec<(u8, Vec<u8>)> = Vec::new(); // (chunk_type, payload)
        let mut cycle_map: Vec<[usize; 2]> = Vec::new();

        while pos + 6 <= data.len() {
            let chunk_type  = data[pos]; pos += 1;
            let compressed  = data[pos] != 0; pos += 1;
            let length = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + length > data.len() { break; }
            let raw = &data[pos..pos + length];
            pos += length;

            let payload = if compressed { zlib_decompress(raw) } else { raw.to_vec() };

            match chunk_type {
                CHUNK_METADATA => {
                    if let Ok(s) = std::str::from_utf8(&payload) {
                        cycle_map = parse_cycle_map_json(s);
                    }
                }
                CHUNK_BACKGROUND => {
                    background = decode_jpeg_rgb(&payload)
                        .unwrap_or_else(|_| vec![128u8; width * height * 3]);
                }
                CHUNK_CYCLE_CANON => {
                    let frames = decode_legacy_cycle(&payload, &background, width, height)
                        .unwrap_or_default();
                    canonicals.push(frames);
                }
                CHUNK_CYCLE_DELTA | CHUNK_FRAME_SKIP => {
                    noncanon_chunks.push((chunk_type, payload));
                }
                _ => {}
            }
        }

        // ── Build ordered entry list ──────────────────────────────────────────
        let entries = build_entries(&cycle_map, &canonicals, &noncanon_chunks);

        Ok(DiffyDecoder {
            fps,
            width,
            height,
            total_frames,
            background,
            canonicals,
            entries,
            entry_idx: 0,
            frame_in_entry: 0,
            current_delta_frames: Vec::new(),
        })
    }

    /// Return the next decoded RGB u8 frame (H*W*3), or None when exhausted.
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        loop {
            if self.entry_idx >= self.entries.len() {
                return None;
            }
            match &self.entries[self.entry_idx] {
                CycleEntry::Canon(idx) => {
                    let idx = *idx;
                    let cycle = &self.canonicals[idx];
                    if self.frame_in_entry < cycle.len() {
                        let frame = cycle[self.frame_in_entry].clone();
                        self.frame_in_entry += 1;
                        return Some(frame);
                    }
                }
                CycleEntry::Skip(canon_idx, frame_count) => {
                    let canon_idx = *canon_idx;
                    let frame_count = *frame_count;
                    if self.frame_in_entry < frame_count {
                        let cycle = &self.canonicals[canon_idx];
                        let frame = if self.frame_in_entry < cycle.len() {
                            cycle[self.frame_in_entry].clone()
                        } else if !cycle.is_empty() {
                            cycle[cycle.len() - 1].clone()
                        } else {
                            self.background.clone()
                        };
                        self.frame_in_entry += 1;
                        return Some(frame);
                    }
                }
                CycleEntry::Delta(payload) => {
                    // Decode the whole delta cycle on first frame
                    if self.frame_in_entry == 0 {
                        let payload = payload.clone();
                        self.current_delta_frames =
                            decode_delta_cycle(&payload, &self.canonicals, &self.background,
                                               self.width, self.height);
                    }
                    if self.frame_in_entry < self.current_delta_frames.len() {
                        let frame = std::mem::take(
                            &mut self.current_delta_frames[self.frame_in_entry]);
                        self.frame_in_entry += 1;
                        return Some(frame);
                    }
                    self.current_delta_frames.clear();
                }
            }
            // Advance to next entry
            self.entry_idx += 1;
            self.frame_in_entry = 0;
        }
    }
}

// ── Chunk decoders ────────────────────────────────────────────────────────────

/// Decode a JPEG background to RGB u8 bytes.
fn decode_jpeg_rgb(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = dec.decode()
        .map_err(|e| format!("JPEG decode: {}", e))?;
    let info = dec.info().ok_or("no JPEG info")?;
    // Convert to RGB if needed
    use jpeg_decoder::PixelFormat;
    let rgb = match info.pixel_format {
        PixelFormat::RGB24 => pixels,
        PixelFormat::L8 => pixels.iter().flat_map(|&v| [v, v, v]).collect(),
        _ => return Err("unsupported JPEG pixel format".to_string()),
    };
    Ok(rgb)
}

/// Decode a CYCLE_CANON legacy payload (FLAG_TEMPORAL=0).
/// Layout: [4B n][1B flags][per-frame: [4B size][residual payload]]
fn decode_legacy_cycle(
    payload: &[u8],
    bg: &[u8],
    _width: usize,
    _height: usize,
) -> Result<Vec<Vec<u8>>, String> {
    if payload.len() < 5 {
        return Ok(Vec::new());
    }
    let n = u32::from_be_bytes(payload[..4].try_into().unwrap()) as usize;
    let flags = payload[4];

    if flags & 0x01 != 0 {
        // FLAG_TEMPORAL — temporal/P-frame encoding used by Python encoder.
        // Fall back: return background repeated so the file still loads.
        let _w = if bg.len() > 0 { bg.len() } else { 1 };
        return Ok(vec![bg.to_vec(); n.min(1)]);
    }

    let mut frames = Vec::with_capacity(n);
    let mut offset = 5usize;

    for _ in 0..n {
        if offset + 4 > payload.len() { break; }
        let size = u32::from_be_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + size > payload.len() { break; }

        match decode_residual_payload(&payload[offset..offset + size]) {
            Ok((residual, h, w)) => {
                let frame: Vec<u8> = bg.iter().zip(residual.iter())
                    .map(|(&b, &r)| (b as i32 + r as i32).clamp(0, 255) as u8)
                    .collect();
                // If bg is larger (padding), truncate to h*w*3
                let frame_size = h * w * 3;
                let frame = if frame.len() > frame_size { frame[..frame_size].to_vec() } else { frame };
                frames.push(frame);
            }
            Err(_) => {
                frames.push(bg.to_vec());
            }
        }
        offset += size;
    }
    Ok(frames)
}

/// Decode a CYCLE_DELTA payload.
/// Layout: [2B canon_idx][4B n][per-frame: [4B size][residual payload]]
fn decode_delta_cycle(
    payload: &[u8],
    canonicals: &[Vec<Vec<u8>>],
    background: &[u8],
    _width: usize,
    _height: usize,
) -> Vec<Vec<u8>> {
    if payload.len() < 6 { return Vec::new(); }
    let canon_idx = u16::from_be_bytes([payload[0], payload[1]]) as usize;
    let n = u32::from_be_bytes(payload[2..6].try_into().unwrap()) as usize;
    let canon = canonicals.get(canon_idx);

    let mut frames = Vec::with_capacity(n);
    let mut offset = 6usize;

    for i in 0..n {
        if offset + 4 > payload.len() { break; }
        let size = u32::from_be_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + size > payload.len() { break; }

        let ref_frame: &[u8] = match canon {
            Some(c) if !c.is_empty() => {
                let fi = i.min(c.len() - 1);
                &c[fi]
            }
            _ => background,
        };

        match decode_residual_payload(&payload[offset..offset + size]) {
            Ok((residual, h, w)) => {
                let frame: Vec<u8> = ref_frame.iter().zip(residual.iter())
                    .map(|(&c, &r)| (c as i32 + r as i32).clamp(0, 255) as u8)
                    .collect();
                let frame_size = h * w * 3;
                let frame = if frame.len() > frame_size { frame[..frame_size].to_vec() } else { frame };
                frames.push(frame);
            }
            Err(_) => {
                frames.push(ref_frame.to_vec());
            }
        }
        offset += size;
    }
    frames
}

// ── Entry list construction ────────────────────────────────────────────────────

fn build_entries(
    cycle_map: &[[usize; 2]],
    canonicals: &[Vec<Vec<u8>>],
    noncanon_chunks: &[(u8, Vec<u8>)],
) -> Vec<CycleEntry> {
    let mut entries = Vec::new();

    if cycle_map.is_empty() {
        // Old files without cycle_map: canonicals then non-canonicals
        for i in 0..canonicals.len() {
            entries.push(CycleEntry::Canon(i));
        }
        for (chunk_type, payload) in noncanon_chunks {
            match *chunk_type {
                CHUNK_FRAME_SKIP => {
                    if payload.len() >= 4 {
                        let ci = u16::from_be_bytes([payload[0], payload[1]]) as usize;
                        let fc = u16::from_be_bytes([payload[2], payload[3]]) as usize;
                        entries.push(CycleEntry::Skip(ci, fc));
                    }
                }
                CHUNK_CYCLE_DELTA => {
                    entries.push(CycleEntry::Delta(payload.clone()));
                }
                _ => {}
            }
        }
        return entries;
    }

    for &[type_flag, idx] in cycle_map {
        if type_flag == 0 {
            // Canonical
            if idx < canonicals.len() {
                entries.push(CycleEntry::Canon(idx));
            }
        } else {
            // Non-canonical
            if idx < noncanon_chunks.len() {
                let (chunk_type, payload) = &noncanon_chunks[idx];
                match *chunk_type {
                    CHUNK_FRAME_SKIP => {
                        if payload.len() >= 4 {
                            let ci = u16::from_be_bytes([payload[0], payload[1]]) as usize;
                            let fc = u16::from_be_bytes([payload[2], payload[3]]) as usize;
                            entries.push(CycleEntry::Skip(ci, fc));
                        }
                    }
                    CHUNK_CYCLE_DELTA => {
                        entries.push(CycleEntry::Delta(payload.clone()));
                    }
                    _ => {}
                }
            }
        }
    }
    entries
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

fn parse_cycle_map_json(s: &str) -> Vec<[usize; 2]> {
    // Parse "cycle_map": [[type, idx], ...] from the metadata JSON
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(s) {
        if let Some(arr) = val.get("cycle_map").and_then(|v| v.as_array()) {
            return arr.iter().filter_map(|entry| {
                let e = entry.as_array()?;
                if e.len() < 2 { return None; }
                let t = e[0].as_u64()? as usize;
                let i = e[1].as_u64()? as usize;
                Some([t, i])
            }).collect();
        }
    }
    Vec::new()
}
