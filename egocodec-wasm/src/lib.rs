//! DiffyCodec WASM — Diffy's difference video encoder compiled to WebAssembly.
//!
//! Exposes a simple API to JavaScript:
//!   1. `create_encoder(fps, width, height, quality, warmup)` → encoder handle
//!   2. `push_frame(handle, rgb_data)` — feed one RGB frame
//!   3. `encode(handle)` → Uint8Array of .dfy bytes

mod background;
mod bitstream;
mod cycle_detector;
mod decoder;
mod encoder;
mod residual_codec;
pub mod slam;
#[cfg(test)]
mod tests;

use wasm_bindgen::prelude::*;
use std::sync::Mutex;

static ENCODERS: Mutex<Vec<Option<encoder::EgoEncoder>>> = Mutex::new(Vec::new());
static DECODERS: Mutex<Vec<Option<decoder::DiffyDecoder>>> = Mutex::new(Vec::new());

#[wasm_bindgen]
pub fn create_encoder(fps: f32, width: u32, height: u32, quality: u8, warmup: u32) -> u32 {
    let enc = encoder::EgoEncoder::new(
        fps,
        width as usize,
        height as usize,
        quality,
        warmup as usize,
    );
    let mut encoders = ENCODERS.lock().unwrap();
    let handle = encoders.len() as u32;
    encoders.push(Some(enc));
    handle
}

#[wasm_bindgen]
pub fn push_frame(handle: u32, rgb_data: &[u8]) {
    let mut encoders = ENCODERS.lock().unwrap();
    if let Some(Some(enc)) = encoders.get_mut(handle as usize) {
        enc.push_frame(rgb_data);
    }
}

#[wasm_bindgen]
pub fn encode(handle: u32) -> Vec<u8> {
    let mut encoders = ENCODERS.lock().unwrap();
    if let Some(Some(enc)) = encoders.get_mut(handle as usize) {
        enc.encode()
    } else {
        Vec::new()
    }
}

#[wasm_bindgen]
pub fn free_encoder(handle: u32) {
    let mut encoders = ENCODERS.lock().unwrap();
    if let Some(slot) = encoders.get_mut(handle as usize) {
        *slot = None;
    }
}

/// Returns the total frames pushed so far.
#[wasm_bindgen]
pub fn get_frame_count(handle: u32) -> u32 {
    let encoders = ENCODERS.lock().unwrap();
    if let Some(Some(enc)) = encoders.get(handle as usize) {
        enc.total_frames() as u32
    } else {
        0
    }
}

// ── Decoder API ───────────────────────────────────────────────────────────────

/// Create a decoder from raw .dfy bytes.  Returns a handle, or panics on error.
#[wasm_bindgen]
pub fn create_decoder(bytes: &[u8]) -> u32 {
    let dec = decoder::DiffyDecoder::new(bytes).expect("create_decoder failed");
    let mut decoders = DECODERS.lock().unwrap();
    let handle = decoders.len() as u32;
    decoders.push(Some(dec));
    handle
}

#[wasm_bindgen]
pub fn decoder_total_frames(handle: u32) -> u32 {
    let decoders = DECODERS.lock().unwrap();
    if let Some(Some(d)) = decoders.get(handle as usize) {
        d.total_frames as u32
    } else {
        0
    }
}

#[wasm_bindgen]
pub fn decoder_fps(handle: u32) -> f32 {
    let decoders = DECODERS.lock().unwrap();
    if let Some(Some(d)) = decoders.get(handle as usize) { d.fps } else { 30.0 }
}

#[wasm_bindgen]
pub fn decoder_width(handle: u32) -> u32 {
    let decoders = DECODERS.lock().unwrap();
    if let Some(Some(d)) = decoders.get(handle as usize) { d.width as u32 } else { 0 }
}

#[wasm_bindgen]
pub fn decoder_height(handle: u32) -> u32 {
    let decoders = DECODERS.lock().unwrap();
    if let Some(Some(d)) = decoders.get(handle as usize) { d.height as u32 } else { 0 }
}

/// Return the next RGB frame (H*W*3 u8), or empty Vec if exhausted.
#[wasm_bindgen]
pub fn decoder_next_frame(handle: u32) -> Vec<u8> {
    let mut decoders = DECODERS.lock().unwrap();
    if let Some(Some(d)) = decoders.get_mut(handle as usize) {
        d.next_frame().unwrap_or_default()
    } else {
        Vec::new()
    }
}

#[wasm_bindgen]
pub fn free_decoder(handle: u32) {
    let mut decoders = DECODERS.lock().unwrap();
    if let Some(slot) = decoders.get_mut(handle as usize) {
        *slot = None;
    }
}
