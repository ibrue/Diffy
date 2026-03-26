//! EgoCodec WASM — Diffy's egocentric video encoder compiled to WebAssembly.
//!
//! Exposes a simple API to JavaScript:
//!   1. `create_encoder(fps, width, height, quality, warmup)` → encoder handle
//!   2. `push_frame(handle, rgb_data)` — feed one RGB frame
//!   3. `encode(handle)` → Uint8Array of .ego bytes

mod background;
mod bitstream;
mod cycle_detector;
mod encoder;
mod residual_codec;

use wasm_bindgen::prelude::*;
use std::sync::Mutex;

static ENCODERS: Mutex<Vec<Option<encoder::EgoEncoder>>> = Mutex::new(Vec::new());

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
