/// .dfy container format writer.
///
/// Layout:
/// [4B magic "DFY\x01"]
/// [8B u64 total_frames][4B f32 fps][2B u16 width][2B u16 height][1B flags]
/// chunks: [1B type][1B compressed][4B length][payload]

use crate::residual_codec::zlib_compress;

pub const MAGIC: &[u8; 4] = b"DFY\x01";

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum ChunkType {
    Background = 0x01,
    CycleCanon = 0x02,
    CycleDelta = 0x03,
    ImuBlock = 0x04,
    FrameSkip = 0x05,
    Metadata = 0x06,
    Codebook = 0x07,
    SplatModel = 0x08,
    Scene3dgs = 0x09,
    SlamPoses = 0x0A,
    CameraK = 0x0B,
}

pub struct BitstreamWriter {
    data: Vec<u8>,
}

impl BitstreamWriter {
    pub fn new(total_frames: u64, fps: f32, width: u16, height: u16, has_imu: bool) -> Self {
        let mut data = Vec::with_capacity(1024 * 1024);
        // Magic
        data.extend_from_slice(MAGIC);
        // Header
        data.extend_from_slice(&total_frames.to_be_bytes());
        data.extend_from_slice(&fps.to_be_bytes());
        data.extend_from_slice(&width.to_be_bytes());
        data.extend_from_slice(&height.to_be_bytes());
        data.push(if has_imu { 0x01 } else { 0x00 });
        Self { data }
    }

    pub fn write_chunk(&mut self, chunk_type: ChunkType, payload: &[u8], compress: bool) {
        let final_payload = if compress {
            zlib_compress(payload)
        } else {
            payload.to_vec()
        };
        self.data.push(chunk_type as u8);
        self.data.push(if compress { 1 } else { 0 });
        self.data.extend_from_slice(&(final_payload.len() as u32).to_be_bytes());
        self.data.extend_from_slice(&final_payload);
    }

    pub fn finish(self) -> Vec<u8> {
        self.data
    }

    pub fn bytes_written(&self) -> usize {
        self.data.len()
    }
}
