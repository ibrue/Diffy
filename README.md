# Diffy

**Difference video compression — exploits what generic codecs ignore.**

Raw factory video is 1.6 MB/frame. Diffy gets it to ~8 KB/frame by exploiting three priors that generic codecs (H.265, AV1) ignore:

1. **Background stationarity** — the factory floor doesn’t move. Encode it once as a JPEG keyframe (~200 KB), never again.
2. **Cycle periodicity** — workers repeat the same motion thousands of times per shift. Store one canonical cycle, then only the tiny per-cycle deltas.
3. **Sparse foreground** — only the worker’s hands and held objects change. Skip the 99% of pixels that are background.

**Result:** 8 hours of 1080p30 → under 10 MB. H.265 gets you to ~5 GB.

---

## Try it — diffy.tech

Go to [diffy.tech](https://diffy.tech). No install. Runs entirely in your browser (WebAssembly + Pyodide).

1. Wait ~20 seconds on first load (downloads numpy/scipy/Pillow into WebAssembly — cached after)
2. Drop a video file, folder of frames, or `.zip` of images onto the drop zone
3. Watch the ASCII progress bar advance through **background model → frame encoding → cycle compression**
4. Download your `.dfy` file

> Videos are downsampled to max 960px wide for encoding speed. The Python CLI below uses full resolution.

No data leaves your machine. No account. No server.

For full-length videos (> 10 min) or batch processing, use the Python CLI below.

---

## What is `.dfy`?

The `.dfy` container format:

```
[4B magic "DFY\x01"]
[header: total_frames, fps, width, height, flags]
--- chunks ---
BACKGROUND   0x01   JPEG background keyframe (stored once)
CYCLE_CANON  0x02   Canonical work cycle (temporal I/P coded)
CYCLE_DELTA  0x03   Per-cycle deviation vs canonical
CYCLE_SKIP   0x05   Clone pointer (4 bytes — identical cycle)
METADATA     0x06   JSON blob
CODEBOOK     0x07   Optional VQ codebook (float16 centroids)
IMU_BLOCK    0x04   IMU quaternions (f16, zlib compressed)
```

Every frame is uniquely reconstructable. Training mode preserves per-cycle pixel variation (tool slip, speed jitter) as temporal deltas — signal for physical AI models.

---

## Python library

```bash
pip install -e .
```

**Encode a video:**
```python
from egocodec.encoder import EgoEncoder

enc = EgoEncoder('output.dfy', fps=30, width=1920, height=1080, quality=25)

for frame in your_frame_source:          # uint8 H×W×3 numpy arrays
    enc.push_frame(frame)

enc.encode()
print(f"Written {enc.bytes_written / 1e6:.1f} MB")
```

**From a video file (requires opencv-python):**
```python
from egocodec.encoder import EgoEncoder
EgoEncoder.from_video('factory_shift.mp4', 'factory_shift.dfy', quality=25)
```

**Decode:**
```python
from egocodec.decoder import EgoDecoder

dec = EgoDecoder('factory_shift.dfy')
for frame in dec.iter_frames():          # yields uint8 H×W×3 numpy arrays
    process(frame)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quality` | 25 | Residual codec quality 1–100 (lower = smaller) |
| `warmup_frames` | 300 | Frames used to build background model |
| `use_temporal` | True | Inter-frame prediction within cycles |
| `use_bbox` | True | Encode only foreground bounding box per frame |
| `use_vq` | False | Train VQ codebook for ~27× extra reduction |

---

## Compression calculator

[diffy.tech/calc](https://diffy.tech/calc) — estimate file size and upload time for your specific setup (workers, hours/day, resolution, network speed, task repetitiveness).

---

## Dataset testing

To test on [builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K) without downloading 16 TB:

```python
from datasets import load_dataset
import numpy as np
from egocodec.encoder import EgoEncoder

# Stream just the evaluation split (5.49 GB)
ds = load_dataset("builddotai/Egocentric-10K-Evaluation", streaming=True, split="train")

for sample in ds.take(5):
    frames = sample['frames']  # list of uint8 H×W×3 arrays
    enc = EgoEncoder(f"sample_{sample['id']}.dfy", fps=30,
                     width=frames[0].shape[1], height=frames[0].shape[0])
    for f in frames:
        enc.push_frame(f)
    enc.encode()
```

---

## Architecture

```
egocodec/
  encoder.py        — EgoEncoder: top-level encode pipeline
  decoder.py        — EgoDecoder: reconstruct frames from .dfy
  bitstream.py      — BitstreamWriter / BitstreamReader (.dfy container)
  background.py     — BackgroundModel: running median + EMA update
  cycle_detector.py — CycleDetector: energy-valley segmentation
  temporal_codec.py — I/P frame coding within cycles (zlib)
  residual_codec.py — DCT + RLE + zlib residual codec
  vq_codec.py       — Optional VQ codebook (numpy k-means++)
  imu.py            — IMU quaternion integration + frame stabilisation
```

---

## Run tests

```bash
pip install pytest numpy scipy Pillow
pytest tests/
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Copyright 2026 ibrue

---

## Contributing

PRs welcome. Key areas:

- **Decoder web UI** — drop a `.dfy` file, watch the video play back in the browser
- **Streaming encoder** — encode while recording, don’t buffer all frames
- **Better cycle detection** — optical flow instead of frame-diff energy
- **Range coder** — replace zlib with arithmetic coding for ~15% extra gain
- **Benchmarks** — compression ratio and PSNR vs H.265 on Egocentric-10K
