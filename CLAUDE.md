# Diffy Project Notes

## Branding
- Diffy = differential video compression
- File extension: `.dfy` (not `.ego`)
- Container magic bytes: `DFY\x01`
- Not just egocentric — works for any repetitive/cyclic video (factory, warehouse, security)
- Python module: `diffycodec`, classes `DiffyEncoder`/`DiffyDecoder`

## Design Language
Dark terminal aesthetic with readable text:
- Black background (#000)
- Body text #ccc, labels #bbb, subtitles #999, secondary #999
- Georgia serif for "Diffy" logo at #aaa
- Monospace fonts (SF Mono, Fira Code, Cascadia Code)
- Dashed borders (#555), minimal UI, muted accent colors (#7aaa7a green, #aaaa7a yellow)
- ASCII progress bars, terminal feel
- Readable but subdued — nothing should pop or feel "bright"

## Business Model
Diffy is open source (Apache 2.0). Good for resume/portfolio.

## Workflow
- After completing feature work, always merge the dev branch into `main` and push, so changes are live on the website for testing.
- Dev branch: `claude/egocentric-video-compression-sjwr2` → merge to `main` when done.

## Technical Notes
- Browser encoder runs via Pyodide (Python in WASM)
- No cv2 or lzma available in Pyodide — use PIL and zlib instead
- Background model uses Welford online accumulation (constant memory, no frame buffering)
- `frame_js.to_py()` returns memoryview — always wrap in `bytes()` before `np.frombuffer`
- All codec files are inlined in index.html as JS template literals (no CDN fetch)
- Rust WASM encoder exists in egocodec-wasm/ (317KB compiled, near-native speed)
