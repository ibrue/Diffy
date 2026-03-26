# Diffy Project Notes

## Design Language
User loves the current dark terminal aesthetic:
- Black background (#000)
- Muted grays (#555, #666, #777) — NOT bright white
- Monospace fonts (SF Mono, Fira Code, Cascadia Code)
- Georgia serif for "Diffy" logo at #666 (not bright white #fff)
- Dashed borders (#333), minimal UI, no color except subtle accents
- ASCII progress bars, terminal feel
- Keep everything subdued and cohesive — nothing should pop or feel "bright"

## Business Model
Diffy is NOT open source. Exploring monetization. Do not add open-source language or encourage contributions in any new files. Existing LICENSE/README may need updating.

## Technical Notes
- Browser encoder runs via Pyodide (Python in WASM)
- No cv2 or lzma available in Pyodide — use PIL and zlib instead
- Background model uses Welford online accumulation (constant memory, no frame buffering)
- `frame_js.to_py()` returns memoryview — always wrap in `bytes()` before `np.frombuffer`
- All codec files are inlined in index.html as JS template literals (no CDN fetch)
