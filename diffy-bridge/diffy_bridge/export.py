"""
Export .dfy files to standard formats: mp4, jpeg frames, png frames.

Usage
-----
    import diffy_bridge as diffy

    diffy.export("recording.dfy", "output.mp4")
    diffy.export("recording.dfy", "frames/", fmt="jpeg")
    diffy.export("recording.dfy", "frames/", fmt="png")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

from .reader import DiffyReader


def export(
    src: Union[str, Path],
    dst: Union[str, Path],
    fmt: str = "auto",
    quality: int = 95,
    on_progress=None,
) -> None:
    """
    Export a .dfy file to a standard format.

    Parameters
    ----------
    src : str or Path
        Input .dfy file.
    dst : str or Path
        Output path. Pass a directory for frame sequences, a file for video.
    fmt : str
        "auto"  — infer from dst extension (.mp4 → video, .jpg/.jpeg → jpeg,
                  .png → png, directory → jpeg frames)
        "mp4"   — H.264 video via imageio / opencv
        "jpeg"  — JPEG frame sequence in dst directory
        "png"   — PNG frame sequence in dst directory
    quality : int
        JPEG quality (1–100, default 95). Ignored for PNG/mp4.
    on_progress : callable, optional
        Called as on_progress(frame_index, total_frames).
    """
    src_path = Path(src)
    dst_path = Path(dst)

    if fmt == "auto":
        if dst_path.suffix.lower() in (".mp4", ".mov", ".avi"):
            fmt = "mp4"
        elif dst_path.suffix.lower() in (".jpg", ".jpeg"):
            fmt = "jpeg"
        elif dst_path.suffix.lower() == ".png":
            fmt = "png"
        elif dst_path.is_dir() or not dst_path.suffix:
            fmt = "jpeg"
        else:
            fmt = "mp4"

    with DiffyReader(src_path) as video:
        n = len(video)
        if fmt == "mp4":
            _export_mp4(video, dst_path, on_progress)
        elif fmt in ("jpeg", "jpg"):
            _export_frames(video, dst_path, "jpeg", quality, on_progress)
        elif fmt == "png":
            _export_frames(video, dst_path, "png", quality, on_progress)
        else:
            raise ValueError(f"Unknown format {fmt!r}. Use 'mp4', 'jpeg', or 'png'.")


def _export_mp4(video: DiffyReader, dst: Path, on_progress) -> None:
    try:
        import imageio
        writer = imageio.get_writer(str(dst), fps=video.fps, codec="libx264", quality=8)
        for i, frame in enumerate(video):
            writer.append_data(frame)
            if on_progress:
                on_progress(i, len(video))
        writer.close()
    except ImportError:
        try:
            import cv2
            h, w = video.height, video.width
            out = cv2.VideoWriter(
                str(dst),
                cv2.VideoWriter_fourcc(*"mp4v"),
                video.fps,
                (w, h),
            )
            for i, frame in enumerate(video):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if on_progress:
                    on_progress(i, len(video))
            out.release()
        except ImportError:
            raise ImportError(
                "MP4 export requires imageio[ffmpeg] or opencv-python. "
                "Install with: pip install imageio[ffmpeg]"
            )


def _export_frames(
    video: DiffyReader,
    dst: Path,
    fmt: str,
    quality: int,
    on_progress,
) -> None:
    from PIL import Image

    dst.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if fmt == "jpeg" else "png"
    digits = len(str(len(video) - 1))

    for i, frame in enumerate(video):
        img = Image.fromarray(frame)
        out_path = dst / f"frame_{i:0{digits}d}.{ext}"
        save_kwargs = {"quality": quality} if fmt == "jpeg" else {}
        img.save(out_path, **save_kwargs)
        if on_progress:
            on_progress(i, len(video))
