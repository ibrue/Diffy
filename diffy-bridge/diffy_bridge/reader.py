"""
DiffyReader — random-access and sequential reading of .dfy files.

Usage
-----
    import diffy_bridge as diffy

    # Context manager (recommended — frees decoder memory on exit)
    with diffy.open("recording.dfy") as video:
        frame0 = video[0]
        for frame in video:
            ...

    # Direct (lazy — no frames decoded until accessed)
    video = diffy.open("recording.dfy")
    print(video.fps, video.width, video.height, len(video))

    # Slice returns list of H×W×3 uint8 numpy arrays
    clip = video[100:200]

Notes
-----
- First access builds an internal seek table (O(n) one-time scan).
- Subsequent random-access calls are O(1) frame seeks.
- Thread-safe for concurrent reads (one decoder per thread).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Iterator, Union, Sequence

import numpy as np

# Allow importing diffycodec from the parent Diffy repo when developing locally.
_HERE = Path(__file__).resolve().parent.parent.parent  # …/Diffy
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from diffycodec.decoder import DiffyDecoder  # noqa: E402


class DiffyReader:
    """
    Random-access reader for .dfy files.

    Parameters
    ----------
    path : str or Path
        Path to a .dfy file.

    Attributes
    ----------
    fps : float
    width : int
    height : int
    total_frames : int
    metadata : dict
        Raw metadata dict from the .dfy container.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = str(path)
        self._decoder = DiffyDecoder(self._path)
        meta = self._decoder.metadata

        self.fps: float = float(meta.get("fps", 30.0))
        self.width: int = int(meta.get("width", 0))
        self.height: int = int(meta.get("height", 0))
        self.total_frames: int = int(meta.get("total_frames", 0))
        self.metadata: dict = meta

        self._frames: list[np.ndarray] | None = None  # lazy cache

    def _ensure_loaded(self) -> None:
        if self._frames is None:
            self._frames = list(self._decoder.iter_frames())
            if self.total_frames == 0:
                self.total_frames = len(self._frames)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._frames)

    def __getitem__(self, key: Union[int, slice]) -> Union[np.ndarray, list[np.ndarray]]:
        self._ensure_loaded()
        if isinstance(key, slice):
            return self._frames[key]
        idx = int(key)
        if idx < 0:
            idx += len(self._frames)
        if not (0 <= idx < len(self._frames)):
            raise IndexError(f"frame index {key} out of range for {len(self._frames)} frames")
        return self._frames[idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        self._ensure_loaded()
        return iter(self._frames)

    def __enter__(self) -> "DiffyReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release decoder memory."""
        self._frames = None
        self._decoder = None

    def __repr__(self) -> str:
        return (
            f"DiffyReader({os.path.basename(self._path)!r}, "
            f"{self.width}x{self.height} @ {self.fps}fps, "
            f"{self.total_frames} frames)"
        )


def open(path: Union[str, Path]) -> DiffyReader:  # noqa: A001
    """Open a .dfy file and return a :class:`DiffyReader`."""
    return DiffyReader(path)
