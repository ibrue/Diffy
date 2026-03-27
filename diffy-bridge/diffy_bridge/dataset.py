"""
PyTorch Dataset for .dfy files.

Usage
-----
    from diffy_bridge.dataset import DiffyDataset
    import torchvision.transforms as T

    # Single file
    ds = DiffyDataset("recording.dfy")
    frame, idx = ds[0]   # (H×W×3 uint8 tensor, frame index)

    # Directory of .dfy files — auto-discovers and concatenates
    ds = DiffyDataset("recordings/")

    # With transforms
    ds = DiffyDataset(
        "recordings/",
        transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]),
    )

    # DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Callable, Optional

import numpy as np


class DiffyDataset:
    """
    PyTorch-compatible Dataset for .dfy files.

    Works without PyTorch installed — just returns numpy arrays.
    When PyTorch is available, returns tensors via the transform pipeline.

    Parameters
    ----------
    path : str or Path
        A .dfy file or a directory containing .dfy files (searched recursively).
    transform : callable, optional
        Applied to each frame (H×W×3 uint8 numpy array) before returning.
        Typically a torchvision.transforms pipeline.
    clip_len : int, optional
        If set, __getitem__ returns a clip of clip_len consecutive frames
        instead of a single frame.
    clip_stride : int
        Stride between clips when clip_len is set (default 1).
    """

    def __init__(
        self,
        path: Union[str, Path],
        transform: Optional[Callable] = None,
        clip_len: Optional[int] = None,
        clip_stride: int = 1,
    ) -> None:
        self.transform = transform
        self.clip_len = clip_len
        self.clip_stride = clip_stride

        path = Path(path)
        if path.is_dir():
            self._files = sorted(path.rglob("*.dfy"))
        elif path.suffix == ".dfy":
            self._files = [path]
        else:
            raise ValueError(f"Expected a .dfy file or directory, got {path}")

        if not self._files:
            raise FileNotFoundError(f"No .dfy files found under {path}")

        # Build index: list of (file_index, frame_index)
        self._index: list[tuple[int, int]] = []
        self._readers: list = [None] * len(self._files)  # lazy-loaded

        from .reader import DiffyReader

        self._DiffyReader = DiffyReader
        self._build_index()

    def _build_index(self) -> None:
        from .reader import DiffyReader

        for fi, fpath in enumerate(self._files):
            r = DiffyReader(fpath)
            n = len(r)
            r.close()

            if self.clip_len is None:
                for frame_i in range(n):
                    self._index.append((fi, frame_i))
            else:
                max_start = n - self.clip_len
                for start in range(0, max_start + 1, self.clip_stride):
                    self._index.append((fi, start))

    def _get_reader(self, file_idx: int):
        if self._readers[file_idx] is None:
            self._readers[file_idx] = self._DiffyReader(self._files[file_idx])
        return self._readers[file_idx]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        file_idx, frame_idx = self._index[idx]
        reader = self._get_reader(file_idx)

        if self.clip_len is None:
            frame = reader[frame_idx]
            if self.transform:
                frame = self.transform(frame)
            return frame, frame_idx
        else:
            clip = reader[frame_idx : frame_idx + self.clip_len]
            if self.transform:
                clip = [self.transform(f) for f in clip]
            return clip, frame_idx

    def __repr__(self) -> str:
        return (
            f"DiffyDataset({len(self._files)} file(s), "
            f"{len(self)} {'clips' if self.clip_len else 'frames'})"
        )
