"""
diffy-bridge — transparent access to .dfy files as standard video/frames.

Quick start
-----------
    import diffy_bridge as diffy

    # Iterate frames as numpy arrays (H×W×3 uint8)
    for frame in diffy.open("recording.dfy"):
        process(frame)

    # Random access
    video = diffy.open("recording.dfy")
    frame = video[42]
    frames = video[10:20]

    # Export to standard video
    diffy.export("recording.dfy", "output.mp4")

    # PyTorch / ML dataset
    from diffy_bridge.dataset import DiffyDataset
    ds = DiffyDataset("recordings/", transform=my_transform)

    # CLI
    #   diffy export recording.dfy output.mp4
    #   diffy info   recording.dfy
    #   diffy frames recording.dfy --out frames/
"""

from .reader import DiffyReader, open   # noqa: F401
from .export import export               # noqa: F401

__version__ = "0.1.0"
__all__ = ["open", "export", "DiffyReader"]
