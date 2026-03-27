"""
diffy CLI — inspect and export .dfy files from the terminal.

Commands
--------
    diffy info   recording.dfy
    diffy export recording.dfy output.mp4
    diffy export recording.dfy frames/ --fmt jpeg
    diffy export recording.dfy frames/ --fmt png
"""

import argparse
import sys
from pathlib import Path


def cmd_info(args) -> None:
    from .reader import DiffyReader

    with DiffyReader(args.file) as v:
        meta = v.metadata
        n = len(v)
        dur = n / v.fps if v.fps else 0
        size = Path(args.file).stat().st_size
        raw = n * v.width * v.height * 3

        print(f"\n  {Path(args.file).name}")
        print(f"  {'─' * 40}")
        print(f"  resolution   {v.width}×{v.height}")
        print(f"  fps          {v.fps}")
        print(f"  frames       {n}")
        print(f"  duration     {dur:.1f}s")
        print(f"  file size    {size/1e6:.2f} MB")
        print(f"  raw equiv    {raw/1e6:.0f} MB")
        print(f"  ratio        {raw//size}:1")
        if "mode" in meta:
            print(f"  mode         {meta['mode']}")
        print()


def cmd_export(args) -> None:
    from .export import export

    dst = Path(args.output)
    fmt = args.fmt or "auto"

    total_ref = [0]

    def progress(i, total):
        total_ref[0] = total
        pct = int(i / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r  [{bar}] {i}/{total}", end="", flush=True)

    print(f"  exporting {Path(args.file).name} → {dst}")
    export(args.file, dst, fmt=fmt, quality=args.quality, on_progress=progress)
    print(f"\r  [{('█'*40)}] {total_ref[0]}/{total_ref[0]}  done")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="diffy",
        description="diffy-bridge — inspect and export .dfy files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="show metadata for a .dfy file")
    p_info.add_argument("file", help=".dfy file path")

    # export
    p_export = sub.add_parser("export", help="export a .dfy file to video or frames")
    p_export.add_argument("file", help=".dfy file path")
    p_export.add_argument("output", help="output path (.mp4 or directory)")
    p_export.add_argument("--fmt", choices=["mp4", "jpeg", "png"], default=None)
    p_export.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
