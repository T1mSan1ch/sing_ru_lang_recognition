#!/usr/bin/env python3
"""Build raw Kinetics manifest from class-subfolder layout.

Expected dataset layout (e.g. duckdai/kinetics400-mini):
  <root>/
    train/<class_name>/<video>.mp4
    val/<class_name>/<video>.mp4
    test/<class_name>/<video>.mp4

Output CSV columns: video,label,split
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--video-exts", nargs="+", default=[".mp4", ".avi", ".mkv", ".webm"])
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    for split in args.splits:
        split_dir = args.root / split
        if not split_dir.exists():
            print(f"  skip missing split: {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for video in sorted(class_dir.iterdir()):
                if video.suffix.lower() in args.video_exts:
                    rows.append({"video": str(video), "label": label, "split": split})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "split"])
        w.writeheader()
        w.writerows(rows)

    classes = len({r["label"] for r in rows})
    print(f"Kinetics: {len(rows)} rows across {classes} classes")


if __name__ == "__main__":
    main()
