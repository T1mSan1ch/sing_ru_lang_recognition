#!/usr/bin/env python3
"""Build AUTSL raw manifest from a Kaggle dataset with split labels CSV files.

Expected dataset layout (e.g. sttaseen/autsl):
  <root>/
    train/  train_labels.csv
    val/    val_labels.csv
    test/   test_labels.csv

Each <split>_labels.csv typically has columns like:
  signer1_sample001, 12

Video files in the Kaggle mirror are usually named with a `_color` suffix:
  train/signer1_sample001_color.mp4

The first underscore-separated token of the filename is treated as user_id.

Output CSV columns: video,label,split,user_id
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def find_video(videos_dir: Path, name: str, exts: list[str]) -> Path | None:
    """Try AUTSL filename variants; allow `name` itself to already include an extension."""
    stems = [name]
    if not name.endswith("_color"):
        stems.append(f"{name}_color")
    for stem in stems:
        direct = videos_dir / stem
        if direct.exists():
            return direct
        for ext in exts:
            cand = videos_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--video-exts", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"])
    args = p.parse_args()

    rows: list[dict[str, str]] = []
    missing = 0
    for split in ["train", "val", "test"]:
        labels_csv = args.root / f"{split}_labels.csv"
        videos_dir = args.root / split
        if not labels_csv.exists() or not videos_dir.exists():
            print(f"  skip {split}: missing {labels_csv} or {videos_dir}")
            continue

        with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, None)
            if first is None:
                continue
            # Detect whether first row is a header (non-numeric label column)
            looks_like_header = False
            if len(first) >= 2:
                try:
                    int(first[1])
                except ValueError:
                    looks_like_header = True
            data_rows = []
            if not looks_like_header:
                data_rows.append(first)
            data_rows.extend(reader)

            for r in data_rows:
                if len(r) < 2:
                    continue
                name, label = r[0].strip(), r[1].strip()
                video_path = find_video(videos_dir, name, args.video_exts)
                if video_path is None:
                    missing += 1
                    continue
                user_id = name.split("_")[0] if "_" in name else ""
                rows.append(
                    {
                        "video": str(video_path),
                        "label": label,
                        "split": split,
                        "user_id": user_id,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "split", "user_id"])
        w.writeheader()
        w.writerows(rows)

    classes = len({r["label"] for r in rows})
    print(f"AUTSL: {len(rows)} rows, {classes} classes, {missing} missing videos")


if __name__ == "__main__":
    main()
