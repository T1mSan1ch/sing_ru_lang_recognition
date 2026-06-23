#!/usr/bin/env python3
"""Build a raw manifest CSV for the Slovo (RSL) Kaggle dataset.

Expected dataset layout (root from the Kaggle Slovo dataset):
  <root>/
    annotations.csv          (whitespace-separated; columns:
                              attachment_id text user_id height width length
                              train begin end)
    slovo/
      train/<attachment_id>.mp4 ...
      test/<attachment_id>.mp4 ...

The root-level annotations.csv is preferred because it contains gesture
boundaries (begin/end), needed for IoU-balanced loss and boundary regression.

Output CSV columns: video,label,split,user_id,begin,end,length
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"attachment_id", "text", "user_id", "train"}


def parse_train_flag(value) -> str:
    """Map the boolean `train` column to a split name."""
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "t"}:
        return "train"
    if v in {"false", "0", "no", "f"}:
        return "test"
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Slovo dataset root containing annotations.csv and slovo/ subdir.",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--video-ext", default=".mp4")
    p.add_argument(
        "--annotations",
        type=Path,
        help="Optional explicit path to annotations.csv (defaults to <root>/annotations.csv).",
    )
    p.add_argument(
        "--sep",
        default=r"\t",
        help="Field separator regex (default: any whitespace). "
        "Use ',' if the file is comma-separated.",
    )
    args = p.parse_args()

    ann_path = args.annotations or (args.root / "annotations.csv")
    if not ann_path.exists():
        raise SystemExit(f"Annotation file not found: {ann_path}")

    train_dir = args.root / "slovo" / "train"
    test_dir = args.root / "slovo" / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit(f"Expected videos in {train_dir} and {test_dir}")

    df = pd.read_csv(ann_path, sep=args.sep, engine="python", encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise SystemExit(
            f"annotations.csv missing required columns: {sorted(missing_cols)}. "
            f"Found: {list(df.columns)}"
        )

    rows: list[dict[str, str]] = []
    missing_videos = 0
    for _, r in df.iterrows():
        split = parse_train_flag(r["train"])
        if not split:
            continue
        attach = str(r["attachment_id"]).strip()
        video_dir = train_dir if split == "train" else test_dir
        video_path = video_dir / f"{attach}{args.video_ext}"
        if not video_path.exists():
            missing_videos += 1
            continue
        rows.append(
            {
                "video": str(video_path),
                "label": str(r["text"]).strip(),
                "split": split,
                "user_id": str(r["user_id"]).strip(),
                "begin": "" if pd.isna(r.get("begin")) else str(r["begin"]).strip(),
                "end": "" if pd.isna(r.get("end")) else str(r["end"]).strip(),
                "length": "" if pd.isna(r.get("length")) else str(r["length"]).strip(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["video", "label", "split", "user_id", "begin", "end", "length"]
        )
        w.writeheader()
        w.writerows(rows)

    classes = len({r["label"] for r in rows})
    train_n = sum(1 for r in rows if r["split"] == "train")
    test_n = sum(1 for r in rows if r["split"] == "test")
    print(
        f"Slovo: {len(rows)} rows ({train_n} train, {test_n} test), "
        f"{classes} classes, {missing_videos} missing video files"
    )


if __name__ == "__main__":
    main()
