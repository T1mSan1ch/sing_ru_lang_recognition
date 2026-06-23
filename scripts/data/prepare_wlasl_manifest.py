#!/usr/bin/env python3
"""Build a raw manifest CSV for the WLASL-processed Kaggle dataset.

Expected dataset layout (e.g. risangbaskoro/wlasl-processed):
  <root>/
    WLASL_v0.3.json
    videos/00001.mp4 ...

Output CSV columns: video,label,split,user_id,begin,end
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", required=True, type=Path, help="WLASL root containing WLASL_v0.3.json and videos/"
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--video-ext", default=".mp4")
    p.add_argument(
        "--nslt-file",
        type=Path,
        help="Optional NSLT JSON (e.g. nslt_100.json) to filter "
        "WLASL_v0.3.json to a pre-built class subset. The "
        "split is also taken from NSLT when present.",
    )
    args = p.parse_args()

    ann_path = args.root / "WLASL_v0.3.json"
    if not ann_path.exists():
        raise SystemExit(f"Annotation file not found: {ann_path}")
    videos_dir = args.root / "videos"
    if not videos_dir.exists():
        raise SystemExit(f"Videos dir not found: {videos_dir}")

    ann = json.loads(ann_path.read_text(encoding="utf-8"))

    nslt_filter: set[str] | None = None
    nslt_splits: dict[str, str] = {}
    if args.nslt_file:
        if not args.nslt_file.exists():
            raise SystemExit(f"NSLT file not found: {args.nslt_file}")
        nslt = json.loads(args.nslt_file.read_text(encoding="utf-8"))
        nslt_filter = set(nslt.keys())
        for vid, data in nslt.items():
            if isinstance(data, dict) and "subset" in data:
                nslt_splits[vid] = str(data["subset"])

    rows: list[dict[str, str]] = []
    missing = 0
    skipped_filter = 0
    for entry in ann:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            vid = inst.get("video_id")
            if not vid:
                continue
            if nslt_filter is not None and vid not in nslt_filter:
                skipped_filter += 1
                continue
            video_path = videos_dir / f"{vid}{args.video_ext}"
            if not video_path.exists():
                missing += 1
                continue
            split = nslt_splits.get(vid, str(inst.get("split", "")))
            rows.append(
                {
                    "video": str(video_path),
                    "label": gloss,
                    "split": split,
                    "user_id": str(inst.get("signer_id", "")),
                    "begin": str(inst.get("frame_start", "")),
                    "end": str(inst.get("frame_end", "")),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "split", "user_id", "begin", "end"])
        w.writeheader()
        w.writerows(rows)

    classes = len({r["label"] for r in rows})
    msg = f"WLASL: {len(rows)} rows, {classes} glosses, {missing} missing video files"
    if nslt_filter is not None:
        msg += f", filtered by NSLT to {len(nslt_filter)} video_ids ({skipped_filter} excluded)"
    print(msg)


if __name__ == "__main__":
    main()
