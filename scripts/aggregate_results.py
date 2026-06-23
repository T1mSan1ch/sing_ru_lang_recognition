#!/usr/bin/env python3
"""Aggregate per-experiment metrics.json files into a single CSV summary.

Walks `--inputs` looking for `<exp_id>/metrics.json`, then merges with the
master CSV produced incrementally by `train.py --append-results-to`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Roots to scan for metrics.json files (typically work_dirs/)",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--master-csv", type=Path, help="Optional existing master CSV to merge with")
    args = p.parse_args()

    rows: dict[str, dict] = {}

    if args.master_csv and args.master_csv.exists():
        with args.master_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["experiment_id"]] = row

    for root in args.inputs:
        for path in root.rglob("metrics.json"):
            exp_id = path.parent.name
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.setdefault(exp_id, {"experiment_id": exp_id}).update(
                {
                    "top1": round(data.get("top1", 0.0), 4),
                    "top5": round(data.get("top5", 0.0), 4),
                    "macro_f1": round(data.get("macro_f1", 0.0), 4),
                    "test_top1": round(data.get("test_top1", 0.0), 4),
                    "test_top5": round(data.get("test_top5", 0.0), 4),
                    "test_macro_f1": round(data.get("test_macro_f1", 0.0), 4),
                    "epoch": data.get("epoch", 0),
                    "train_time_sec": round(data.get("train_time_sec", 0.0), 1),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "model",
        "pretraining",
        "augmentations",
        "losses",
        "top1",
        "top5",
        "macro_f1",
        "test_top1",
        "test_top5",
        "test_macro_f1",
        "epoch",
        "train_time_sec",
    ]
    sorted_rows = sorted(rows.values(), key=lambda r: -float(r.get("top1") or 0))
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow(r)
    print(f"Wrote {len(sorted_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
