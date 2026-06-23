#!/usr/bin/env python3
"""Create K-shot train splits while preserving the same test split."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "label" not in fieldnames:
        raise SystemExit(f"{path} must contain label column")
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    train_rows, fieldnames = read_csv(args.train)
    test_rows, test_fieldnames = read_csv(args.test)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        by_label[row["label"]].append(row)

    for seed in args.seeds:
        rng = random.Random(seed)
        for shot in args.shots:
            fewshot_rows: list[dict[str, str]] = []
            for rows in by_label.values():
                rows = rows[:]
                rng.shuffle(rows)
                fewshot_rows.extend(rows[: min(shot, len(rows))])
            prefix = f"shot{shot}_seed{seed}"
            write_csv(args.output_dir / f"{prefix}_train.csv", fewshot_rows, fieldnames)
            write_csv(args.output_dir / f"{prefix}_test.csv", test_rows, test_fieldnames)
            print(f"{prefix}: train={len(fewshot_rows)} test={len(test_rows)}")


if __name__ == "__main__":
    main()
