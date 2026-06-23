#!/usr/bin/env python3
"""Build balanced sampled manifests for ISLR experiments."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    missing = {"video", "label"} - set(rows[0])
    if missing:
        raise SystemExit(f"Manifest must contain columns: {sorted(missing)}")
    return rows


def choose_classes(
    rows: list[dict[str, str]], num_classes: int, explicit: Path | None
) -> list[str]:
    if explicit:
        classes = [
            line.strip()
            for line in explicit.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return classes[:num_classes]
    counts = Counter(row["label"] for row in rows)
    return [label for label, _ in counts.most_common(num_classes)]


def sample_rows(
    rows: list[dict[str, str]],
    labels: list[str],
    samples_per_class: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["label"] in labels:
            by_label[row["label"]].append(row)

    sampled: list[dict[str, str]] = []
    for label in labels:
        label_rows = by_label[label]
        if not label_rows:
            continue
        rng.shuffle(label_rows)
        sampled.extend(label_rows if samples_per_class <= 0 else label_rows[:samples_per_class])
    return sampled


def split_existing(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    if "split" not in rows[0]:
        raise SystemExit("--split-mode existing requires a split column")
    splits = {"train": [], "val": [], "test": []}
    aliases = {"valid": "val", "validation": "val", "dev": "val"}
    for row in rows:
        split = aliases.get(row.get("split", "").lower(), row.get("split", "").lower())
        if split in splits:
            splits[split].append(row)
    if not splits["test"]:
        train = splits["train"]
        n_test = max(1, int(len(train) * 0.1))
        splits["test"] = train[:n_test]
        splits["train"] = train[n_test:]
    if not splits["val"]:
        train = splits["train"]
        n_val = max(1, int(len(train) * 0.1))
        splits["val"] = train[:n_val]
        splits["train"] = train[n_val:]
    return splits


def split_random(rows: list[dict[str, str]], rng: random.Random) -> dict[str, list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        splits["test"].extend(label_rows[:n_test])
        splits["val"].extend(label_rows[n_test : n_test + n_val])
        splits["train"].extend(label_rows[n_test + n_val :])
    return splits


def split_signer_independent(
    rows: list[dict[str, str]], rng: random.Random, val_signers: int, test_signers: int
) -> dict[str, list[dict[str, str]]]:
    if "user_id" not in rows[0]:
        raise SystemExit("--split-mode signer_independent requires a user_id column")
    signers = sorted({row["user_id"] for row in rows if row.get("user_id")})
    rng.shuffle(signers)
    if len(signers) < val_signers + test_signers + 1:
        raise SystemExit("Not enough signers for signer-independent split")
    val_set = set(signers[:val_signers])
    test_set = set(signers[val_signers : val_signers + test_signers])
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        signer = row.get("user_id")
        if signer in val_set:
            splits["val"].append(row)
        elif signer in test_set:
            splits["test"].append(row)
        else:
            splits["train"].append(row)
    return splits


def write_split(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_label_map(path: Path, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "label"])
        for idx, label in enumerate(labels):
            writer.writerow([idx, label])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument(
        "--samples-per-class", type=int, default=0, help="0 means keep all selected samples"
    )
    parser.add_argument("--class-list", type=Path)
    parser.add_argument(
        "--split-mode", choices=["existing", "random", "signer_independent"], default="existing"
    )
    parser.add_argument("--val-signers", type=int, default=1)
    parser.add_argument("--test-signers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_rows(args.input)
    labels = choose_classes(rows, args.num_classes, args.class_list)
    selected = sample_rows(rows, labels, args.samples_per_class, rng)
    fieldnames = list(rows[0].keys())

    if args.split_mode == "existing":
        splits = split_existing(selected)
    elif args.split_mode == "signer_independent":
        splits = split_signer_independent(selected, rng, args.val_signers, args.test_signers)
    else:
        splits = split_random(selected, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_split(args.output_dir / f"{args.dataset_name}_{split}.csv", split_rows, fieldnames)
    write_label_map(args.output_dir / f"{args.dataset_name}_label_map.csv", labels)

    for split, split_rows in splits.items():
        print(f"{split}: {len(split_rows)}")
    print(f"classes: {len(labels)}")


if __name__ == "__main__":
    main()
