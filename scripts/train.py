#!/usr/bin/env python3
"""Main training script for ISLR ablation, K-block, and cross-lingual experiments.

A single CLI handles all variants. Ablation experiments differ by toggle flags:
  A00 (full):         all flags ON
  A01 no image augs:  --no-aug-image
  A03 no video augs:  --no-aug-video
  A09 no boundary:    --no-boundary-head
  A11 no IoU loss:    --no-iou-loss
  A12 plain CE:       --no-iou-loss --mixup 0 --cutmix 0
  K0 no Kinetics:     --pretrained none
  K2b after K2a:      --load-from <K2a_best.pt>
  C1 step1 (WLASL):   --train-csv .../wlasl100_train.csv ...
  C1 step2 (Slovo):   --load-from <C1_step1_best.pt> --train-csv .../slovo100_train.csv

After training, appends a row to `--append-results-to` master CSV and writes a
`<exp_id>.done` flag in the work_dir.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is on the path so `import islr` works from scripts/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from islr.data import VideoCsvDataset, collate
from islr.models import build_model, layer_wise_lr_groups
from islr.training import TrainConfig, train
from islr.transforms import build_eval_pipeline, build_train_pipeline
from islr.utils import (
    device,
    env_info,
    is_done,
    load_checkpoint,
    load_label_map,
    set_seed,
    write_done_flag,
    write_json,
)

VIDEOMAE_MEAN = (0.485, 0.456, 0.406)
VIDEOMAE_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--test-csv", type=Path)
    p.add_argument("--label-map", required=True, type=Path)

    # Frame sampling
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--frame-interval", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)

    # Model
    p.add_argument("--model", choices=["videomae", "mvit_v2_s"], default="videomae")
    p.add_argument("--pretrained", choices=["k400", "none"], default="k400")
    p.add_argument(
        "--load-from",
        type=Path,
        help="Optional: load weights from this .pt before training "
        "(used for cross-lingual / K2 chains)",
    )

    # Augmentation toggles (defaults = full pipeline)
    p.add_argument("--aug-video", dest="aug_video", action="store_true", default=True)
    p.add_argument("--no-aug-video", dest="aug_video", action="store_false")
    p.add_argument("--aug-image", dest="aug_image", action="store_true", default=True)
    p.add_argument("--no-aug-image", dest="aug_image", action="store_false")
    p.add_argument("--boundary-head", dest="boundary_head", action="store_true", default=True)
    p.add_argument("--no-boundary-head", dest="boundary_head", action="store_false")
    p.add_argument("--iou-loss", dest="iou_loss", action="store_true", default=True)
    p.add_argument("--no-iou-loss", dest="iou_loss", action="store_false")
    p.add_argument("--mixup", type=float, default=0.8, help="MixUp alpha (0 = off)")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (0 = off)")
    p.add_argument("--boundary-shift", type=int, default=5)

    # Training
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--layer-wise-lr-decay", type=float, default=0.75)
    p.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="For layer-wise LR decay (12=VideoMAE-base, 16=MViT-V2-S)",
    )
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)

    # Output / experiment id
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--experiment-id", required=True)
    p.add_argument(
        "--append-results-to", type=Path, help="Master CSV to append a row to after training"
    )
    p.add_argument("--seed", type=int, default=42)

    # Resilience
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="If <work_dir>/<experiment_id>.done exists, exit immediately",
    )

    return p.parse_args()


def append_result(master_csv: Path, exp_id: str, args: argparse.Namespace, metrics: dict) -> None:
    master_csv.parent.mkdir(parents=True, exist_ok=True)
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
    aug_parts = []
    if args.aug_video:
        aug_parts.append("video")
    if args.aug_image:
        aug_parts.append("image")
    if args.mixup > 0:
        aug_parts.append(f"mixup({args.mixup})")
    if args.cutmix > 0:
        aug_parts.append(f"cutmix({args.cutmix})")
    losses = []
    losses.append("IoU-balanced CE" if args.iou_loss else "CE")
    if args.boundary_head:
        losses.append("Huber bounds")

    row = {
        "experiment_id": exp_id,
        "model": args.model,
        "pretraining": args.pretrained
        if not args.load_from
        else f"load_from={args.load_from.name}",
        "augmentations": "+".join(aug_parts) or "none",
        "losses": "+".join(losses),
        "top1": round(metrics.get("top1", 0.0), 4),
        "top5": round(metrics.get("top5", 0.0), 4),
        "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
        "test_top1": round(metrics.get("test_top1", 0.0), 4),
        "test_top5": round(metrics.get("test_top5", 0.0), 4),
        "test_macro_f1": round(metrics.get("test_macro_f1", 0.0), 4),
        "epoch": metrics.get("epoch", 0),
        "train_time_sec": round(metrics.get("train_time_sec", 0.0), 1),
    }
    # Upsert by experiment_id
    existing = []
    if master_csv.exists():
        with master_csv.open("r", encoding="utf-8-sig", newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("experiment_id") != exp_id]
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    print(f"Appended result row for {exp_id} -> {master_csv}")


def main() -> None:
    args = parse_args()
    work_dir = args.output_dir / args.experiment_id

    if args.skip_if_done and is_done(work_dir, args.experiment_id):
        print(f"SKIP {args.experiment_id}: done flag present in {work_dir}")
        return

    set_seed(args.seed)
    dev = device()
    print("Env:", env_info())

    label_to_id = load_label_map(args.label_map)
    num_classes = len(label_to_id)
    print(f"num_classes = {num_classes}")

    train_pipeline = build_train_pipeline(
        clip_len=args.clip_len,
        frame_interval=args.frame_interval,
        image_size=args.image_size,
        aug_video=args.aug_video,
        aug_image=args.aug_image,
        boundary_shift=args.boundary_shift,
    )
    eval_pipeline = build_eval_pipeline(
        clip_len=args.clip_len,
        frame_interval=args.frame_interval,
        image_size=args.image_size,
    )

    train_ds = VideoCsvDataset(args.train_csv, label_to_id, args.clip_len, train_pipeline)
    val_ds = VideoCsvDataset(args.val_csv, label_to_id, args.clip_len, eval_pipeline)
    test_ds = (
        VideoCsvDataset(args.test_csv, label_to_id, args.clip_len, eval_pipeline)
        if args.test_csv
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )
        if test_ds
        else None
    )

    print(
        f"Building model: {args.model} (pretrained={args.pretrained}, boundary_head={args.boundary_head})"
    )
    model = build_model(
        name=args.model,
        num_classes=num_classes,
        pretrained=(args.pretrained == "k400"),
        boundary_head=args.boundary_head,
    ).to(dev)

    if args.load_from:
        print(f"Loading weights from {args.load_from}")
        load_checkpoint(args.load_from, model, strict=False)

    param_groups = layer_wise_lr_groups(
        model,
        base_lr=args.lr,
        decay=args.layer_wise_lr_decay,
        num_layers=args.num_layers,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    cfg = TrainConfig(
        epochs=args.epochs,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        iou_loss=args.iou_loss,
        boundary_head=args.boundary_head,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        amp=args.amp,
    )
    metrics = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        cfg,
        num_classes=num_classes,
        device=dev,
        work_dir=work_dir,
        test_loader=test_loader,
    )

    write_json(work_dir / "metrics.json", metrics)
    if args.append_results_to:
        append_result(args.append_results_to, args.experiment_id, args, metrics)
    write_done_flag(work_dir, args.experiment_id)
    print(f"DONE {args.experiment_id}: best top1={metrics['top1']:.4f}")


if __name__ == "__main__":
    main()
