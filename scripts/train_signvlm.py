#!/usr/bin/env python3
"""SignVLM-style classifier training (block F: F0 zero-shot, F1/F2/F3 supervised).

Modes:
  zero_shot  - no training; cosine sim of frame averages with text prompts
  train      - frozen VLM encoder + trainable temporal decoder + classifier
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from islr.data import VideoCsvDataset, collate
from islr.metrics import MetricAccumulator
from islr.signvlm import SignVLM, build_frame_encoder, zero_shot_predict
from islr.transforms import build_eval_pipeline
from islr.utils import (
    device,
    env_info,
    is_done,
    load_label_map,
    save_checkpoint,
    set_seed,
    write_done_flag,
    write_json,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP2_MEAN = (0.5, 0.5, 0.5)
SIGLIP2_STD = (0.5, 0.5, 0.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero_shot", "train"], required=True)
    p.add_argument("--train-csv", type=Path)
    p.add_argument("--test-csv", required=True, type=Path)
    p.add_argument("--label-map", required=True, type=Path)
    p.add_argument("--vlm-backbone", choices=["siglip2", "clip"], default="siglip2")
    p.add_argument("--encoder-model", default="google/siglip2-base-patch16-224")
    p.add_argument(
        "--clip-model",
        default="ViT-B-16",
        help="Deprecated alias for --encoder-model when --vlm-backbone clip",
    )
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--frame-interval", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--decoder-heads", type=int, default=8)
    p.add_argument("--decoder-hidden", type=int, default=512)
    p.add_argument("--prompt", default="жест {label}")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--append-results-to", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-if-done", action="store_true")
    return p.parse_args()


def append_result(
    master_csv: Path, exp_id: str, mode: str, metrics: dict, args: argparse.Namespace
) -> None:
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
    row = {
        "experiment_id": exp_id,
        "model": encoder_label(args) + (" + temporal decoder" if mode == "train" else ""),
        "pretraining": args.clip_pretrained if args.vlm_backbone == "clip" else "hf",
        "augmentations": "none",
        "losses": "CE" if mode == "train" else "none",
        "top1": round(metrics.get("top1", 0.0), 4),
        "top5": round(metrics.get("top5", 0.0), 4),
        "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
        "test_top1": round(metrics.get("test_top1", 0.0), 4),
        "test_top5": round(metrics.get("test_top5", 0.0), 4),
        "test_macro_f1": round(metrics.get("test_macro_f1", 0.0), 4),
        "epoch": metrics.get("epoch", 0),
        "train_time_sec": round(metrics.get("train_time_sec", 0.0), 1),
    }
    existing = []
    if master_csv.exists():
        with master_csv.open("r", encoding="utf-8-sig", newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("experiment_id") != exp_id]
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    print(f"Appended SignVLM row for {exp_id}")


def encoder_model_name(args: argparse.Namespace) -> str:
    return (
        args.clip_model
        if args.vlm_backbone == "clip" and args.encoder_model == "google/siglip2-base-patch16-224"
        else args.encoder_model
    )


def encoder_label(args: argparse.Namespace) -> str:
    if args.vlm_backbone == "clip":
        return f"CLIP {encoder_model_name(args)}"
    return f"SigLIP2 {encoder_model_name(args)}"


def encoder_norm(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if args.vlm_backbone == "clip":
        return CLIP_MEAN, CLIP_STD
    return SIGLIP2_MEAN, SIGLIP2_STD


def run_zero_shot(args, dev, label_to_id, test_loader) -> dict:
    encoder = build_frame_encoder(
        args.vlm_backbone, encoder_model_name(args), args.clip_pretrained
    ).to(dev)
    labels = sorted(label_to_id, key=lambda k: label_to_id[k])
    text_features = encoder.encode_texts(
        [args.prompt.format(label=label) for label in labels],
        dev,
    )
    acc = MetricAccumulator(len(labels))
    for batch in test_loader:
        pixels = batch["pixels"].to(dev, non_blocking=True)
        labels_t = batch["label"].to(dev, non_blocking=True)
        logits = zero_shot_predict(encoder, pixels, text_features)
        acc.update(logits, labels_t)
    return acc.compute()


def evaluate_signvlm(model: SignVLM, loader, num_classes, dev) -> dict:
    model.eval()
    acc = MetricAccumulator(num_classes)
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            logits = model(pixels)
            acc.update(logits, labels)
    return acc.compute()


def train_supervised(args, dev, label_to_id, train_loader, test_loader) -> dict:
    num_classes = len(label_to_id)
    work_dir = args.output_dir / args.experiment_id
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "train.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    log(f"Logging training to {log_path}")
    model = SignVLM(
        num_classes=num_classes,
        vlm_backbone=args.vlm_backbone,
        encoder_model=encoder_model_name(args),
        clip_pretrained=args.clip_pretrained,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_hidden=args.decoder_hidden,
    ).to(dev)

    trainable = [p for p in model.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = {"top1": -1.0}
    patience_left = args.patience
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            pixels = batch["pixels"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            logits = model(pixels)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        m = evaluate_signvlm(model, test_loader, num_classes, dev)
        avg = sum(losses) / max(1, len(losses))
        log(f"epoch {epoch:03d} | loss {avg:.4f} | test top1 {m['top1']:.4f}")
        if m["top1"] > best["top1"] + 0.003:
            best = {**m, "epoch": epoch}
            patience_left = args.patience
            save_checkpoint(work_dir / "best.pt", model)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(
                    f"Early stop at epoch {epoch} (best top1={best['top1']:.4f} @ epoch {best['epoch']})"
                )
                break
    best["train_time_sec"] = time.time() - started
    best["test_top1"] = best.get("top1", 0.0)
    best["test_top5"] = best.get("top5", 0.0)
    best["test_macro_f1"] = best.get("macro_f1", 0.0)
    return best


def main() -> None:
    args = parse_args()
    work_dir = args.output_dir / args.experiment_id
    if args.skip_if_done and is_done(work_dir, args.experiment_id):
        print(f"SKIP {args.experiment_id}: done flag present")
        return

    set_seed(args.seed)
    dev = device()
    print("Env:", env_info())

    label_to_id = load_label_map(args.label_map)
    mean, std = encoder_norm(args)

    eval_pipeline = build_eval_pipeline(
        clip_len=args.clip_len,
        frame_interval=args.frame_interval,
        image_size=args.image_size,
        mean=mean,
        std=std,
    )
    test_ds = VideoCsvDataset(args.test_csv, label_to_id, args.clip_len, eval_pipeline)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    if args.mode == "zero_shot":
        metrics = run_zero_shot(args, dev, label_to_id, test_loader)
        metrics = {
            **metrics,
            "test_top1": metrics["top1"],
            "test_top5": metrics["top5"],
            "test_macro_f1": metrics["macro_f1"],
            "epoch": 0,
            "train_time_sec": 0.0,
        }
    else:
        if not args.train_csv:
            raise SystemExit("--train-csv is required for --mode train")
        train_pipeline = build_eval_pipeline(  # F-block uses light/no augmentation
            clip_len=args.clip_len,
            frame_interval=args.frame_interval,
            image_size=args.image_size,
            mean=mean,
            std=std,
        )
        train_ds = VideoCsvDataset(args.train_csv, label_to_id, args.clip_len, train_pipeline)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,
        )
        metrics = train_supervised(args, dev, label_to_id, train_loader, test_loader)

    write_json(work_dir / "metrics.json", metrics)
    if args.append_results_to:
        append_result(args.append_results_to, args.experiment_id, args.mode, metrics, args)
    write_done_flag(work_dir, args.experiment_id)
    print(f"DONE {args.experiment_id}: top1={metrics.get('top1', 0):.4f}")


if __name__ == "__main__":
    main()
