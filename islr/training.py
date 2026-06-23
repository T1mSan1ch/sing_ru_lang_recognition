"""Training/evaluation loops with mixed precision, warmup+cosine LR, early stopping."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from .losses import BoundaryRegressionLoss, build_classification_loss
from .metrics import MetricAccumulator
from .models import VideoClassifier
from .transforms import mixup_cutmix
from .utils import save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 60
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 7
    min_delta: float = 0.003
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    iou_loss: bool = True
    boundary_head: bool = True
    boundary_weight: float = 5.0
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    amp: bool = True
    log_interval: int = 20


def warmup_cosine_lambda(warmup_epochs: int, total_epochs: int, min_ratio: float = 0.001):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(min_ratio, (epoch + 1) / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def evaluate(
    model: VideoClassifier,
    loader: Iterable,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    acc = MetricAccumulator(num_classes)
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            out = model(pixels)
            acc.update(out["logits"], labels)
    return acc.compute()


def train(
    model: VideoClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    num_classes: int,
    device: torch.device,
    work_dir: Path,
    test_loader: DataLoader | None = None,
) -> dict[str, float]:
    """Run a full training loop. Returns best metrics dict."""
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "train.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    log(f"Logging training to {log_path}")
    cls_loss_fn = build_classification_loss(
        iou_balanced=cfg.iou_loss,
        label_smoothing=cfg.label_smoothing,
    ).to(device)
    boundary_loss_fn = (
        BoundaryRegressionLoss(weight=cfg.boundary_weight).to(device) if cfg.boundary_head else None
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine_lambda(cfg.warmup_epochs, cfg.epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best = {"top1": -1.0, "top5": -1.0, "macro_f1": -1.0, "epoch": -1}
    patience_left = cfg.patience
    started = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            spans = batch["span"].to(device, non_blocking=True)

            # MixUp / CutMix
            mixed_pixels, mixed_targets = mixup_cutmix(
                pixels,
                labels,
                num_classes,
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=cfg.cutmix_alpha,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                out = model(mixed_pixels)
                if cfg.iou_loss:
                    loss = cls_loss_fn(out["logits"], mixed_targets, spans)
                else:
                    loss = cls_loss_fn(out["logits"], mixed_targets)
                if boundary_loss_fn is not None and out["bounds"] is not None:
                    loss = loss + boundary_loss_fn(out["bounds"], spans)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach())
            n_batches += 1
            if step % cfg.log_interval == 0:
                log(
                    f"  [epoch {epoch:03d}] step {step:04d} | loss {float(loss):.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        metrics = evaluate(model, val_loader, num_classes, device)
        elapsed = time.time() - started
        log(
            f"epoch {epoch:03d} | loss {avg_loss:.4f} | val top1 {metrics['top1']:.4f} top5 {metrics['top5']:.4f} f1 {metrics['macro_f1']:.4f} | {elapsed:.0f}s"
        )

        if metrics["top1"] > best["top1"] + cfg.min_delta:
            best = {**metrics, "epoch": epoch, "loss": avg_loss}
            patience_left = cfg.patience
            save_checkpoint(
                work_dir / "best.pt", model, optimizer, scheduler, extra={"epoch": epoch, **metrics}
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(
                    f"Early stop at epoch {epoch} (best top1={best['top1']:.4f} @ epoch {best['epoch']})"
                )
                break

    if test_loader is not None:
        # Final evaluation on test set with the best checkpoint
        ckpt = torch.load(work_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader, num_classes, device)
        best = {**best, **{f"test_{k}": v for k, v in test_metrics.items()}}
        log(
            f"TEST | top1 {test_metrics['top1']:.4f} top5 {test_metrics['top5']:.4f} f1 {test_metrics['macro_f1']:.4f}"
        )

    best["train_time_sec"] = time.time() - started
    return best
