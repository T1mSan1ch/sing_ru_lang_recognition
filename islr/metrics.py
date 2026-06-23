"""Top-k accuracy and macro-F1 — pure PyTorch / sklearn-free implementations."""

from __future__ import annotations

import torch


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Top-k accuracy. Returns scalar in [0, 1]."""
    k = min(k, logits.shape[1])
    pred = logits.topk(k, dim=1).indices
    return pred.eq(target.view(-1, 1)).any(dim=1).float().mean().item()


@torch.no_grad()
def macro_f1(
    preds: torch.Tensor | list[int], targets: torch.Tensor | list[int], num_classes: int
) -> float:
    """Macro-averaged F1 over `num_classes`. Classes with no support score 0."""
    if isinstance(preds, torch.Tensor):
        preds = preds.tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    f1s: list[float] = []
    for c in range(num_classes):
        tp = sum(1 for p, t in zip(preds, targets) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, targets) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, targets) if p != c and t == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1s.append(2 * precision * recall / denom if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


class MetricAccumulator:
    """Accumulates predictions across batches to compute final top-1, top-5, macro-F1."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.preds: list[int] = []
        self.targets: list[int] = []
        self.top1_sum = 0.0
        self.top5_sum = 0.0
        self.n = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        bsz = target.shape[0]
        self.top1_sum += topk_accuracy(logits, target, 1) * bsz
        self.top5_sum += topk_accuracy(logits, target, 5) * bsz
        self.n += bsz
        self.preds.extend(logits.argmax(dim=1).cpu().tolist())
        self.targets.extend(target.cpu().tolist())

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {"top1": 0.0, "top5": 0.0, "macro_f1": 0.0}
        return {
            "top1": self.top1_sum / self.n,
            "top5": self.top5_sum / self.n,
            "macro_f1": macro_f1(self.preds, self.targets, self.num_classes),
        }
