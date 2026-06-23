"""Classification + boundary regression losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropy(nn.Module):
    """Cross-entropy that accepts soft labels (B, C). Optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If target is integer, convert to one-hot
        if target.dtype in (torch.long, torch.int):
            target = F.one_hot(target, num_classes=logits.shape[1]).float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / target.shape[1]
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()


class IouBalancedSoftCrossEntropy(nn.Module):
    """Cross-entropy weighted by gesture-window IoU.

    For each sample we have a normalised `span` in [0, 1] of the sampled clip
    (set by `SampleFrames`). The loss multiplies the per-sample CE by the
    `span_length` (i.e. fraction of the clip occupied by the gesture), so
    samples where most of the gesture missed the window count less.
    """

    def __init__(self, label_smoothing: float = 0.0, eps: float = 0.05) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.eps = eps  # floor weight to avoid zeroing samples completely

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, span: torch.Tensor
    ) -> torch.Tensor:
        if target.dtype in (torch.long, torch.int):
            target = F.one_hot(target, num_classes=logits.shape[1]).float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / target.shape[1]
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample = -(target * log_probs).sum(dim=-1)
        # span: (B, 2) with (begin, end) normalised to clip
        gesture_in_clip = (span[:, 1] - span[:, 0]).clamp(min=0.0, max=1.0)
        weight = gesture_in_clip.clamp(min=self.eps)
        return (per_sample * weight).mean()


class BoundaryRegressionLoss(nn.Module):
    """Huber loss on (begin, end) of the gesture inside the sampled clip."""

    def __init__(self, delta: float = 0.1, weight: float = 5.0) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.weight = weight

    def forward(self, pred: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
        return self.weight * self.huber(pred, span)


def build_classification_loss(
    iou_balanced: bool = True,
    label_smoothing: float = 0.1,
) -> nn.Module:
    if iou_balanced:
        return IouBalancedSoftCrossEntropy(label_smoothing=label_smoothing)
    return SoftCrossEntropy(label_smoothing=label_smoothing)
