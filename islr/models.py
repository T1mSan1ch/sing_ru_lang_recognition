"""Backbone wrappers + optional boundary-regression head.

Two backbones are supported (both pure PyTorch / no OpenMIM):

  * `videomae`  - HuggingFace `MCG-NJU/videomae-base-finetuned-kinetics`
                  (ViT-Base pretrained on Kinetics-400)
  * `mvit_v2_s` - torchvision `mvit_v2_s` with `MViT_V2_S_Weights.KINETICS400_V1`

Each `build_model(...)` returns a `VideoClassifier` exposing:
    forward(pixels) -> dict(logits=(B,C), bounds=(B,2) or None)
"""

from __future__ import annotations

import torch
import torch.nn as nn

VIDEOMAE_DEFAULT_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"


class BoundaryHead(nn.Module):
    """Predicts (begin, end) of the gesture in the sampled clip, normalised to [0, 1]."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# VideoMAE backbone
# ----------------------------------------------------------------------------


class VideoMAEBackbone(nn.Module):
    """Wraps HuggingFace VideoMAE, exposes pooled CLS features."""

    def __init__(self, ckpt: str = VIDEOMAE_DEFAULT_CKPT, pretrained: bool = True) -> None:
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        if pretrained:
            self.model = VideoMAEModel.from_pretrained(ckpt)
        else:
            cfg = VideoMAEConfig.from_pretrained(ckpt)
            self.model = VideoMAEModel(cfg)
        self.feature_dim = self.model.config.hidden_size  # 768

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, T, C, H, W) — VideoMAE expects (B, T, C, H, W)
        out = self.model(pixel_values=pixels)
        # last_hidden_state: (B, num_tokens, dim); pool over tokens
        return out.last_hidden_state.mean(dim=1)


# ----------------------------------------------------------------------------
# MViTv2 backbone
# ----------------------------------------------------------------------------


class MViTBackbone(nn.Module):
    """torchvision MViTv2-S, returns pooled features (no head)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        net = mvit_v2_s(weights=weights)
        # Strip the final classification head; we replace it
        self.feature_dim = net.head[-1].in_features
        net.head = nn.Identity()
        self.model = net

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # torchvision MViT expects (B, C, T, H, W)
        x = pixels.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)


# ----------------------------------------------------------------------------
# Composite classifier
# ----------------------------------------------------------------------------


class VideoClassifier(nn.Module):
    """Backbone + linear classifier + optional boundary head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        boundary_head: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.feature_dim, num_classes)
        self.boundary_head = BoundaryHead(backbone.feature_dim) if boundary_head else None

    def forward(self, pixels: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.backbone(pixels)
        features = self.dropout(features)
        logits = self.classifier(features)
        bounds = self.boundary_head(features) if self.boundary_head is not None else None
        return {"logits": logits, "bounds": bounds, "features": features}


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    boundary_head: bool = True,
    videomae_ckpt: str = VIDEOMAE_DEFAULT_CKPT,
) -> VideoClassifier:
    if name == "videomae":
        backbone = VideoMAEBackbone(ckpt=videomae_ckpt, pretrained=pretrained)
    elif name == "mvit_v2_s":
        backbone = MViTBackbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {name}. Use 'videomae' or 'mvit_v2_s'.")
    return VideoClassifier(backbone, num_classes=num_classes, boundary_head=boundary_head)


# ----------------------------------------------------------------------------
# Layer-wise LR decay parameter groups (for AdamW)
# ----------------------------------------------------------------------------


def layer_wise_lr_groups(
    model: VideoClassifier,
    base_lr: float,
    decay: float = 0.75,
    num_layers: int = 12,
    weight_decay: float = 0.05,
) -> list[dict]:
    """Returns parameter groups with per-layer LR multipliers.

    Heuristic: any name containing 'layer.<i>' (VideoMAE) or 'blocks.<i>'
    (MViT) is assigned to layer i. Heads/classifier/boundary_head get the
    largest LR (multiplier 1.0).
    """
    import re

    block_re = re.compile(r"\.(?:layer|blocks)\.(\d+)\.")
    groups: dict[int, list[torch.Tensor]] = {}
    no_decay: dict[int, list[torch.Tensor]] = {}

    def assign(name: str) -> int:
        m = block_re.search(name)
        if m:
            return int(m.group(1))
        if name.startswith("classifier") or name.startswith("boundary_head"):
            return num_layers  # head layer
        return 0  # embeddings, patch_embed, etc.

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer = assign(name)
        # Don't decay biases / norms
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.setdefault(layer, []).append(param)
        else:
            groups.setdefault(layer, []).append(param)

    out = []
    for layer, params in groups.items():
        lr = base_lr * (decay ** (num_layers - layer))
        out.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    for layer, params in no_decay.items():
        lr = base_lr * (decay ** (num_layers - layer))
        out.append({"params": params, "lr": lr, "weight_decay": 0.0})
    return out
