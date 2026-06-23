"""Smoke tests that verify the library imports and basic flows work.

Run: `pytest tests/` or `python -m pytest tests/test_smoke.py -v`
These do NOT touch GPU or download model weights.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def test_imports():
    """All key modules import without error."""
    # Models module imports cleanly even if HF/torchvision are not loaded
    from islr import (  # noqa: F401  # noqa: F401
        data,
        losses,
        metrics,
        models,
        signvlm,
        training,
        transforms,
        utils,
    )


def test_metrics_topk():
    from islr.metrics import MetricAccumulator, macro_f1, topk_accuracy

    logits = torch.tensor([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]])
    target = torch.tensor([1, 0])
    assert abs(topk_accuracy(logits, target, 1) - 1.0) < 1e-6
    assert abs(macro_f1([1, 0], [1, 0], 3) - 2 / 3) < 1e-3
    acc = MetricAccumulator(num_classes=3)
    acc.update(logits, target)
    out = acc.compute()
    assert out["top1"] == 1.0


def test_losses():
    from islr.losses import BoundaryRegressionLoss, IouBalancedSoftCrossEntropy

    logits = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 2, 3])
    span = torch.tensor([[0.0, 1.0], [0.0, 0.5], [0.2, 0.8], [0.5, 0.5]])
    loss = IouBalancedSoftCrossEntropy(label_smoothing=0.1)(logits, labels, span)
    assert loss.ndim == 0 and not torch.isnan(loss)
    bounds_pred = torch.rand(4, 2)
    bloss = BoundaryRegressionLoss()(bounds_pred, span)
    assert bloss.ndim == 0 and not torch.isnan(bloss)


def test_transforms_pipeline():
    from islr.transforms import build_eval_pipeline, build_train_pipeline

    pipeline = build_train_pipeline(clip_len=8, frame_interval=1, image_size=64)
    video = torch.rand(20, 3, 100, 100)
    meta = {"span": (0.2, 0.8), "total_frames": 20, "clip_len": 8, "frame_indices": None}
    out, meta_out = pipeline(video, meta)
    assert out.shape == (8, 3, 64, 64)
    assert meta_out["frame_indices"] is not None
    assert 0 <= meta_out["span"][0] <= 1 and 0 <= meta_out["span"][1] <= 1

    eval_pipeline = build_eval_pipeline(clip_len=8, frame_interval=1, image_size=64)
    out2, _ = eval_pipeline(video, meta)
    assert out2.shape == (8, 3, 64, 64)


def test_mixup_cutmix():
    from islr.transforms import mixup_cutmix

    pixels = torch.rand(4, 8, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mixed_pixels, soft_labels = mixup_cutmix(
        pixels, labels, num_classes=10, mixup_alpha=0.8, cutmix_alpha=1.0
    )
    assert mixed_pixels.shape == pixels.shape
    assert soft_labels.shape == (4, 10)
    # Soft labels should sum to ~1 per row
    assert torch.allclose(soft_labels.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_dataset_csv_roundtrip(tmp_path):
    from islr.data import VideoCsvDataset

    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "split", "begin", "end"])
        writer.writeheader()
        # We don't read the actual videos in this test; only metadata access
    label_to_id = {"hello": 0}
    ds = VideoCsvDataset(csv_path, label_to_id, clip_len=8, transform=None)
    assert len(ds) == 0


def test_label_map_loader(tmp_path):
    from islr.utils import load_label_map

    p = tmp_path / "labels.csv"
    p.write_text("class_id,label\n0,привет\n1,спасибо\n", encoding="utf-8")
    m = load_label_map(p)
    assert m == {"привет": 0, "спасибо": 1}


def test_existing_split_fallback_creates_test():
    from scripts.data.build_sampled_manifest import split_existing

    rows = [{"video": f"v{i}.mp4", "label": "x", "split": "train"} for i in range(20)]
    splits = split_existing(rows)
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_layer_wise_lr_groups():
    import torch.nn as nn

    from islr.models import VideoClassifier, layer_wise_lr_groups

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 16
            self.layer = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
            for i, layer in enumerate(self.layer):
                # Rename so block_re matches "layer.<i>"
                self.add_module(f"_l{i}", layer)

        def forward(self, x):
            for layer in self.layer:
                x = layer(x)
            return x

    backbone = Tiny()
    model = VideoClassifier(backbone, num_classes=5, boundary_head=True)
    groups = layer_wise_lr_groups(model, base_lr=1e-3, decay=0.5, num_layers=3, weight_decay=0.05)
    assert len(groups) > 0
    assert all("lr" in g and "params" in g for g in groups)
