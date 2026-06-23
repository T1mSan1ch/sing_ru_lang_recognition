"""Generic utilities: seed, checkpoint I/O, label-map helpers."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_map(path: Path) -> dict[str, int]:
    """Read CSV with `class_id,label` columns -> {label: class_id}."""
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["label"]] = int(row["class_id"])
    return out


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: Path, model: torch.nn.Module, strict: bool = False) -> dict:
    """Load checkpoint into model. Returns the full payload (for extra info)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"  load_checkpoint: missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"  load_checkpoint: unexpected keys (first 5): {list(unexpected)[:5]}")
    return payload


def write_done_flag(work_dir: Path, exp_id: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / f"{exp_id}.done").write_text("ok\n", encoding="utf-8")


def is_done(work_dir: Path, exp_id: str) -> bool:
    return (work_dir / f"{exp_id}.done").exists()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def env_info() -> dict[str, str]:
    return {
        "python": os.popen("python3 --version").read().strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "device": str(device()),
    }
