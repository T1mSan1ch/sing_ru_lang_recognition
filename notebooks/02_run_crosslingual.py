# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # 02 — WLASL transfer variants (C0–C3)
#
# Каждая цепочка = два последовательных вызова `train.py`:
#   шаг 1 — обучение на промежуточном жестовом домене WLASL
#   шаг 2 — fine-tune на Slovo, начиная с лучшего checkpoint из шага 1
# C0 уже посчитан в notebook 01 как A00_full_pipeline.

# %%
import subprocess
from pathlib import Path

WORK = "/kaggle/working"
PROJECT = f"{WORK}/diploma_claude"
MANIFESTS = f"{WORK}/manifests"
WORK_DIRS = f"{WORK}/work_dirs"
RESULTS = f"{WORK}/results"
MASTER_CSV = f"{RESULTS}/all_results.csv"

SLOVO = f"{MANIFESTS}/slovo100"


def run_step(
    exp_id: str,
    dataset_dir: str,
    dataset_name: str,
    *extra,
    model: str = "videomae",
    num_layers: str = "12",
) -> int:
    cmd = [
        "python",
        f"{PROJECT}/scripts/train.py",
        "--train-csv",
        f"{dataset_dir}/{dataset_name}_train.csv",
        "--val-csv",
        f"{dataset_dir}/{dataset_name}_val.csv",
        "--test-csv",
        f"{dataset_dir}/{dataset_name}_test.csv",
        "--label-map",
        f"{dataset_dir}/{dataset_name}_label_map.csv",
        "--output-dir",
        WORK_DIRS,
        "--experiment-id",
        exp_id,
        "--append-results-to",
        MASTER_CSV,
        "--skip-if-done",
        "--model",
        model,
        "--pretrained",
        "k400",
        "--num-layers",
        num_layers,
        *extra,
    ]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def run_chain(
    chain_id: str,
    step1_dir: str,
    step1_name: str,
    *,
    model: str = "videomae",
    num_layers: str = "12",
    step1_epochs: int = 15,
    step2_epochs: int = 30,
) -> None:
    step1_id = f"{chain_id}_step1"
    step2_id = chain_id
    run_step(
        step1_id,
        step1_dir,
        step1_name,
        "--epochs",
        str(step1_epochs),
        model=model,
        num_layers=num_layers,
    )
    step1_best = Path(f"{WORK_DIRS}/{step1_id}/best.pt")
    if not step1_best.exists():
        print(f"  step1 best.pt missing for {step1_id} — cannot run step2")
        return
    run_step(
        step2_id,
        SLOVO,
        "slovo100",
        "--load-from",
        str(step1_best),
        "--epochs",
        str(step2_epochs),
        model=model,
        num_layers=num_layers,
    )


# %% [markdown]
# ## C1 — WLASL → Slovo

# %%
run_chain("C1_wlasl_then_slovo", f"{MANIFESTS}/wlasl100", "wlasl100")

# %% [markdown]
# ## C2 — WLASL longer pretraining → Slovo

# %%
run_chain(
    "C2_wlasl_long_then_slovo",
    f"{MANIFESTS}/wlasl100",
    "wlasl100",
    step1_epochs=30,
    step2_epochs=30,
)

# %% [markdown]
# ## C3 — WLASL MViTv2-S → Slovo

# %%
run_chain(
    "C3_wlasl_mvit_then_slovo",
    f"{MANIFESTS}/wlasl100",
    "wlasl100",
    model="mvit_v2_s",
    num_layers="16",
    step1_epochs=15,
    step2_epochs=30,
)

# %% [markdown]
# ## Snapshot

# %%
import pandas as pd

if Path(MASTER_CSV).exists():
    df = pd.read_csv(MASTER_CSV)
    df["top1"] = pd.to_numeric(df["top1"], errors="coerce")
    df[df.experiment_id.astype(str).str.startswith("C")].sort_values("top1", ascending=False)
