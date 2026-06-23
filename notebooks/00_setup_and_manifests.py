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
# # 00 — Setup & manifests (Pure PyTorch version)
#
# Устанавливает зависимости, копирует код проекта в `/kaggle/working/`, строит
# CSV-манифесты для Slovo/WLASL/Kinetics и few-shot splits для F-блока.
#
# Никакого MMAction2 / MMCV / MMEngine — только PyTorch + torchvision +
# transformers + open_clip.

# %% [markdown]
# ## 1. Install dependencies (Python 3.12 compatible)

# %%
# !pip install -q --upgrade pip setuptools wheel
# # P100 needs sm_60 support; current torch 2.10/cu128 Kaggle wheels do not include it.
# !pip install -q "torch==2.6.0" "torchvision==0.21.0" \
#                 --index-url https://download.pytorch.org/whl/cu126
# !pip install -q "transformers>=4.51,<5.0" "accelerate>=0.34" \
#                 "open_clip_torch>=2.26" "av>=12.0" \
#                 "pandas>=2.1" "scikit-learn>=1.4" "pyyaml>=6.0" "tqdm>=4.66"

# %%
import torch
import torchvision
import transformers

print(
    "torch:",
    torch.__version__,
    "| cuda:",
    torch.cuda.is_available(),
    "| cuda ver:",
    torch.version.cuda,
)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| arch list:", torch.cuda.get_arch_list())

# %% [markdown]
# ## 2. Copy research code into /kaggle/working/

# %%
import shutil
from pathlib import Path

SRC_CANDIDATES = [
    "/kaggle/input/datasets/<your_username>/diploma-claude/Diploma_claude",
    "/kaggle/input/datasets/<your_username>/diploma-claude",
    "/kaggle/input/diploma-claude/Diploma_claude",
    "/kaggle/input/diploma-claude",
]
DST = Path("/kaggle/working/diploma_claude")
src = next((Path(p) for p in SRC_CANDIDATES if (Path(p) / "scripts").exists()), None)
if src is None:
    raise FileNotFoundError(f"Code dataset not found in any of: {SRC_CANDIDATES}")
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(src, DST)
print(f"Copied: {src} -> {DST}")
print(sorted(p.name for p in DST.iterdir()))

# %% [markdown]
# ## 3. Paths

# %%
import os

WORK = "/kaggle/working"
PROJECT = f"{WORK}/diploma_claude"
MANIFESTS = f"{WORK}/manifests"
WORK_DIRS = f"{WORK}/work_dirs"
RESULTS = f"{WORK}/results"
CKPTS = f"{WORK}/checkpoints"
MASTER_CSV = f"{RESULTS}/all_results.csv"
for d in [MANIFESTS, WORK_DIRS, RESULTS, CKPTS]:
    os.makedirs(d, exist_ok=True)

# Adjust to actual Kaggle Input mount paths
SLOVO_ROOT = "/kaggle/input/datasets/<author>/slovo-russian-sign-language-dataset"
WLASL_ROOT = "/kaggle/input/datasets/risangbaskoro/wlasl-processed"
K400_ROOT = "/kaggle/input/datasets/duckdai/kinetics400-mini/kinetics400_mini"

os.environ.update(
    {
        "PROJECT": PROJECT,
        "MANIFESTS": MANIFESTS,
        "WORK_DIRS": WORK_DIRS,
        "RESULTS": RESULTS,
        "MASTER_CSV": MASTER_CSV,
        "SLOVO_ROOT": SLOVO_ROOT,
        "WLASL_ROOT": WLASL_ROOT,
        "K400_ROOT": K400_ROOT,
    }
)

# %% [markdown]
# ## 4. Build raw manifests (3 required dataset adapters)

# %%
# !python $PROJECT/scripts/data/prepare_slovo_manifest.py \
#   --root $SLOVO_ROOT --output $MANIFESTS/slovo_raw.csv

# %%
# !python $PROJECT/scripts/data/prepare_wlasl_manifest.py \
#   --root $WLASL_ROOT --output $MANIFESTS/wlasl_raw.csv

# %%
# !python $PROJECT/scripts/data/prepare_kinetics_folder_manifest.py \
#   --root $K400_ROOT --output $MANIFESTS/kinetics_raw.csv

# %% [markdown]
# ## 5. Sample 100-class manifests

# %%
# Slovo: signer-independent split
# !python $PROJECT/scripts/data/build_sampled_manifest.py \
#   --input $MANIFESTS/slovo_raw.csv \
#   --output-dir $MANIFESTS/slovo100 --dataset-name slovo100 \
#   --num-classes 100 --samples-per-class 0 \
#   --split-mode signer_independent --val-signers 1 --test-signers 1 --seed 42

# WLASL: existing split
# !python $PROJECT/scripts/data/build_sampled_manifest.py \
#   --input $MANIFESTS/wlasl_raw.csv \
#   --output-dir $MANIFESTS/wlasl100 --dataset-name wlasl100 \
#   --num-classes 100 --samples-per-class 0 \
#   --split-mode existing --seed 42

# Kinetics-400 sample
# The Kaggle mini mirror often contains only train/<class> folders, so make a
# class-stratified random train/val/test split from the available videos.
# !python $PROJECT/scripts/data/build_sampled_manifest.py \
#   --input $MANIFESTS/kinetics_raw.csv \
#   --output-dir $MANIFESTS/kinetics400_sample --dataset-name kinetics400_sample \
#   --num-classes 100 --samples-per-class 30 \
#   --split-mode random --seed 42

# %% [markdown]
# ## 6. Few-shot splits for F2 (SignVLM)

# %%
# !python $PROJECT/scripts/data/make_fewshot_splits.py \
#   --train $MANIFESTS/slovo100/slovo100_train.csv \
#   --test $MANIFESTS/slovo100/slovo100_test.csv \
#   --output-dir $MANIFESTS/slovo100_fewshot \
#   --shots 1 2 4 8 --seeds 42 43 44

# %% [markdown]
# ## 7. Sanity check

# %%
import glob

import pandas as pd

print("Manifests built:")
for p in sorted(glob.glob(f"{MANIFESTS}/*/*_train.csv")):
    df = pd.read_csv(p)
    print(f"  {p}: {len(df)} rows, {df['label'].nunique()} classes")


def check_unknown_labels(dataset_name: str, *patterns: str) -> None:
    """Print rows/classes matching unknown/background label patterns."""
    patterns = patterns or ("1001", "не определ", "неопредел", "undefined", "unknown")
    regex = "|".join(patterns)
    raw_path = f"{MANIFESTS}/{dataset_name}_raw.csv"
    label_map_path = f"{MANIFESTS}/{dataset_name}100/{dataset_name}100_label_map.csv"

    print(f"\nUnknown-label check for {dataset_name}: {patterns}")
    if Path(raw_path).exists():
        raw = pd.read_csv(raw_path)
        raw_hits = raw[raw["label"].astype(str).str.contains(regex, case=False, na=False)]
        print(f"raw matches: {len(raw_hits)}")
        if not raw_hits.empty:
            print(raw_hits["label"].value_counts().to_string())
    else:
        print(f"raw manifest not found: {raw_path}")

    if Path(label_map_path).exists():
        label_map = pd.read_csv(label_map_path)
        map_hits = label_map[
            label_map["label"].astype(str).str.contains(regex, case=False, na=False)
        ]
        print(f"label_map matches: {len(map_hits)}")
        if not map_hits.empty:
            print(map_hits.to_string(index=False))
    else:
        print(f"label_map not found: {label_map_path}")


check_unknown_labels("slovo")
