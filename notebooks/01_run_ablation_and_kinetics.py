# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — Ablation (A00–A12) + Kinetics pretraining (K0/K2)
#
# Запускает `scripts/train.py` для каждой ablation-конфигурации и для K-блока.
# Все эксперименты используют один CLI: `train.py` принимает toggle-флаги
# `--no-aug-image`, `--no-iou-loss`, и т.п.
#
# **Resilience:** после каждого успешного эксперимента создаётся
# `<exp_id>.done` флаг. При повторе ноутбука уже посчитанные эксперименты
# пропускаются.

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
TRAIN_CSV = f"{SLOVO}/slovo100_train.csv"
VAL_CSV = f"{SLOVO}/slovo100_val.csv"
TEST_CSV = f"{SLOVO}/slovo100_test.csv"
LABEL_MAP = f"{SLOVO}/slovo100_label_map.csv"


# %% [markdown]
# ## 0. Resume from previous Kaggle session (if `diploma-results` is mounted)

# %%
PREV = "/kaggle/input/diploma-results"
if Path(PREV).exists():
    # !cp -rn $PREV/work_dirs/* $WORK_DIRS/ 2>/dev/null || true
    # !cp -n $PREV/results/all_results.csv $MASTER_CSV 2>/dev/null || true
    print("Restored from", PREV)
else:
    print("No previous results — starting fresh")

# %% [markdown]
# ## Helper


# %%
def run_train(exp_id: str, *extra_args: str) -> int:
    cmd = [
        "python",
        f"{PROJECT}/scripts/train.py",
        "--train-csv",
        TRAIN_CSV,
        "--val-csv",
        VAL_CSV,
        "--test-csv",
        TEST_CSV,
        "--label-map",
        LABEL_MAP,
        "--output-dir",
        WORK_DIRS,
        "--experiment-id",
        exp_id,
        "--append-results-to",
        MASTER_CSV,
        "--skip-if-done",
        *extra_args,
    ]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def run_continue(source_exp_id: str, new_exp_id: str | None = None, *extra_args: str) -> int:
    ckpt = Path(f"{WORK_DIRS}/{source_exp_id}/best.pt")
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return 1
    target_id = new_exp_id or f"{source_exp_id}_continue"
    return run_train(target_id, "--load-from", str(ckpt), *extra_args)


# Example: fine-tune A00 from best.pt for 10 more epochs.
# run_continue("A00_full_pipeline", "A00_full_pipeline_continue",
#              "--model", "videomae", "--pretrained", "k400",
#              "--epochs", "10", "--patience", "3", "--batch-size", "8")


# %% [markdown]
# ## A00 — full pipeline (highest priority — реплицирует SberDevices A00)

# %%
run_train("A00_full_pipeline", "--model", "videomae", "--pretrained", "k400")

# %% [markdown]
# ## A01–A12 — ablation: один компонент off за раз

# %%
ABLATIONS = [
    ("A01_no_image_augs", ["--no-aug-image"]),
    ("A02_no_mixup_cutmix", ["--mixup", "0", "--cutmix", "0"]),
    ("A03_no_video_augs", ["--no-aug-video"]),
    ("A04_no_boundary_shift", ["--boundary-shift", "0"]),
    ("A09_no_boundary_head", ["--no-boundary-head"]),
    ("A11_no_iou_loss", ["--no-iou-loss"]),
    ("A12_plain_ce", ["--no-iou-loss", "--mixup", "0", "--cutmix", "0"]),
]
for exp_id, extra in ABLATIONS:
    run_train(exp_id, "--model", "videomae", "--pretrained", "k400", *extra)

# %% [markdown]
# ## K0 — no Kinetics pretraining

# %%
run_train(
    "K0_no_kinetics", "--model", "videomae", "--pretrained", "none", "--epochs", "60"
)  # больше эпох — учится с нуля

# %% [markdown]
# ## K2a — fine-tune VideoMAE on Kinetics-400 sample

# %%
K400 = f"{MANIFESTS}/kinetics400_sample"
subprocess.run(
    [
        "python",
        f"{PROJECT}/scripts/train.py",
        "--train-csv",
        f"{K400}/kinetics400_sample_train.csv",
        "--val-csv",
        f"{K400}/kinetics400_sample_val.csv",
        "--test-csv",
        f"{K400}/kinetics400_sample_test.csv",
        "--label-map",
        f"{K400}/kinetics400_sample_label_map.csv",
        "--output-dir",
        WORK_DIRS,
        "--experiment-id",
        "K2a_kinetics_sample",
        "--append-results-to",
        MASTER_CSV,
        "--skip-if-done",
        "--model",
        "videomae",
        "--pretrained",
        "k400",
        "--no-iou-loss",
        "--no-boundary-head",
        "--mixup",
        "0",
        "--cutmix",
        "0",
        "--epochs",
        "20",
    ],
    check=False,
)

# %% [markdown]
# ## K2 — K2a checkpoint → Slovo

# %%
k2a_best = Path(f"{WORK_DIRS}/K2a_kinetics_sample/best.pt")
if k2a_best.exists():
    run_train(
        "K2_kinetics_sample_then_slovo",
        "--model",
        "videomae",
        "--pretrained",
        "k400",
        "--load-from",
        str(k2a_best),
    )
else:
    print("K2a checkpoint not found — skipping K2")

# %% [markdown]
# ## Snapshot

# %%
import pandas as pd

if Path(MASTER_CSV).exists():
    df = pd.read_csv(MASTER_CSV)
    df["top1"] = pd.to_numeric(df["top1"], errors="coerce")
    df.sort_values("top1", ascending=False)
