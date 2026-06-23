# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — SigLIP2-style Foundation Model (F0–F3)
#
# F0 — SigLIP2 zero-shot
# F3 — frozen SigLIP2 + temporal decoder, fully supervised
# F2 — few-shot k = 1, 2, 4, 8 × seeds 42, 43, 44

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
SCRIPT = f"{PROJECT}/scripts/train_signvlm.py"


def run(*args) -> int:
    cmd = ["python", SCRIPT, *args, "--append-results-to", MASTER_CSV, "--skip-if-done"]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


# %% [markdown]
# ## F0 — SigLIP2 zero-shot

# %%
run(
    "--mode",
    "zero_shot",
    "--test-csv",
    f"{SLOVO}/slovo100_test.csv",
    "--label-map",
    f"{SLOVO}/slovo100_label_map.csv",
    "--output-dir",
    WORK_DIRS,
    "--experiment-id",
    "F0_siglip2_zeroshot",
    "--prompt",
    "жест {label}",
)

# %% [markdown]
# ## F3 — Full supervised SigLIP2 + temporal decoder

# %%
run(
    "--mode",
    "train",
    "--train-csv",
    f"{SLOVO}/slovo100_train.csv",
    "--test-csv",
    f"{SLOVO}/slovo100_test.csv",
    "--label-map",
    f"{SLOVO}/slovo100_label_map.csv",
    "--output-dir",
    WORK_DIRS,
    "--experiment-id",
    "F3_siglip2_full_supervised",
    "--epochs",
    "30",
)

# %% [markdown]
# ## F2 — Few-shot (k=1,2,4,8 × seeds 42,43,44)

# %%
for seed in [42, 43, 44]:
    for shot in [1, 2, 4, 8]:
        prefix = f"shot{shot}_seed{seed}"
        run(
            "--mode",
            "train",
            "--train-csv",
            f"{MANIFESTS}/slovo100_fewshot/{prefix}_train.csv",
            "--test-csv",
            f"{MANIFESTS}/slovo100_fewshot/{prefix}_test.csv",
            "--label-map",
            f"{SLOVO}/slovo100_label_map.csv",
            "--output-dir",
            WORK_DIRS,
            "--experiment-id",
            f"F2_siglip2_{prefix}",
            "--epochs",
            "20",
            "--patience",
            "4",
            "--seed",
            str(seed),
        )

# %% [markdown]
# ## Snapshot

# %%
import pandas as pd

if Path(MASTER_CSV).exists():
    df = pd.read_csv(MASTER_CSV)
    df["top1"] = pd.to_numeric(df["top1"], errors="coerce")
    df[df.experiment_id.astype(str).str.startswith("F")].sort_values("top1", ascending=False)
