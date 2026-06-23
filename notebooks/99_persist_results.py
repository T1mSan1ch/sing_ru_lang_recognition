# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 99 — Persist results before Save Version
#
# Чистит epoch-чекпоинты (оставляет `best.pt`, `metrics.json`, `train.log`),
# упаковывает `work_dirs/` + `results/` + `manifests/` в один tar.gz. После этого нажмите **Save
# Version** в Kaggle UI, и Output ноутбука можно опубликовать как Kaggle
# Dataset `diploma-results`.

# %%
import shutil
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
WORK_DIRS = WORK / "work_dirs"
RESULTS = WORK / "results"
MANIFESTS = WORK / "manifests"
OUT = WORK / "kaggle_output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Удалить промежуточные веса (оставить best.pt, metrics.json, train.log)

# %%
removed = 0
for ckpt in WORK_DIRS.rglob("*.pt"):
    if ckpt.name != "best.pt":
        ckpt.unlink()
        removed += 1
print(f"Removed {removed} non-best checkpoints")

# %% [markdown]
# ## 2. Архивирование

# %%
archive = OUT / "diploma_results.tar.gz"
if archive.exists():
    archive.unlink()
subprocess.run(
    [
        "tar",
        "-czf",
        str(archive),
        "-C",
        str(WORK),
        "work_dirs",
        "results",
        "manifests",
    ],
    check=True,
)
print(f"Archived to {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 3. Master CSV — продублировать в kaggle_output/ для удобства

# %%
master = RESULTS / "all_results.csv"
if master.exists():
    shutil.copy(master, OUT / "all_results.csv")
    print(f"Copied master CSV ({master.stat().st_size / 1024:.1f} KB)")

# %%
print("Ready to Save Version. Files in /kaggle/working/kaggle_output/:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")
