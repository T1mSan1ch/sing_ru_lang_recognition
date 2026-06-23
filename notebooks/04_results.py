# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 04 — Сводная таблица и графики

# %%
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

WORK = "/kaggle/working"
PROJECT = f"{WORK}/diploma_claude"
WORK_DIRS = f"{WORK}/work_dirs"
RESULTS = f"{WORK}/results"
MASTER_CSV = f"{RESULTS}/all_results.csv"
SUMMARY = f"{RESULTS}/all_summary.csv"

subprocess.run(
    [
        "python",
        f"{PROJECT}/scripts/aggregate_results.py",
        "--inputs",
        WORK_DIRS,
        "--master-csv",
        MASTER_CSV,
        "--output",
        SUMMARY,
    ]
)
df = pd.read_csv(SUMMARY)
for col in ["top1", "top5", "macro_f1", "test_top1"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df.sort_values("top1", ascending=False).reset_index(drop=True)

# %% [markdown]
# ## Ablation: A00 baseline + deltas

# %%
ab = df[df.experiment_id.astype(str).str.startswith("A")].copy()
if not ab.empty:
    a00 = ab[ab.experiment_id == "A00_full_pipeline"]["top1"]
    if not a00.empty:
        ab["delta_vs_A00"] = (ab["top1"] - a00.values[0]).round(4)
    print(
        ab[
            [
                "experiment_id",
                "top1",
                "test_top1",
                "delta_vs_A00" if "delta_vs_A00" in ab else "top5",
            ]
        ].to_string(index=False)
    )

# %% [markdown]
# ## Ablation bar chart

# %%
if not ab.empty and len(ab) > 1:
    s = ab.sort_values("top1", ascending=True)
    colors = ["steelblue" if e == "A00_full_pipeline" else "lightcoral" for e in s.experiment_id]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(s.experiment_id, s.top1, color=colors)
    ax.set_xlabel("Top-1 accuracy (val)")
    ax.set_title("Ablation — Effect of removing each pipeline component")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/ablation_chart.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Few-shot learning curve

# %%
fs = df[df.experiment_id.astype(str).str.startswith("F2_shot")].copy()
if not fs.empty:
    fs["shot"] = fs["experiment_id"].str.extract(r"shot(\d+)").astype(int)
    curve = fs.groupby("shot")["top1"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        curve.index,
        curve["mean"],
        yerr=curve["std"],
        marker="o",
        capsize=4,
        label="Few-shot CLIP+Decoder",
    )
    f3 = df.loc[df.experiment_id == "F3_signvlm_full_supervised", "top1"]
    if not f3.empty:
        ax.axhline(
            f3.values[0],
            linestyle="--",
            color="gray",
            label=f"Full supervised ({f3.values[0]:.3f})",
        )
    ax.set_xlabel("Shots per class (k)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels([1, 2, 4, 8])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fewshot_curve.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Cross-lingual table

# %%
cl = df[df.experiment_id.astype(str).str.match(r"^C\d_")].copy()
if not cl.empty:
    print(
        cl[["experiment_id", "top1", "test_top1", "macro_f1"]]
        .sort_values("top1", ascending=False)
        .to_string(index=False)
    )

# %% [markdown]
# ## Final ranked table

# %%
df.sort_values("top1", ascending=False).reset_index(drop=True)
