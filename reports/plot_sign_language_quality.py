import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sign_language_quality_data.csv"
OUT_DIR = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_line(years: pd.Series, scores: pd.Series):
    x = years.to_numpy(dtype=float).reshape(-1, 1)
    y = scores.to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(x, y)
    return model.predict(x), model.coef_[0]


def style_axes(ax, title, ylabel="Quality score, %"):
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(2022.8, 2026.2)
    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_comparable_trends(df: pd.DataFrame):
    trendable = df.groupby("benchmark").filter(lambda group: len(group) >= 2)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = plt.get_cmap("tab10")

    for idx, (benchmark, group) in enumerate(trendable.groupby("benchmark")):
        group = group.sort_values("year")
        color = colors(idx % 10)
        ax.plot(
            group["year"], group["score"], marker="o", linewidth=2, label=benchmark, color=color
        )
        pred, slope = fit_line(group["year"], group["score"])
        ax.plot(group["year"], pred, linestyle="--", alpha=0.65, color=color)
        last = group.iloc[-1]
        ax.annotate(
            f"{benchmark}: {slope:+.2f} pp/year",
            xy=(last["year"], last["score"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

    style_axes(ax, "Published sign-language recognition quality, 2023-2026")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparable_benchmark_trends.png", dpi=220)
    plt.close(fig)


def plot_all_reported_points(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6.2))

    for benchmark, group in df.sort_values("year").groupby("benchmark"):
        ax.scatter(group["year"], group["score"], s=70, label=benchmark)
        for _, row in group.iterrows():
            ax.annotate(
                row["method"],
                xy=(row["year"], row["score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.82,
            )

    pred, slope = fit_line(df["year"], df["score"])
    order = df["year"].argsort()
    ax.plot(
        df["year"].iloc[order],
        pred[order],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"LinearRegression overall trend: {slope:+.2f} pp/year",
    )

    style_axes(ax, "All collected published scores (mixed benchmarks)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "all_reported_scores_with_sklearn_trend.png", dpi=220)
    plt.close(fig)


def plot_language_summary(df: pd.DataFrame):
    best_by_language_year = (
        df.groupby(["language", "year"], as_index=False)["score"]
        .max()
        .sort_values(["language", "year"])
    )

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for language, group in best_by_language_year.groupby("language"):
        ax.plot(group["year"], group["score"], marker="o", linewidth=2, label=language)
        if len(group) >= 2:
            pred, _ = fit_line(group["year"], group["score"])
            ax.plot(group["year"], pred, linestyle="--", alpha=0.55)

    style_axes(ax, "Best collected score by sign language and year")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "language_summary_trends.png", dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    plot_comparable_trends(df)
    plot_all_reported_points(df)
    plot_language_summary(df)

    summary = (
        df.groupby("benchmark")
        .agg(first_year=("year", "min"), last_year=("year", "max"), best_score=("score", "max"))
        .sort_values(["last_year", "benchmark"])
    )
    print(summary.to_string())
    print(f"\nWrote figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
