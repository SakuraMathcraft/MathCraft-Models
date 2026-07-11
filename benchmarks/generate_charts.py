from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"


def _save(fig: plt.Figure, name: str) -> None:
    CHARTS.mkdir(exist_ok=True)
    fig.savefig(CHARTS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quality_chart() -> None:
    labels = ["UniMER-Test", "MathWriting test"]
    bleu = [0.7946, 0.5467]
    cdm = [0.9288, 0.7500]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars_bleu = ax.bar([i - 0.18 for i in x], bleu, 0.36, label="BLEU-4", color="#2878B5")
    bars_cdm = ax.bar([i + 0.18 for i in x], cdm, 0.36, label="Official CDM", color="#D95F02")
    ax.set_title("MathCraft OCR formula recognition quality")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x), labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.bar_label(bars_bleu, fmt="%.4f", padding=3)
    ax.bar_label(bars_cdm, fmt="%.4f", padding=3)
    fig.text(
        0.5,
        -0.02,
        "Scores are reported within each dataset's fixed protocol; datasets are not directly comparable.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    _save(fig, "formula_quality.png")


def reliability_chart() -> None:
    labels = ["UniMER-Test", "MathWriting test", "OpenStax pages"]
    completed = [23757, 7644, 200]
    failed = [0, 0, 0]
    empty = [0, 0, 0]
    success = [100 * (total - bad - blank) / total for total, bad, blank in zip(completed, failed, empty)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, success, color=["#2878B5", "#4C956C", "#D95F02"], width=0.58)
    ax.set_title("End-to-end benchmark completion")
    ax.set_ylabel("Non-empty successful outputs (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(
        bars,
        labels=[f"100%\n{n:,} samples" for n in completed],
        label_type="center",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "All recorded runs used CUDAExecutionProvider. OpenStax measures mixed-page completion, not formula accuracy.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    _save(fig, "benchmark_completion.png")


if __name__ == "__main__":
    quality_chart()
    reliability_chart()
