"""
report.py — Визуализация итогов эксперимента.

Запускайте после experiment.py (файлы CSV уже должны быть в outputs/).
Создаёт:
  outputs/plots/metrics_comparison.png  — AUC/F1 по пайплайнам и моделям
  outputs/plots/cleaning_impact.png     — сколько данных убрал каждый шаг
  outputs/report_card.csv               — финальная карточка для таблицы в ВКР
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, PLOT_DIR

logger = logging.getLogger(__name__)

PALETTE = {
    "baseline" : "#4C72B0",
    "standard" : "#DD8452",
    "advanced" : "#55A868",
}
MODEL_MARKERS = {
    "logistic_regression": "o",
    "decision_tree"      : "s",
    "xgboost"            : "D",
}


def plot_metrics_comparison(metrics_csv: Path | None = None) -> None:
    """
    Grouped bar chart: AUC и F1 для каждой модели × каждого пайплайна.
    """
    path = metrics_csv or OUTPUT_DIR / "metrics_summary.csv"
    df   = pd.read_csv(path)

    models    = df["model"].unique().tolist()
    pipelines = df["pipeline"].unique().tolist()
    n_models  = len(models)
    n_pipes   = len(pipelines)

    x      = np.arange(n_models)
    width  = 0.22
    offset = np.linspace(-(n_pipes - 1) / 2, (n_pipes - 1) / 2, n_pipes) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in zip(axes, ["roc_auc", "f1"], ["ROC-AUC", "F1"]):
        for i, pname in enumerate(pipelines):
            sub    = df[df["pipeline"] == pname].set_index("model")
            values = [sub.loc[m, metric] if m in sub.index else 0 for m in models]
            bars   = ax.bar(x + offset[i], values, width,
                            label=pname, color=PALETTE.get(pname, f"C{i}"),
                            alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=12)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} по моделям и пайплайнам")
        ax.legend(title="Pipeline")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Влияние предобработки на качество модели", fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = PLOT_DIR / "metrics_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("metrics_comparison.png сохранён")
    plt.close()


def plot_cleaning_impact(cleaning_csv: Path | None = None) -> None:
    """
    Горизонтальные stacked bar: сколько строк убрал каждый шаг очистки.
    """
    path = cleaning_csv or OUTPUT_DIR / "cleaning_steps.csv"
    df   = pd.read_csv(path)

    if "dataset_split" in df.columns:
        df = df[df["dataset_split"] == "train"].copy()

    pipelines = df["pipeline"].unique().tolist()
    fig, axes = plt.subplots(1, len(pipelines), figsize=(6 * len(pipelines), 5),
                             sharey=False)
    if len(pipelines) == 1:
        axes = [axes]

    for ax, pname in zip(axes, pipelines):
        sub = df[df["pipeline"] == pname].copy()
        sub["rows_kept"]    = sub["rows_after"]
        sub["rows_removed"] = sub["rows_removed"]

        y_pos = range(len(sub))
        ax.barh(y_pos, sub["rows_after"],   label="Осталось",  color="#55A868", alpha=0.8)
        ax.barh(y_pos, sub["rows_removed"], left=sub["rows_after"],
                label="Удалено", color="#C44E52", alpha=0.7)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["step"].tolist())
        ax.set_xlabel("Строк")
        ax.set_title(f"Очистка: {pname}")
        ax.legend(loc="lower right", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))

    plt.suptitle("Влияние шагов очистки на размер датасета", fontsize=12, y=1.01)
    plt.tight_layout()
    save_path = PLOT_DIR / "cleaning_impact.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("cleaning_impact.png сохранён")
    plt.close()


def generate_report_card(metrics_csv: Path | None = None) -> pd.DataFrame:
    """
    Финальная карточка для таблицы в ВКР:
    pipeline × model → AUC, F1, ΔFI_top1 (лучший признак).
    """
    path = metrics_csv or OUTPUT_DIR / "metrics_summary.csv"
    df   = pd.read_csv(path)

    # Лучший пайплайн по AUC для каждой модели
    best_idx = df.groupby("model")["roc_auc"].idxmax()
    df["is_best"] = df.index.isin(best_idx)

    out_path = OUTPUT_DIR / "report_card.csv"
    df.to_csv(out_path, index=False)
    logger.info("report_card.csv сохранён")
    return df


def run_report() -> None:
    plot_metrics_comparison()
    plot_cleaning_impact()
    card = generate_report_card()
    print("\nФинальная карточка результатов:")
    print(card.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run_report()
