"""
interpretability.py — Объяснение моделей: SHAP, LIME, Feature Importance.

Главный вопрос ВКР: как изменяется интерпретация при смене пайплайна?
Поэтому все функции сохраняют результаты в структурированном виде,
пригодном для сравнения между экспериментами.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # без GUI
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Опциональные зависимости
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap не установлен: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("lime не установлен: pip install lime")


# ─── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Возвращает DataFrame с importance для любой из трёх моделей.
    LR  → abs(coef_)
    DT  → feature_importances_
    XGB → feature_importances_
    """
    model_type = type(model).__name__

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        kind = "abs_coef"
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        kind = "gain"
    else:
        raise ValueError(f"Модель {model_type} не поддерживает feature importance.")

    df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importance,
        "kind"      : kind,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def plot_feature_importance(
    fi_df       : pd.DataFrame,
    title       : str,
    save_path   : Optional[Path] = None,
    top_n       : int = 15,
) -> None:
    df = fi_df.head(top_n).iloc[::-1]   # перевернуть для горизонтального бара

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(df)))
    ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_xlabel("Важность признака")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance сохранён: %s", save_path)
    plt.close()


def plot_signed_coefficients(
    model,
    feature_names: list,
    title: str,
    save_path: Optional[Path] = None,
    top_n: int = 15,
) -> None:
    """
    Signed coefficients for linear models.
    """
    if not hasattr(model, "coef_"):
        return

    coef_df = (
        pd.DataFrame({"feature": feature_names, "coef": model.coef_[0]})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(top_n)
        .sort_values("coef")
    )

    colors = np.where(coef_df["coef"] >= 0, "#55A868", "#C44E52")
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    ax.barh(coef_df["feature"], coef_df["coef"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Signed coefficients сохранены: %s", save_path)
    plt.close()


def compare_feature_importance(
    fi_list     : list[tuple[str, pd.DataFrame]],  # [(label, fi_df), ...]
    save_path   : Optional[Path] = None,
    top_n       : int = 10,
) -> pd.DataFrame:
    """
    Сравниваем важность признаков между пайплайнами для одной модели.
    Возвращает сводный DataFrame с нормированными важностями.
    """
    frames = []
    for label, fi_df in fi_list:
        df = fi_df.head(top_n).copy()
        df["importance_norm"] = df["importance"] / df["importance"].sum()
        df["experiment"]      = label
        frames.append(df[["feature", "importance_norm", "experiment"]])

    combined = pd.concat(frames, ignore_index=True)

    # Pivot для тепловой карты
    pivot = combined.pivot_table(
        index="feature", columns="experiment",
        values="importance_norm", fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Нормированная важность")
    ax.set_title("Сравнение важности признаков по пайплайнам")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Сравнение FI сохранено: %s", save_path)
    plt.close()

    return pivot


# ─── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Считает SHAP-значения. Гарантирует возврат 2D массива (samples, features).
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP недоступен.")
        return None

    model_type = type(model).__name__
    logger.info("Вычисление SHAP для %s (%d образцов)...", model_type, len(X))

    try:
        if model_type in ("XGBClassifier", "LGBMClassifier", "DecisionTreeClassifier"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
        else:
            # Для линейных моделей и прочих
            explainer = shap.Explainer(model, X)
            sv = explainer.shap_values(X)

        # --- ОБРАБОТКА ФОРМАТА SHAP ---
        # 1. Если это объект Explanation (новые версии), берем .values
        if hasattr(sv, "values"):
            sv = sv.values

        # 2. Если это список (старые версии для деревьев), берем значения для класса 1
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        # 3. Если массив 3D (samples, features, classes), берем срез для класса 1
        if len(sv.shape) == 3:
            sv = sv[:, :, 1]

        return np.array(sv)

    except Exception as e:
        logger.warning("SHAP упал: %s. Пробую KernelExplainer (может быть медленно)...", e)
        try:
            # Резервный метод
            bg = shap.sample(X, 50)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            sv = explainer.shap_values(X, nsamples=100)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            return np.array(sv)
        except Exception as fallback_error:
            logger.warning("KernelExplainer тоже завершился с ошибкой: %s", fallback_error)
            return None


def shap_mean_abs(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Mean |SHAP| по всем образцам. Гарантирует 1D результат для DataFrame."""
    # Берем модуль и среднее по строкам
    vals = np.abs(shap_values).mean(axis=0)

    # Если вдруг на выходе осталась лишняя размерность (напр. (19, 1)), выпрямляем её
    if len(vals.shape) > 1:
        vals = vals.ravel()

    if len(vals) != len(feature_names):
        raise ValueError(
            "Размерность SHAP не совпадает с числом признаков: "
            f"{len(vals)} vs {len(feature_names)}"
        )

    return pd.DataFrame({
        "feature"   : feature_names,
        "mean_abs_shap": vals,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

def plot_shap_summary(
    shap_values : np.ndarray,
    X           : pd.DataFrame,
    title       : str,
    save_path   : Optional[Path] = None,
    max_display : int = 15,
) -> None:
    if not SHAP_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.45)))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary сохранён: %s", save_path)
    plt.close("all")


def plot_shap_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
    max_display: int = 15,
) -> None:
    if not SHAP_AVAILABLE:
        return

    shap_df = shap_mean_abs(shap_values, list(X.columns)).head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, max_display * 0.35)))
    ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#4C72B0", alpha=0.9)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP bar сохранён: %s", save_path)
    plt.close()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_name: str,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    if not SHAP_AVAILABLE or feature_name not in X.columns:
        return

    try:
        shap.dependence_plot(
            feature_name,
            shap_values,
            X,
            interaction_index=None,
            show=False,
        )
        plt.title(title, fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP dependence сохранён: %s", save_path)
        plt.close("all")
    except Exception as exc:
        logger.warning("SHAP dependence skipped for %s: %s", feature_name, exc)


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    sample_idx: int,
    title: str,
    save_path: Optional[Path] = None,
    max_display: int = 12,
) -> None:
    if not SHAP_AVAILABLE or len(X) == 0:
        return

    sample_idx = min(sample_idx, len(X) - 1)
    values = np.asarray(shap_values[sample_idx]).ravel()
    base_value = 0.0
    data_row = X.iloc[sample_idx].values
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data_row,
        feature_names=list(X.columns),
    )

    try:
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(title, fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP waterfall сохранён: %s", save_path)
        plt.close("all")
    except Exception as exc:
        logger.warning("SHAP waterfall skipped: %s", exc)
def compare_shap_rankings(
    shap_list   : list[tuple[str, np.ndarray, list]],  # [(label, shap_vals, feat_names)]
    save_path   : Optional[Path] = None,
    top_n       : int = 10,
) -> pd.DataFrame:
    """Тепловая карта: насколько стабилен топ-N SHAP между пайплайнами."""
    frames = []
    for label, sv, feat_names in shap_list:
        df = shap_mean_abs(sv, feat_names).head(top_n).copy()
        df["importance_norm"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
        df["experiment"]      = label
        frames.append(df[["feature", "importance_norm", "experiment"]])

    combined = pd.concat(frames, ignore_index=True)
    pivot    = combined.pivot_table(
        index="feature", columns="experiment",
        values="importance_norm", fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Нормированный |SHAP|")
    ax.set_title("Сравнение SHAP-важности по пайплайнам")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return pivot


# ─── LIME ─────────────────────────────────────────────────────────────────────

def explain_with_lime(
    model,
    X_train     : pd.DataFrame,
    X_test      : pd.DataFrame,
    sample_idx  : int = 0,
    num_features: int = 10,
    num_samples : int = 500,
) -> Optional[object]:
    """
    LIME-объяснение для одного клиента.
    Возвращает объект LimeTabularExplanation.
    """
    if not LIME_AVAILABLE:
        logger.warning("LIME недоступен.")
        return None

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X_train.values,
        feature_names  = list(X_train.columns),
        class_names    = ["no_purchase", "purchase"],
        mode           = "classification",
        discretize_continuous = True,
        random_state   = 42,
    )

    exp = explainer.explain_instance(
        data_row       = X_test.iloc[sample_idx].values,
        predict_fn     = model.predict_proba,
        num_features   = num_features,
        num_samples    = num_samples,
    )
    return exp


def plot_lime_explanation(
    exp,
    title       : str,
    save_path   : Optional[Path] = None,
) -> None:
    if exp is None:
        return

    fig = exp.as_pyplot_figure(label=1)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("LIME plot сохранён: %s", save_path)
    plt.close()


def lime_to_dataframe(exp) -> pd.DataFrame:
    """Превращаем LIME explanation в DataFrame для сравнения."""
    if exp is None:
        return pd.DataFrame()
    rows = exp.as_list(label=1)
    return pd.DataFrame(rows, columns=["rule", "weight"])


# ─── Сводный отчёт интерпретируемости ────────────────────────────────────────

def interpretability_summary(
    results_list: list,  # list[ModelResult]
) -> pd.DataFrame:
    """
    Для каждого эксперимента считаем:
    - топ-3 признака по FI
    - топ-3 признака по SHAP (если доступно)
    - стабильность (Jaccard между FI и SHAP топами)
    """
    rows = []
    for res in results_list:
        fi = get_feature_importance(res.model, res.feature_names)
        top3_fi = fi["feature"].head(3).tolist()

        row = {
            "experiment" : res.experiment_id,
            "pipeline"   : res.pipeline_name,
            "cleaning_pipeline": res.cleaning_pipeline or res.pipeline_name,
            "feature_builder": res.feature_builder or res.pipeline_name,
            "model"      : res.model_name,
            "n_features" : len(res.feature_names),
            "top3_fi"    : ", ".join(top3_fi),
            "roc_auc"    : res.metrics.get("roc_auc"),
            "f1"         : res.metrics.get("f1"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def fi_shap_agreement_summary(
    results_list: list,
    shap_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare top FI and top SHAP features for each experiment.
    """
    rows = []
    for res in results_list:
        experiment_id = res.experiment_id
        fi = get_feature_importance(res.model, res.feature_names)
        top3_fi = fi["feature"].head(3).tolist()
        top5_fi = fi["feature"].head(5).tolist()

        row = {
            "experiment": experiment_id,
            "pipeline": res.pipeline_name,
            "cleaning_pipeline": res.cleaning_pipeline or res.pipeline_name,
            "feature_builder": res.feature_builder or res.pipeline_name,
            "model": res.model_name,
            "top3_fi": ", ".join(top3_fi),
            "top5_fi": ", ".join(top5_fi),
            "top3_shap": "",
            "top5_shap": "",
            "jaccard_top3": np.nan,
            "jaccard_top5": np.nan,
        }

        shap_values = shap_lookup.get(experiment_id)
        if shap_values is not None:
            shap_df = shap_mean_abs(shap_values, res.feature_names)
            top3_shap = shap_df["feature"].head(3).tolist()
            top5_shap = shap_df["feature"].head(5).tolist()

            fi3, shap3 = set(top3_fi), set(top3_shap)
            fi5, shap5 = set(top5_fi), set(top5_shap)
            row["top3_shap"] = ", ".join(top3_shap)
            row["top5_shap"] = ", ".join(top5_shap)
            row["jaccard_top3"] = round(len(fi3 & shap3) / len(fi3 | shap3), 4)
            row["jaccard_top5"] = round(len(fi5 & shap5) / len(fi5 | shap5), 4)

        rows.append(row)

    return pd.DataFrame(rows)
