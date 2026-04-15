"""
experiment.py — Главный оркестратор экспериментов.

Запускает полный цикл для всех комбинаций:
  {baseline, standard, advanced} × {logistic_regression, decision_tree, xgboost}

Сохраняет:
  outputs/metrics_summary.csv        — сводная таблица метрик
  outputs/cleaning_steps.csv         — что убрала каждая очистка
  outputs/plots/fi_*.png             — feature importance
  outputs/plots/shap/shap_*.png      — SHAP summary plots
  outputs/plots/lime/lime_*.png      — LIME для одного клиента
  outputs/interpretability_summary.csv — топ-признаки по пайплайнам
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")

from .config import (
    OUTPUT_DIR, PLOT_DIR, SHAP_DIR, LIME_DIR,
    TARGET_COL, PIPELINE_NAMES,
)
from .data_loader    import load_raw_data, print_eda_report
from .preprocessing  import PIPELINES
from .features       import FEATURE_BUILDERS, scale_features
from .models         import (
    MODELS, split_customer_features, train_and_evaluate,
    results_to_dataframe, ModelResult,
)
from .interpretability import (
    get_feature_importance, plot_feature_importance,
    compute_shap_values, plot_shap_summary, shap_mean_abs,
    compare_shap_rankings, compare_feature_importance,
    explain_with_lime, plot_lime_explanation, lime_to_dataframe,
    interpretability_summary, SHAP_AVAILABLE, LIME_AVAILABLE,
)


# ─── Нужно ли масштабировать для данной модели? ───────────────────────────────
_SCALE_FOR = {"logistic_regression"}


def run_experiment(
    run_shap : bool = True,
    run_lime : bool = True,
    run_cv   : bool = True,
    models   : list[str] | None = None,
    pipelines: list[str] | None = None,
) -> list[ModelResult]:
    """
    Полный запуск. Возвращает список ModelResult для дальнейшего анализа.

    Параметры:
        run_shap  — вычислять SHAP (может быть медленно для XGB на большом наборе)
        run_lime  — вычислять LIME
        run_cv    — кросс-валидация на train
        models    — список моделей для запуска (None → все)
        pipelines — список пайплайнов (None → все)
    """
    models    = models    or MODELS
    pipelines = pipelines or list(PIPELINES.keys())

    # ── 1. Загрузка данных ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Загрузка данных...")
    raw_df = load_raw_data()
    print_eda_report(raw_df)

    # ── 2. Очистка по каждому пайплайну ──────────────────────────────────────
    cleaned   : dict[str, pd.DataFrame]     = {}
    cl_reports: list                         = []

    for pname in pipelines:
        logger.info("─ Очистка: %s", pname)
        clean_df, report = PIPELINES[pname](raw_df)
        cleaned[pname]   = clean_df
        cl_reports.append(report.to_dataframe())

    # Сохраняем отчёт об очистке
    cleaning_df = pd.concat(cl_reports, ignore_index=True)
    cleaning_df.to_csv(OUTPUT_DIR / "cleaning_steps.csv", index=False)
    logger.info("Шаги очистки сохранены → outputs/cleaning_steps.csv")

    # ── 3. Генерация признаков ────────────────────────────────────────────────
    features_cache: dict[str, pd.DataFrame] = {}

    for pname in pipelines:
        logger.info("─ Признаки: %s", pname)
        feat_df = FEATURE_BUILDERS[pname](cleaned[pname])
        features_cache[pname] = feat_df
        logger.info(
            "  %s: %d клиентов, %d признаков, target pos=%.1f%%",
            pname, len(feat_df), feat_df.shape[1] - 1,
            feat_df[TARGET_COL].mean() * 100,
        )

    # ── 4. Обучение всех комбинаций ───────────────────────────────────────────
    all_results: list[ModelResult] = []

    for pname in pipelines:
        feat_df = features_cache[pname]
        X_tr, X_te, y_tr, y_te = split_customer_features(feat_df)

        for mname in models:
            # Масштабирование только для LR
            if mname in _SCALE_FOR:
                X_tr_m, X_te_m, _ = scale_features(X_tr, X_te)
            else:
                X_tr_m, X_te_m = X_tr.copy(), X_te.copy()

            # Заполняем оставшиеся NaN медианой (после масштабирования)
            X_tr_m = X_tr_m.fillna(X_tr_m.median(numeric_only=True))
            X_te_m = X_te_m.fillna(X_tr_m.median(numeric_only=True))

            result = train_and_evaluate(
                pipeline_name = pname,
                model_name    = mname,
                X_train       = X_tr_m,
                X_test        = X_te_m,
                y_train       = y_tr,
                y_test        = y_te,
                run_cv        = run_cv,
            )
            all_results.append(result)

    # ── 5. Метрики → CSV ──────────────────────────────────────────────────────
    metrics_df = results_to_dataframe(all_results)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    logger.info("Метрики сохранены → outputs/metrics_summary.csv")
    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА МЕТРИК")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    # ── 6. Feature Importance ──────────────────────────────────────────────────
    logger.info("─ Feature Importance...")
    fi_by_model: dict[str, list] = {m: [] for m in models}

    for res in all_results:
        fi = get_feature_importance(res.model, res.feature_names)
        plot_feature_importance(
            fi,
            title     = f"FI: {res.pipeline_name} / {res.model_name}",
            save_path = PLOT_DIR / f"fi_{res.experiment_id}.png",
            top_n     = 15,
        )
        fi_by_model[res.model_name].append((res.pipeline_name, fi))

    # Сравнение FI по пайплайнам для каждой модели
    for mname, fi_list in fi_by_model.items():
        compare_feature_importance(
            fi_list,
            save_path = PLOT_DIR / f"fi_compare_{mname}.png",
            top_n     = 10,
        )

    # ── 7. SHAP ────────────────────────────────────────────────────────────────
    shap_data: dict[str, list] = {m: [] for m in models}

    if run_shap and SHAP_AVAILABLE:
        logger.info("─ SHAP...")
        for res in all_results:
            # Считаем на тестовой выборке (≤200 образцов достаточно)
            X_sample = res.X_test.sample(
                min(200, len(res.X_test)), random_state=42
            )
            sv = compute_shap_values(res.model, X_sample)
            if sv is None:
                continue

            plot_shap_summary(
                sv, X_sample,
                title     = f"SHAP: {res.pipeline_name} / {res.model_name}",
                save_path = SHAP_DIR / f"shap_{res.experiment_id}.png",
            )
            shap_data[res.model_name].append((res.pipeline_name, sv, res.feature_names))

        # Сравнение SHAP-рангов
        for mname, sl in shap_data.items():
            if len(sl) > 1:
                compare_shap_rankings(
                    sl,
                    save_path = SHAP_DIR / f"shap_compare_{mname}.png",
                    top_n     = 10,
                )
    elif not SHAP_AVAILABLE:
        logger.warning("SHAP пропущен (не установлен).")

    # ── 8. LIME ────────────────────────────────────────────────────────────────
    if run_lime and LIME_AVAILABLE:
        logger.info("─ LIME...")
        for res in all_results:
            # Берём первого клиента из test, у которого target=1
            pos_idx_list = res.y_test[res.y_test == 1].index.tolist()
            if not pos_idx_list:
                continue

            # Позиция в X_test (iloc)
            sample_pos = res.X_test.index.get_loc(pos_idx_list[0])

            exp = explain_with_lime(
                model      = res.model,
                X_train    = res.X_train,
                X_test     = res.X_test,
                sample_idx = sample_pos,
            )
            if exp is None:
                continue

            plot_lime_explanation(
                exp,
                title     = f"LIME: {res.pipeline_name} / {res.model_name}",
                save_path = LIME_DIR / f"lime_{res.experiment_id}.png",
            )

            lime_df = lime_to_dataframe(exp)
            lime_df.to_csv(
                LIME_DIR / f"lime_{res.experiment_id}.csv", index=False
            )
    elif not LIME_AVAILABLE:
        logger.warning("LIME пропущен (не установлен).")

    # ── 9. Итоговый отчёт интерпретируемости ──────────────────────────────────
    interp_df = interpretability_summary(all_results)
    interp_df.to_csv(OUTPUT_DIR / "interpretability_summary.csv", index=False)
    logger.info("Отчёт по интерпретируемости → outputs/interpretability_summary.csv")

    print("\n" + "=" * 60)
    print("ИНТЕРПРЕТИРУЕМОСТЬ: топ-3 признака по пайплайнам")
    print("=" * 60)
    print(interp_df.to_string(index=False))
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_experiment(
        run_shap  = True,
        run_lime  = True,
        run_cv    = True,
    )
