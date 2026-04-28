"""
experiment.py - Main experiment orchestrator.

Leakage-safe flow:
1. Split customers into train/test before any fitted transformations.
2. Fit cleaning thresholds on train history only.
3. Fit feature artifacts on cleaned train history only.
4. Apply the same fitted artifacts to train and test.
"""

import argparse
import logging
import warnings
from dataclasses import dataclass

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")

from .config import CUTOFF_DATE, OUTPUT_DIR, PLOT_DIR, SHAP_DIR, LIME_DIR, TARGET_COL
from .data_loader import load_raw_data, print_eda_report
from .features import (
    apply_feature_builder,
    build_target,
    fit_feature_builder,
    scale_features,
)
from .interpretability import (
    LIME_AVAILABLE,
    SHAP_AVAILABLE,
    compare_feature_importance,
    compare_shap_rankings,
    compute_shap_values,
    explain_with_lime,
    fi_shap_agreement_summary,
    get_feature_importance,
    interpretability_summary,
    lime_to_dataframe,
    plot_feature_importance,
    plot_lime_explanation,
    plot_shap_bar,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_signed_coefficients,
    shap_mean_abs,
)
from .models import MODELS, ModelResult, results_to_dataframe, split_customer_ids, train_and_evaluate
from .preprocessing import apply_cleaning_pipeline, fit_cleaning_pipeline


_SCALE_FOR = {"logistic_regression"}
DEFAULT_PIPELINES = ["baseline", "standard", "advanced"]


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    cleaning_pipeline: str
    feature_builder: str


def _default_specs(pipelines: list[str]) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            label=pipeline_name,
            cleaning_pipeline=pipeline_name,
            feature_builder=pipeline_name,
        )
        for pipeline_name in pipelines
    ]


def _single_model_profile(
    description: str,
    model_name: str,
    pipelines: list[str] | None = None,
) -> dict:
    return {
        "specs": _default_specs(pipelines or DEFAULT_PIPELINES),
        "models": [model_name],
        "description": description,
    }


def _implemented_all_specs() -> list[ExperimentSpec]:
    return [
        *_default_specs(DEFAULT_PIPELINES),
        ExperimentSpec(
            label="rfm_features",
            cleaning_pipeline="standard",
            feature_builder="baseline",
        ),
        ExperimentSpec(
            label="behavioral_features",
            cleaning_pipeline="standard",
            feature_builder="standard",
        ),
        ExperimentSpec(
            label="standard_features",
            cleaning_pipeline="advanced",
            feature_builder="standard",
        ),
        ExperimentSpec(
            label="advanced_features",
            cleaning_pipeline="advanced",
            feature_builder="advanced",
        ),
        ExperimentSpec(
            label="standard_cleaning",
            cleaning_pipeline="standard",
            feature_builder="advanced",
        ),
        ExperimentSpec(
            label="advanced_cleaning",
            cleaning_pipeline="advanced",
            feature_builder="advanced",
        ),
        ExperimentSpec(
            label="only_rfm",
            cleaning_pipeline="advanced",
            feature_builder="rfm_only",
        ),
        ExperimentSpec(
            label="only_behavioral",
            cleaning_pipeline="advanced",
            feature_builder="behavioral_only",
        ),
        ExperimentSpec(
            label="rfm_behavioral",
            cleaning_pipeline="advanced",
            feature_builder="rfm_behavioral",
        ),
        ExperimentSpec(
            label="rfm_temporal",
            cleaning_pipeline="advanced",
            feature_builder="rfm_temporal",
        ),
        ExperimentSpec(
            label="full_features",
            cleaning_pipeline="advanced",
            feature_builder="full",
        ),
        ExperimentSpec(
            label="with_cancelled_filter",
            cleaning_pipeline="standard",
            feature_builder="standard",
        ),
        ExperimentSpec(
            label="without_cancelled_filter",
            cleaning_pipeline="no_cancelled_filter",
            feature_builder="standard",
        ),
        ExperimentSpec(
            label="with_invalid_filter",
            cleaning_pipeline="standard",
            feature_builder="standard",
        ),
        ExperimentSpec(
            label="without_invalid_filter",
            cleaning_pipeline="no_invalid_filter",
            feature_builder="standard",
        ),
    ]


def _save_target_distribution(
    customer_targets: pd.Series,
    train_ids: list,
    test_ids: list,
) -> pd.DataFrame:
    frames = {
        "full": customer_targets,
        "train": customer_targets.loc[train_ids],
        "test": customer_targets.loc[test_ids],
    }
    rows = []
    for split_name, split_target in frames.items():
        positives = int(split_target.sum())
        total = int(len(split_target))
        negatives = total - positives
        rows.append(
            {
                "split": split_name,
                "n_customers": total,
                "n_positive": positives,
                "n_negative": negatives,
                "positive_rate": round(positives / total, 4) if total else 0.0,
            }
        )
    target_df = pd.DataFrame(rows)
    target_df.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)
    logger.info("Target distribution saved -> outputs/target_distribution.csv")
    return target_df


def _save_best_config_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = metrics_df.copy()
    ranking_df["rank_roc_auc"] = ranking_df["roc_auc"].rank(
        ascending=False, method="dense"
    )
    ranking_df["rank_f1"] = ranking_df["f1"].rank(ascending=False, method="dense")
    ranking_df["rank_avg_prec"] = ranking_df["avg_prec"].rank(
        ascending=False, method="dense"
    )
    ranking_df["composite_score"] = (
        0.5 * ranking_df["roc_auc"]
        + 0.3 * ranking_df["f1"]
        + 0.2 * ranking_df["avg_prec"]
    ).round(4)
    ranking_df = ranking_df.sort_values(
        ["composite_score", "roc_auc", "f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking_df["overall_rank"] = ranking_df.index + 1
    ranking_df.to_csv(OUTPUT_DIR / "best_config_summary.csv", index=False)
    logger.info("Best config summary saved -> outputs/best_config_summary.csv")
    return ranking_df


def _save_cv_gap_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    gap_df = metrics_df.copy()
    if "cv_auc_mean" not in gap_df.columns:
        gap_df["cv_auc_mean"] = pd.NA
    gap_df["cv_test_gap"] = (gap_df["cv_auc_mean"] - gap_df["roc_auc"]).round(4)
    gap_df = gap_df.sort_values(
        ["cv_test_gap", "roc_auc"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    gap_df.to_csv(OUTPUT_DIR / "cv_test_gap_summary.csv", index=False)
    logger.info("CV/test gap summary saved -> outputs/cv_test_gap_summary.csv")
    return gap_df


EXPERIMENT_PROFILES = {
    "all": {
        "specs": _implemented_all_specs(),
        "models": MODELS,
        "description": "Run all implemented experiment configurations in one pass",
    },
    "exp1": {
        "specs": _default_specs(["baseline"]),
        "models": MODELS,
        "description": "Experiment 1: model comparison on the baseline pipeline",
    },
    "exp2": {
        "specs": _default_specs(["standard"]),
        "models": MODELS,
        "description": "Experiment 2: model comparison on the standard pipeline",
    },
    "exp3": {
        "specs": _default_specs(["advanced"]),
        "models": ["logistic_regression", "decision_tree", "xgboost"],
        "description": "Experiment 3: model comparison on the advanced pipeline",
    },
    "exp4": _single_model_profile(
        "Experiment 4: preprocessing sensitivity of logistic regression",
        "logistic_regression",
    ),
    "exp5": _single_model_profile(
        "Experiment 5: preprocessing sensitivity of decision tree",
        "decision_tree",
    ),
    "exp6": _single_model_profile(
        "Experiment 6: preprocessing sensitivity of xgboost",
        "xgboost",
    ),
    "exp7": {
        "specs": [
            ExperimentSpec(
                label="rfm_features",
                cleaning_pipeline="standard",
                feature_builder="baseline",
            ),
            ExperimentSpec(
                label="behavioral_features",
                cleaning_pipeline="standard",
                feature_builder="standard",
            ),
        ],
        "models": MODELS,
        "description": (
            "Experiment 7: RFM vs behavioral features under the same standard cleaning"
        ),
    },
    "exp8": {
        "specs": [
            ExperimentSpec(
                label="standard_features",
                cleaning_pipeline="advanced",
                feature_builder="standard",
            ),
            ExperimentSpec(
                label="advanced_features",
                cleaning_pipeline="advanced",
                feature_builder="advanced",
            ),
        ],
        "models": ["logistic_regression", "decision_tree", "xgboost"],
        "description": (
            "Experiment 8: contribution of temporal and country features "
            "under the same advanced cleaning"
        ),
    },
    "exp9": {
        "specs": [
            ExperimentSpec(
                label="only_rfm",
                cleaning_pipeline="advanced",
                feature_builder="rfm_only",
            ),
            ExperimentSpec(
                label="only_behavioral",
                cleaning_pipeline="advanced",
                feature_builder="behavioral_only",
            ),
            ExperimentSpec(
                label="rfm_behavioral",
                cleaning_pipeline="advanced",
                feature_builder="rfm_behavioral",
            ),
            ExperimentSpec(
                label="rfm_temporal",
                cleaning_pipeline="advanced",
                feature_builder="rfm_temporal",
            ),
            ExperimentSpec(
                label="full_features",
                cleaning_pipeline="advanced",
                feature_builder="full",
            ),
        ],
        "models": MODELS,
        "description": "Experiment 9: ablation of feature groups",
    },
    "exp10": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 10: compare feature importance between pipelines",
    },
    "exp11": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": ["decision_tree", "xgboost"],
        "description": "Experiment 11: compare SHAP explanations between pipelines",
    },
    "exp12": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 12: local interpretation with LIME",
    },
    "exp13": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 13: FI/SHAP agreement analysis",
    },
    "exp14": {
        "specs": [
            ExperimentSpec(
                label="with_cancelled_filter",
                cleaning_pipeline="standard",
                feature_builder="standard",
            ),
            ExperimentSpec(
                label="without_cancelled_filter",
                cleaning_pipeline="no_cancelled_filter",
                feature_builder="standard",
            ),
        ],
        "models": MODELS,
        "description": "Experiment 14: effect of removing cancelled invoices",
    },
    "exp15": {
        "specs": [
            ExperimentSpec(
                label="with_invalid_filter",
                cleaning_pipeline="standard",
                feature_builder="standard",
            ),
            ExperimentSpec(
                label="without_invalid_filter",
                cleaning_pipeline="no_invalid_filter",
                feature_builder="standard",
            ),
        ],
        "models": MODELS,
        "description": "Experiment 15: effect of filtering invalid quantity and price",
    },
    "exp16": {
        "specs": [
            ExperimentSpec(
                label="standard_cleaning",
                cleaning_pipeline="standard",
                feature_builder="advanced",
            ),
            ExperimentSpec(
                label="advanced_cleaning",
                cleaning_pipeline="advanced",
                feature_builder="advanced",
            ),
        ],
        "models": MODELS,
        "description": (
            "Experiment 16: impact of IQR outlier filtering under the same advanced features"
        ),
    },
    "exp17": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 17: cleaning-step impact analysis",
    },
    "exp18": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": ["logistic_regression", "xgboost"],
        "description": "Experiment 18: simple model vs complex model",
    },
    "exp19": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 19: decision tree as a compromise model",
    },
    "exp20": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 20: compare CV-AUC and test ROC-AUC",
    },
    "exp21": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 21: target distribution analysis",
    },
    "exp29": {
        "specs": _default_specs(DEFAULT_PIPELINES),
        "models": MODELS,
        "description": "Experiment 29: full ranking of configurations",
    },
}


def _eligible_customers(raw_df: pd.DataFrame) -> pd.Index:
    history = raw_df[(raw_df["customer"].notna()) & (raw_df["date"] < CUTOFF_DATE)]
    return pd.Index(history["customer"].drop_duplicates())


def run_experiment(
    run_shap: bool = True,
    run_lime: bool = True,
    run_cv: bool = True,
    models: list[str] | None = None,
    pipelines: list[str] | None = None,
    experiment_specs: list[ExperimentSpec] | None = None,
) -> list[ModelResult]:
    models = models or MODELS
    if experiment_specs is None:
        pipelines = pipelines or DEFAULT_PIPELINES
        experiment_specs = _default_specs(pipelines)

    logger.info("=" * 60)
    logger.info("Loading data...")
    raw_df = load_raw_data()
    print_eda_report(raw_df)

    customers = _eligible_customers(raw_df)
    customer_targets = build_target(raw_df).reindex(customers).fillna(0).astype(int)
    train_ids, test_ids = split_customer_ids(customers, customer_targets)
    target_df = _save_target_distribution(customer_targets, train_ids, test_ids)

    all_results: list[ModelResult] = []
    cleaning_reports: list[pd.DataFrame] = []

    for spec in experiment_specs:
        logger.info(
            "- Experiment: %s (cleaning=%s, features=%s)",
            spec.label,
            spec.cleaning_pipeline,
            spec.feature_builder,
        )

        raw_train = raw_df[raw_df["customer"].isin(train_ids)].copy()
        raw_test = raw_df[raw_df["customer"].isin(test_ids)].copy()

        clean_artifacts = fit_cleaning_pipeline(
            spec.cleaning_pipeline,
            raw_train[raw_train["date"] < CUTOFF_DATE].copy(),
        )
        clean_train, train_report = apply_cleaning_pipeline(
            raw_train,
            clean_artifacts,
            dataset_split="train",
        )
        clean_test, test_report = apply_cleaning_pipeline(
            raw_test,
            clean_artifacts,
            dataset_split="test",
        )
        train_report_df = train_report.to_dataframe()
        train_report_df["experiment"] = spec.label
        train_report_df["feature_builder"] = spec.feature_builder
        test_report_df = test_report.to_dataframe()
        test_report_df["experiment"] = spec.label
        test_report_df["feature_builder"] = spec.feature_builder
        cleaning_reports.extend([train_report_df, test_report_df])

        feature_artifacts = fit_feature_builder(
            spec.feature_builder,
            clean_train[clean_train["date"] < CUTOFF_DATE].copy(),
        )
        feat_train = apply_feature_builder(clean_train, feature_artifacts)
        feat_test = apply_feature_builder(clean_test, feature_artifacts)

        logger.info(
            "  Features -> train=%d customers, test=%d customers, n_features=%d",
            len(feat_train),
            len(feat_test),
            feat_train.shape[1] - 1,
        )

        X_tr = feat_train.drop(columns=[TARGET_COL])
        y_tr = feat_train[TARGET_COL]
        X_te = feat_test.drop(columns=[TARGET_COL]).reindex(columns=X_tr.columns)
        y_te = feat_test[TARGET_COL]

        for model_name in models:
            X_tr_m = X_tr.copy()
            X_te_m = X_te.copy()

            if model_name in _SCALE_FOR:
                X_tr_m, X_te_m, _ = scale_features(X_tr_m, X_te_m)

            train_medians = X_tr_m.median(numeric_only=True)
            X_tr_m = X_tr_m.fillna(train_medians)
            X_te_m = X_te_m.fillna(train_medians)

            result = train_and_evaluate(
                pipeline_name=spec.label,
                cleaning_pipeline=spec.cleaning_pipeline,
                feature_builder=spec.feature_builder,
                model_name=model_name,
                X_train=X_tr_m,
                X_test=X_te_m,
                y_train=y_tr,
                y_test=y_te,
                run_cv=run_cv,
            )
            all_results.append(result)

    cleaning_df = pd.concat(cleaning_reports, ignore_index=True)
    cleaning_df.to_csv(OUTPUT_DIR / "cleaning_steps.csv", index=False)
    logger.info("Cleaning steps saved -> outputs/cleaning_steps.csv")

    metrics_df = results_to_dataframe(all_results)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    logger.info("Metrics saved -> outputs/metrics_summary.csv")
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("\nTARGET DISTRIBUTION")
    print("=" * 60)
    print(target_df.to_string(index=False))

    logger.info("- Feature Importance...")
    fi_by_model: dict[str, list] = {model_name: [] for model_name in models}

    for result in all_results:
        fi = get_feature_importance(result.model, result.feature_names)
        plot_feature_importance(
            fi,
            title=f"FI: {result.pipeline_name} / {result.model_name}",
            save_path=PLOT_DIR / f"fi_{result.experiment_id}.png",
            top_n=15,
        )
        if result.model_name == "logistic_regression":
            plot_signed_coefficients(
                result.model,
                result.feature_names,
                title=f"Signed Coefficients: {result.pipeline_name}",
                save_path=PLOT_DIR / f"coef_signed_{result.experiment_id}.png",
                top_n=15,
            )
        fi_by_model[result.model_name].append((result.pipeline_name, fi))

    for model_name, fi_list in fi_by_model.items():
        compare_feature_importance(
            fi_list,
            save_path=PLOT_DIR / f"fi_compare_{model_name}.png",
            top_n=10,
        )

    shap_data: dict[str, list] = {model_name: [] for model_name in models}
    shap_lookup: dict[str, object] = {}

    if run_shap and SHAP_AVAILABLE:
        logger.info("- SHAP...")
        for result in all_results:
            X_sample = result.X_test.sample(min(200, len(result.X_test)), random_state=42)
            shap_values = compute_shap_values(result.model, X_sample)
            if shap_values is None:
                continue

            plot_shap_summary(
                shap_values,
                X_sample,
                title=f"SHAP: {result.pipeline_name} / {result.model_name}",
                save_path=SHAP_DIR / f"shap_{result.experiment_id}.png",
            )
            plot_shap_bar(
                shap_values,
                X_sample,
                title=f"SHAP Bar: {result.pipeline_name} / {result.model_name}",
                save_path=SHAP_DIR / f"shap_bar_{result.experiment_id}.png",
                max_display=15,
            )
            top_shap_features = (
                shap_mean_abs(shap_values, result.feature_names)["feature"].head(3).tolist()
            )
            for feature_name in top_shap_features:
                plot_shap_dependence(
                    shap_values,
                    X_sample,
                    feature_name=feature_name,
                    title=(
                        f"SHAP Dependence: {result.pipeline_name} / "
                        f"{result.model_name} / {feature_name}"
                    ),
                    save_path=SHAP_DIR / f"shap_dependence_{result.experiment_id}_{feature_name}.png",
                )
            sample_waterfall_idx = 0
            if len(result.y_test[result.y_test == 1]) > 0:
                sample_label = result.y_test[result.y_test == 1].index[0]
                if sample_label in X_sample.index:
                    sample_waterfall_idx = X_sample.index.get_loc(sample_label)
            plot_shap_waterfall(
                shap_values,
                X_sample,
                sample_idx=sample_waterfall_idx,
                title=f"SHAP Waterfall: {result.pipeline_name} / {result.model_name}",
                save_path=SHAP_DIR / f"shap_waterfall_{result.experiment_id}.png",
                max_display=12,
            )
            shap_data[result.model_name].append(
                (result.pipeline_name, shap_values, result.feature_names)
            )
            shap_lookup[result.experiment_id] = shap_values
    elif not SHAP_AVAILABLE:
        logger.warning("SHAP skipped because it is not installed.")

    for model_name, shap_list in shap_data.items():
        if len(shap_list) > 1:
            compare_shap_rankings(
                shap_list,
                save_path=SHAP_DIR / f"shap_compare_{model_name}.png",
                top_n=10,
            )

    if run_lime and LIME_AVAILABLE:
        logger.info("- LIME...")
        for result in all_results:
            pos_idx_list = result.y_test[result.y_test == 1].index.tolist()
            if not pos_idx_list:
                continue

            sample_pos = result.X_test.index.get_loc(pos_idx_list[0])
            explanation = explain_with_lime(
                model=result.model,
                X_train=result.X_train,
                X_test=result.X_test,
                sample_idx=sample_pos,
            )
            if explanation is None:
                continue

            plot_lime_explanation(
                explanation,
                title=f"LIME: {result.pipeline_name} / {result.model_name}",
                save_path=LIME_DIR / f"lime_{result.experiment_id}.png",
            )
            lime_to_dataframe(explanation).to_csv(
                LIME_DIR / f"lime_{result.experiment_id}.csv",
                index=False,
            )
    elif not LIME_AVAILABLE:
        logger.warning("LIME skipped because it is not installed.")

    interp_df = interpretability_summary(all_results)
    interp_df.to_csv(OUTPUT_DIR / "interpretability_summary.csv", index=False)
    logger.info("Interpretability summary saved -> outputs/interpretability_summary.csv")
    agreement_df = fi_shap_agreement_summary(all_results, shap_lookup)
    agreement_df.to_csv(OUTPUT_DIR / "fi_shap_agreement_summary.csv", index=False)
    logger.info("FI/SHAP agreement saved -> outputs/fi_shap_agreement_summary.csv")
    best_df = _save_best_config_summary(metrics_df)
    cv_gap_df = _save_cv_gap_summary(metrics_df)

    print("\n" + "=" * 60)
    print("INTERPRETABILITY SUMMARY")
    print("=" * 60)
    print(interp_df.to_string(index=False))
    if not agreement_df.empty:
        print("\nFI/SHAP AGREEMENT SUMMARY")
        print("=" * 60)
        print(agreement_df.to_string(index=False))
    print("\nBEST CONFIG SUMMARY")
    print("=" * 60)
    print(best_df.to_string(index=False))
    print("\nCV VS TEST SUMMARY")
    print("=" * 60)
    print(cv_gap_df.to_string(index=False))
    print("=" * 60)

    return all_results


def run_experiment_3(
    run_shap: bool = True,
    run_lime: bool = True,
    run_cv: bool = True,
) -> list[ModelResult]:
    """
    Experiment 3.

    Compare three models on the advanced preprocessing pipeline to assess
    whether aggressive cleaning with IQR-based outlier removal improves
    quality without removing useful signal.
    """
    logger.info("=" * 60)
    logger.info("Experiment 3: model comparison on advanced pipeline")
    logger.info(
        "Goal: evaluate aggressive cleaning impact, including IQR-based outlier removal."
    )
    logger.info(
        "Comparing: advanced + logistic_regression, decision_tree, xgboost"
    )

    results = run_experiment(
        run_shap=run_shap,
        run_lime=run_lime,
        run_cv=run_cv,
        models=["logistic_regression", "decision_tree", "xgboost"],
        experiment_specs=_default_specs(["advanced"]),
    )

    metrics_df = results_to_dataframe(results)
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)

    return results


def run_experiment_4(
    run_shap: bool = True,
    run_lime: bool = True,
    run_cv: bool = True,
) -> list[ModelResult]:
    """
    Experiment 4.

    Compare preprocessing pipelines for logistic regression.
    """
    logger.info("=" * 60)
    logger.info("Experiment 4: preprocessing sensitivity of logistic regression")

    results = run_experiment(
        run_shap=run_shap,
        run_lime=run_lime,
        run_cv=run_cv,
        models=["logistic_regression"],
        experiment_specs=EXPERIMENT_PROFILES["exp4"]["specs"],
    )

    metrics_df = results_to_dataframe(results)
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)

    return results


def run_experiment_8(
    run_shap: bool = True,
    run_lime: bool = True,
    run_cv: bool = True,
) -> list[ModelResult]:
    """
    Experiment 8.

    Compare standard vs advanced feature sets under the same advanced cleaning
    to isolate the contribution of temporal and country features.
    """
    logger.info("=" * 60)
    logger.info("Experiment 8: feature contribution under advanced cleaning")
    logger.info(
        "Comparing: standard features vs advanced features "
        "(standard + temporal + country)"
    )
    results = run_experiment(
        run_shap=run_shap,
        run_lime=run_lime,
        run_cv=run_cv,
        models=["logistic_regression", "decision_tree", "xgboost"],
        experiment_specs=EXPERIMENT_PROFILES["exp8"]["specs"],
    )
    metrics_df = results_to_dataframe(results)
    print("\n" + "=" * 60)
    print("EXPERIMENT 8 SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)
    return results


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preprocessing/model comparison experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_PROFILES.keys()),
        default="all",
        help="Predefined experiment profile to run. Default: all.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=DEFAULT_PIPELINES,
        help="Optional custom pipelines list. Builds same-name cleaning/features.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        help="Optional custom models list. Overrides --experiment models.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP explanations.",
    )
    parser.add_argument(
        "--no-lime",
        action="store_true",
        help="Skip LIME explanations.",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation.",
    )
    return parser


def main() -> list[ModelResult]:
    parser = _build_cli_parser()
    args = parser.parse_args()

    profile = EXPERIMENT_PROFILES[args.experiment]
    specs = profile["specs"]
    if args.pipelines:
        specs = _default_specs(args.pipelines)
    models = args.models or profile["models"]

    logger.info("=" * 60)
    logger.info("Selected experiment profile: %s", args.experiment)
    logger.info("Description: %s", profile["description"])
    logger.info(
        "Experiments: %s",
        ", ".join(
            f"{spec.label}[clean={spec.cleaning_pipeline},feat={spec.feature_builder}]"
            for spec in specs
        ),
    )
    logger.info("Models: %s", ", ".join(models))

    return run_experiment(
        run_shap=not args.no_shap,
        run_lime=not args.no_lime,
        run_cv=not args.no_cv,
        models=models,
        experiment_specs=specs,
    )


if __name__ == "__main__":
    main()
