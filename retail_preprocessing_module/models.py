"""
models.py — Обучение и оценка моделей.

Три модели × три пайплайна признаков = 9 экспериментов.
Метрики: ROC-AUC, F1 (threshold=0.5), Precision, Recall.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.metrics        import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost не установлен. Используйте: pip install xgboost")

from .config import MODEL_PARAMS, RANDOM_STATE, TARGET_COL

logger = logging.getLogger(__name__)


# ─── Контейнер результатов ────────────────────────────────────────────────────

@dataclass
class ModelResult:
    pipeline_name : str
    model_name    : str
    model         : object
    feature_names : list
    X_train       : pd.DataFrame
    X_test        : pd.DataFrame
    y_train       : pd.Series
    y_test        : pd.Series
    y_pred_proba  : np.ndarray  # вероятности класса 1 на test
    metrics       : Dict[str, float] = field(default_factory=dict)
    cv_auc        : Optional[float]  = None

    @property
    def experiment_id(self) -> str:
        return f"{self.pipeline_name}__{self.model_name}"


# ─── Фабрика моделей ──────────────────────────────────────────────────────────

def _make_model(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(**MODEL_PARAMS["logistic_regression"])

    if model_name == "decision_tree":
        return DecisionTreeClassifier(**MODEL_PARAMS["decision_tree"])

    if model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost не установлен.")
        return XGBClassifier(**MODEL_PARAMS["xgboost"])

    raise ValueError(f"Неизвестная модель: {model_name}")


# ─── Вычисление метрик ────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc"  : round(roc_auc_score(y_true, y_pred_proba), 4),
        "avg_prec" : round(average_precision_score(y_true, y_pred_proba), 4),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


# ─── Разбивка на train/test ───────────────────────────────────────────────────

def split_customer_features(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Стратифицированное разбиение клиентов 80/20.
    Признаки = все колонки кроме target, target = TARGET_COL.
    """
    from sklearn.model_selection import train_test_split

    X = features_df.drop(columns=[TARGET_COL])
    y = features_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split: train=%d (pos=%.1f%%), test=%d (pos=%.1f%%)",
        len(y_train), y_train.mean() * 100,
        len(y_test),  y_test.mean()  * 100,
    )
    return X_train, X_test, y_train, y_test


# ─── Основная функция обучения ────────────────────────────────────────────────

def train_and_evaluate(
    pipeline_name  : str,
    model_name     : str,
    X_train        : pd.DataFrame,
    X_test         : pd.DataFrame,
    y_train        : pd.Series,
    y_test         : pd.Series,
    run_cv         : bool = True,
) -> ModelResult:
    """
    Обучаем модель, считаем метрики на test, опционально — CV-AUC на train.
    """
    logger.info("▶ Обучение: pipeline=%s  model=%s", pipeline_name, model_name)

    model = _make_model(model_name)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics      = _compute_metrics(y_test, y_pred_proba)

    cv_auc = None
    if run_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            _make_model(model_name), X_train, y_train,
            scoring="roc_auc", cv=cv, n_jobs=-1,
        )
        cv_auc = round(scores.mean(), 4)
        logger.info("  CV-AUC: %.4f ± %.4f", scores.mean(), scores.std())

    logger.info(
        "  Test → AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
        metrics["roc_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"],
    )

    return ModelResult(
        pipeline_name = pipeline_name,
        model_name    = model_name,
        model         = model,
        feature_names = list(X_train.columns),
        X_train       = X_train,
        X_test        = X_test,
        y_train       = y_train,
        y_test        = y_test,
        y_pred_proba  = y_pred_proba,
        metrics       = metrics,
        cv_auc        = cv_auc,
    )


# ─── Сводная таблица результатов ──────────────────────────────────────────────

def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "pipeline"   : r.pipeline_name,
            "model"      : r.model_name,
            "n_features" : len(r.feature_names),
            **r.metrics,
        }
        if r.cv_auc is not None:
            row["cv_auc_mean"] = r.cv_auc
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "pipeline"])


MODELS = ["logistic_regression", "decision_tree", "xgboost"]
