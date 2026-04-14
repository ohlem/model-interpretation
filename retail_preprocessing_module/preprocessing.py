"""
preprocessing.py — Три пайплайна очистки данных.

Каждый пайплайн логирует, сколько строк/клиентов удалено на каждом шаге —
это ключевой артефакт для анализа влияния очистки на качество и интерпретацию.

Пайплайны:
  baseline  — минимальная очистка (только NaN в customer)
  standard  — + фильтр отмен, отрицательных цен/количества, топ-страны
  advanced  — + выброс по IQR, агрессивная нормализация
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ─── Отчёт о шагах очистки ────────────────────────────────────────────────────

@dataclass
class CleaningStep:
    name       : str
    rows_before: int
    rows_after : int
    customers_before: int
    customers_after : int
    description: str = ""

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def customers_removed(self) -> int:
        return self.customers_before - self.customers_after

    def __str__(self) -> str:
        return (
            f"  [{self.name}] {self.description}\n"
            f"    Строк:    {self.rows_before:>8,} → {self.rows_after:>8,}"
            f"  (удалено {self.rows_removed:>7,})\n"
            f"    Клиентов: {self.customers_before:>8,} → {self.customers_after:>8,}"
            f"  (удалено {self.customers_removed:>7,})"
        )


@dataclass
class CleaningReport:
    pipeline_name: str
    steps: List[CleaningStep] = field(default_factory=list)

    def add_step(self, df: pd.DataFrame, name: str, description: str,
                 df_before: pd.DataFrame) -> None:
        step = CleaningStep(
            name=name,
            rows_before=len(df_before),
            rows_after=len(df),
            customers_before=df_before["customer"].nunique(),
            customers_after=df["customer"].nunique(),
            description=description,
        )
        self.steps.append(step)
        logger.info(str(step))

    def summary(self) -> str:
        lines = [f"\n{'='*60}", f"Cleaning Report: {self.pipeline_name}", "="*60]
        for s in self.steps:
            lines.append(str(s))
        lines.append("="*60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "pipeline"          : self.pipeline_name,
                "step"              : s.name,
                "description"       : s.description,
                "rows_before"       : s.rows_before,
                "rows_after"        : s.rows_after,
                "rows_removed"      : s.rows_removed,
                "rows_removed_pct"  : round(s.rows_removed / s.rows_before * 100, 2),
                "customers_before"  : s.customers_before,
                "customers_after"   : s.customers_after,
                "customers_removed" : s.customers_removed,
            }
            for s in self.steps
        ])


# ─── Атомарные шаги очистки ───────────────────────────────────────────────────

def _drop_null_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Шаг 1: удаляем строки без CustomerID."""
    return df.dropna(subset=["customer"])


def _drop_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """Шаг 2: удаляем отменённые заказы (Invoice начинается с 'C')."""
    mask = ~df["invoice"].astype(str).str.upper().str.startswith("C")
    return df[mask]


def _drop_non_positive_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """Шаг 3: удаляем строки с нулевым или отрицательным количеством."""
    return df[df["quantity"] > 0]


def _drop_non_positive_price(df: pd.DataFrame) -> pd.DataFrame:
    """Шаг 4: удаляем строки с нулевой или отрицательной ценой."""
    return df[df["price"] > 0]


def _drop_test_stock_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Шаг 5: удаляем тестовые/служебные артикулы (POSTAGE, D, M, ...)."""
    test_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT"}
    mask = ~df["stock_code"].astype(str).str.upper().isin(test_codes)
    return df[mask]


def _drop_price_outliers_iqr(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Шаг 6 (advanced): удаляем выбросы по Price методом IQR×k."""
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (df["price"] >= q1 - k * iqr) & (df["price"] <= q3 + k * iqr)
    return df[mask]


def _drop_quantity_outliers_iqr(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Шаг 7 (advanced): удаляем выбросы по Quantity методом IQR×k."""
    q1, q3 = df["quantity"].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (df["quantity"] >= q1 - k * iqr) & (df["quantity"] <= q3 + k * iqr)
    return df[mask]


def _add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляем столбец выручки по строке."""
    df = df.copy()
    df["revenue"] = df["quantity"] * df["price"]
    return df


# ─── Пайплайны ────────────────────────────────────────────────────────────────

def pipeline_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Baseline: минимальная очистка.
    Только удаляем строки без CustomerID + добавляем revenue.
    Намеренно «грязный» датасет — для сравнения интерпретации.
    """
    report = CleaningReport(pipeline_name="baseline")

    before = df
    df = _drop_null_customer(df)
    report.add_step(df, "drop_null_customer",
                    "Удаление строк без CustomerID", before)

    df = _add_revenue(df)

    logger.info(report.summary())
    return df.reset_index(drop=True), report


def pipeline_standard(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Standard: полная базовая очистка.
    NaN customer → отмены → отрицательные qty/price → служебные коды.
    """
    report = CleaningReport(pipeline_name="standard")

    before = df
    df = _drop_null_customer(df)
    report.add_step(df, "drop_null_customer",
                    "Удаление строк без CustomerID", before)

    before = df
    df = _drop_cancelled(df)
    report.add_step(df, "drop_cancelled",
                    "Удаление отменённых заказов (Invoice ~ C...)", before)

    before = df
    df = _drop_non_positive_quantity(df)
    report.add_step(df, "drop_neg_quantity",
                    "Удаление строк с Quantity ≤ 0", before)

    before = df
    df = _drop_non_positive_price(df)
    report.add_step(df, "drop_neg_price",
                    "Удаление строк с Price ≤ 0", before)

    before = df
    df = _drop_test_stock_codes(df)
    report.add_step(df, "drop_test_codes",
                    "Удаление служебных StockCode (POST, D, M...)", before)

    df = _add_revenue(df)

    logger.info(report.summary())
    return df.reset_index(drop=True), report


def pipeline_advanced(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Advanced: полная очистка + удаление выбросов по IQR.
    """
    report = CleaningReport(pipeline_name="advanced")

    before = df
    df = _drop_null_customer(df)
    report.add_step(df, "drop_null_customer",
                    "Удаление строк без CustomerID", before)

    before = df
    df = _drop_cancelled(df)
    report.add_step(df, "drop_cancelled",
                    "Удаление отменённых заказов (Invoice ~ C...)", before)

    before = df
    df = _drop_non_positive_quantity(df)
    report.add_step(df, "drop_neg_quantity",
                    "Удаление строк с Quantity ≤ 0", before)

    before = df
    df = _drop_non_positive_price(df)
    report.add_step(df, "drop_neg_price",
                    "Удаление строк с Price ≤ 0", before)

    before = df
    df = _drop_test_stock_codes(df)
    report.add_step(df, "drop_test_codes",
                    "Удаление служебных StockCode (POST, D, M...)", before)

    before = df
    df = _drop_price_outliers_iqr(df, k=3.0)
    report.add_step(df, "iqr_price",
                    "Удаление выбросов Price (IQR × 3)", before)

    before = df
    df = _drop_quantity_outliers_iqr(df, k=3.0)
    report.add_step(df, "iqr_quantity",
                    "Удаление выбросов Quantity (IQR × 3)", before)

    df = _add_revenue(df)

    logger.info(report.summary())
    return df.reset_index(drop=True), report


# ─── Реестр пайплайнов ────────────────────────────────────────────────────────

PIPELINES = {
    "baseline" : pipeline_baseline,
    "standard" : pipeline_standard,
    "advanced" : pipeline_advanced,
}
