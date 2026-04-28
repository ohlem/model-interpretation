"""
preprocessing.py — пайплайны очистки данных с явным разделением на fit/apply.

Ключевая идея модуля — контроль утечки данных:
все пороги, зависящие от распределения данных, оцениваются только на train-истории,
а затем без изменений применяются и к train, и к test.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from .config import CUTOFF_DATE

logger = logging.getLogger(__name__)


# --- Отчёт по очистке -------------------------------------------------------


@dataclass
class CleaningStep:
    name: str
    rows_before: int
    rows_after: int
    customers_before: int
    customers_after: int
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
            f"    Rows:      {self.rows_before:>8,} -> {self.rows_after:>8,}"
            f"  (removed {self.rows_removed:>7,})\n"
            f"    Customers: {self.customers_before:>8,} -> {self.customers_after:>8,}"
            f"  (removed {self.customers_removed:>7,})"
        )


@dataclass
class CleaningReport:
    pipeline_name: str
    dataset_split: str = "full"
    steps: list[CleaningStep] = field(default_factory=list)

    def add_step(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        df_before: pd.DataFrame,
    ) -> None:
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
        lines = [
            f"\n{'=' * 60}",
            f"Cleaning Report: {self.pipeline_name} [{self.dataset_split}]",
            "=" * 60,
        ]
        for step in self.steps:
            lines.append(str(step))
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pipeline": self.pipeline_name,
                    "dataset_split": self.dataset_split,
                    "step": step.name,
                    "description": step.description,
                    "rows_before": step.rows_before,
                    "rows_after": step.rows_after,
                    "rows_removed": step.rows_removed,
                    "rows_removed_pct": round(
                        step.rows_removed / step.rows_before * 100, 2
                    ),
                    "customers_before": step.customers_before,
                    "customers_after": step.customers_after,
                    "customers_removed": step.customers_removed,
                }
                for step in self.steps
            ]
        )


@dataclass
class CleaningArtifacts:
    pipeline_name: str
    price_bounds: tuple[float, float] | None = None
    quantity_bounds: tuple[float, float] | None = None
    apply_cancelled: bool = False
    apply_non_positive_quantity: bool = False
    apply_non_positive_price: bool = False
    apply_test_codes: bool = False
    apply_iqr_price: bool = False
    apply_iqr_quantity: bool = False


# --- Базовые шаги очистки ---------------------------------------------------


def _drop_null_customer(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["customer"])


def _drop_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["invoice"].astype(str).str.upper().str.startswith("C")
    return df[mask]


def _drop_non_positive_quantity(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["quantity"] > 0]


def _drop_non_positive_price(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["price"] > 0]


def _drop_test_stock_codes(df: pd.DataFrame) -> pd.DataFrame:
    test_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT"}
    mask = ~df["stock_code"].astype(str).str.upper().isin(test_codes)
    return df[mask]


def _iqr_bounds(series: pd.Series, k: float = 3.0) -> tuple[float, float]:
    clean = series.dropna()
    if clean.empty:
        return float("-inf"), float("inf")

    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _apply_bounds(
    df: pd.DataFrame,
    column: str,
    bounds: tuple[float, float] | None,
) -> pd.DataFrame:
    if bounds is None:
        return df
    lower, upper = bounds
    mask = (df[column] >= lower) & (df[column] <= upper)
    return df[mask]


def _add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue"] = df["quantity"] * df["price"]
    return df


def _pipeline_options(pipeline_name: str) -> dict[str, bool]:
    # Для каждого сценария задаём, какие шаги очистки нужно включить.
    options = {
        "apply_cancelled": False,
        "apply_non_positive_quantity": False,
        "apply_non_positive_price": False,
        "apply_test_codes": False,
        "apply_iqr_price": False,
        "apply_iqr_quantity": False,
    }

    if pipeline_name in {
        "standard",
        "advanced",
        "no_cancelled_filter",
        "no_invalid_filter",
    }:
        options.update(
            {
                "apply_cancelled": True,
                "apply_non_positive_quantity": True,
                "apply_non_positive_price": True,
                "apply_test_codes": True,
            }
        )

    if pipeline_name == "advanced":
        options.update({"apply_iqr_price": True, "apply_iqr_quantity": True})

    if pipeline_name == "no_cancelled_filter":
        options["apply_cancelled"] = False

    if pipeline_name == "no_invalid_filter":
        options["apply_non_positive_quantity"] = False
        options["apply_non_positive_price"] = False

    return options


def _apply_optional_steps(
    df: pd.DataFrame,
    report: CleaningReport,
    artifacts: CleaningArtifacts,
) -> pd.DataFrame:
    # Эти шаги включаются или отключаются в зависимости от выбранного пайплайна.
    if artifacts.apply_cancelled:
        before = df
        df = _drop_cancelled(df)
        report.add_step(df, "drop_cancelled", "Удаление отменённых инвойсов", before)

    if artifacts.apply_non_positive_quantity:
        before = df
        df = _drop_non_positive_quantity(df)
        report.add_step(
            df,
            "drop_neg_quantity",
            "Удаление строк с Quantity <= 0",
            before,
        )

    if artifacts.apply_non_positive_price:
        before = df
        df = _drop_non_positive_price(df)
        report.add_step(df, "drop_neg_price", "Удаление строк с Price <= 0", before)

    if artifacts.apply_test_codes:
        before = df
        df = _drop_test_stock_codes(df)
        report.add_step(
            df,
            "drop_test_codes",
            "Удаление служебных и тестовых stock_code",
            before,
        )

    return df


# --- API fit/apply ----------------------------------------------------------


def fit_cleaning_pipeline(
    pipeline_name: str,
    df_fit: pd.DataFrame,
) -> CleaningArtifacts:
    """
    Оценить параметры очистки, зависящие от данных, только на train-истории.
    """
    options = _pipeline_options(pipeline_name)
    artifacts = CleaningArtifacts(pipeline_name=pipeline_name, **options)

    if not (artifacts.apply_iqr_price or artifacts.apply_iqr_quantity):
        return artifacts

    fitted = _drop_null_customer(df_fit)
    fitted = _apply_optional_steps(
        fitted,
        CleaningReport(pipeline_name=pipeline_name, dataset_split="fit"),
        artifacts,
    )

    if artifacts.apply_iqr_price:
        # Границы для price оцениваются на train и потом переиспользуются без refit.
        artifacts.price_bounds = _iqr_bounds(fitted["price"], k=3.0)
        fitted = _apply_bounds(fitted, "price", artifacts.price_bounds)
    if artifacts.apply_iqr_quantity:
        # Quantity оцениваем уже после базовой очистки и, при необходимости, после price-фильтра.
        artifacts.quantity_bounds = _iqr_bounds(fitted["quantity"], k=3.0)

    return artifacts


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    artifacts: CleaningArtifacts,
    dataset_split: str = "full",
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Применить заранее настроенный пайплайн очистки к любому сплиту данных.
    """
    report = CleaningReport(
        pipeline_name=artifacts.pipeline_name,
        dataset_split=dataset_split,
    )

    before = df
    df = _drop_null_customer(df)
    report.add_step(
        df,
        "drop_null_customer",
        "Удаление строк без CustomerID",
        before,
    )

    df = _apply_optional_steps(df, report, artifacts)

    if artifacts.apply_iqr_price:
        before = df
        df = _apply_bounds(df, "price", artifacts.price_bounds)
        report.add_step(
            df,
            "iqr_price",
            "Применение IQR-границ для Price, оценённых на train",
            before,
        )

    if artifacts.apply_iqr_quantity:
        before = df
        df = _apply_bounds(df, "quantity", artifacts.quantity_bounds)
        report.add_step(
            df,
            "iqr_quantity",
            "Применение IQR-границ для Quantity, оценённых на train",
            before,
        )

    # Revenue добавляется после завершения фильтрации, чтобы признак считался
    # уже на согласованной очищенной выборке.
    df = _add_revenue(df)

    logger.info(report.summary())
    return df.reset_index(drop=True), report


# --- Обёртки для обратной совместимости ------------------------------------


def pipeline_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    artifacts = fit_cleaning_pipeline(
        "baseline",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_cleaning_pipeline(df, artifacts)


def pipeline_standard(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    artifacts = fit_cleaning_pipeline(
        "standard",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_cleaning_pipeline(df, artifacts)


def pipeline_advanced(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    artifacts = fit_cleaning_pipeline(
        "advanced",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_cleaning_pipeline(df, artifacts)


PIPELINES = {
    "baseline": pipeline_baseline,
    "standard": pipeline_standard,
    "advanced": pipeline_advanced,
    "no_cancelled_filter": pipeline_standard,
    "no_invalid_filter": pipeline_standard,
}
