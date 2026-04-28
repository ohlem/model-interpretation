"""
features.py - Customer-level feature engineering with explicit fit/apply stages.

Any data-dependent feature configuration is fitted on the train history only
and then reused for both train and test splits.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import CUTOFF_DATE, TARGET_COL, TARGET_END_DATE

logger = logging.getLogger(__name__)


@dataclass
class FeatureArtifacts:
    builder_name: str
    top_countries: list[str] = field(default_factory=list)


# --- Target -----------------------------------------------------------------


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    target_repurchase = 1 if the customer bought in [CUTOFF_DATE, TARGET_END_DATE).
    """
    future = df[(df["date"] >= CUTOFF_DATE) & (df["date"] < TARGET_END_DATE)]
    return future.groupby("customer")["invoice"].nunique().gt(0).astype(int)


# --- Aggregates --------------------------------------------------------------


def _rfm(df_train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    grp = df_train.groupby("customer")

    recency = (cutoff - grp["date"].max()).dt.days.rename("rfm_recency")
    frequency = grp["invoice"].nunique().rename("rfm_frequency")
    monetary = grp["revenue"].sum().rename("rfm_monetary")

    return pd.concat([recency, frequency, monetary], axis=1)


def _behavioral(df_train: pd.DataFrame) -> pd.DataFrame:
    grp = df_train.groupby("customer")

    avg_order_value = (
        df_train.groupby(["customer", "invoice"])["revenue"]
        .sum()
        .groupby("customer")
        .mean()
        .rename("beh_avg_order_value")
    )

    unique_products = grp["stock_code"].nunique().rename("beh_unique_products")
    unique_countries = grp["country"].nunique().rename("beh_unique_countries")
    avg_qty_per_line = grp["quantity"].mean().rename("beh_avg_qty_per_line")
    avg_price = grp["price"].mean().rename("beh_avg_price")

    single_unit_ratio = (
        (df_train["quantity"] == 1)
        .groupby(df_train["customer"])
        .mean()
        .rename("beh_single_unit_ratio")
    )

    return pd.concat(
        [
            avg_order_value,
            unique_products,
            unique_countries,
            avg_qty_per_line,
            avg_price,
            single_unit_ratio,
        ],
        axis=1,
    )


def _temporal(df_train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    grp = df_train.groupby("customer")

    tenure = (grp["date"].max() - grp["date"].min()).dt.days.rename("tmp_tenure_days")

    def avg_interval(dates: pd.Series) -> float:
        sorted_dates = dates.sort_values()
        if len(sorted_dates) < 2:
            return np.nan
        return sorted_dates.diff().dt.days.dropna().mean()

    avg_gap = (
        df_train.groupby("customer")["date"]
        .apply(avg_interval)
        .rename("tmp_avg_gap_days")
    )

    last_90d = (
        df_train[df_train["date"] >= cutoff - pd.Timedelta(days=90)]
        .groupby("customer")["invoice"]
        .nunique()
        .rename("tmp_orders_last_90d")
    )

    fav_dow = (
        df_train.assign(dow=df_train["date"].dt.dayofweek)
        .groupby("customer")["dow"]
        .agg(lambda x: x.mode().iat[0])
        .rename("tmp_fav_dow")
    )

    return pd.concat([tenure, avg_gap, last_90d, fav_dow], axis=1)


def _fit_top_countries(df_train: pd.DataFrame, top_n: int = 5) -> list[str]:
    return (
        df_train.groupby("country")["customer"]
        .nunique()
        .nlargest(top_n)
        .index.tolist()
    )


def _country_feature(
    df_train: pd.DataFrame,
    top_countries: list[str],
) -> pd.DataFrame:
    """
    Map dominant country to a fixed train-fitted one-hot space.
    """

    def dominant_country(countries: pd.Series) -> str:
        mode = countries.mode()
        country = mode.iat[0] if len(mode) else "Other"
        return country if country in top_countries else "Other"

    dominant = (
        df_train.groupby("customer")["country"]
        .apply(dominant_country)
        .rename("country_dominant")
    )

    dummies = pd.get_dummies(dominant, prefix="ctry").astype(int)
    expected_cols = [f"ctry_{country}" for country in top_countries] + ["ctry_Other"]
    return dummies.reindex(columns=expected_cols, fill_value=0)


# --- Fit/apply API ----------------------------------------------------------


def fit_feature_builder(
    builder_name: str,
    df_fit: pd.DataFrame,
) -> FeatureArtifacts:
    artifacts = FeatureArtifacts(builder_name=builder_name)

    if builder_name in {"advanced", "full"}:
        artifacts.top_countries = _fit_top_countries(df_fit)

    return artifacts


def apply_feature_builder(
    df: pd.DataFrame,
    artifacts: FeatureArtifacts,
) -> pd.DataFrame:
    """
    Build customer features using artifacts fitted on the train history only.
    """
    df_train = df[df["date"] < CUTOFF_DATE].copy()

    rfm = _rfm(df_train, CUTOFF_DATE)
    behavioral = _behavioral(df_train)
    temporal = _temporal(df_train, CUTOFF_DATE)
    country = _country_feature(df_train, artifacts.top_countries)

    builder_frames = {
        "baseline": [rfm],
        "standard": [rfm, behavioral],
        "advanced": [rfm, behavioral, temporal, country],
        "rfm_only": [rfm],
        "behavioral_only": [behavioral],
        "rfm_behavioral": [rfm, behavioral],
        "rfm_temporal": [rfm, temporal],
        "full": [rfm, behavioral, temporal, country],
    }
    if artifacts.builder_name not in builder_frames:
        raise ValueError(f"Unknown feature builder: {artifacts.builder_name}")

    result = builder_frames[artifacts.builder_name][0].copy()
    for frame in builder_frames[artifacts.builder_name][1:]:
        result = result.join(frame, how="left")

    if artifacts.builder_name in {"advanced", "rfm_temporal", "full"}:
        if "tmp_orders_last_90d" in result.columns:
            result["tmp_orders_last_90d"] = result["tmp_orders_last_90d"].fillna(0)

    target = build_target(df)
    result[TARGET_COL] = target.reindex(result.index).fillna(0).astype(int)

    logger.info(
        "%s features: %d customers, %d features (+target)",
        artifacts.builder_name.capitalize(),
        len(result),
        result.shape[1] - 1,
    )

    # Для наборов без RFM-признаков колонка rfm_recency не существует,
    # поэтому фильтрацию делаем только там, где этот признак действительно есть.
    if "rfm_recency" in result.columns:
        return result.dropna(subset=["rfm_recency"])
    return result


# --- Backward-compatible wrappers ------------------------------------------


def build_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "baseline",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_standard(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "standard",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "advanced",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_rfm_only(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "rfm_only",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_behavioral_only(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "behavioral_only",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_rfm_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "rfm_behavioral",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_rfm_temporal(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "rfm_temporal",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


def build_features_full(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = fit_feature_builder(
        "full",
        df[df["date"] < CUTOFF_DATE].copy(),
    )
    return apply_feature_builder(df, artifacts)


# --- Scaling ----------------------------------------------------------------


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    StandardScaler fit on train, transform on test.
    Binary columns are kept as-is.
    """
    exclude_cols = exclude_cols or []
    binary_cols = [
        col
        for col in X_train.columns
        if set(X_train[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]
    skip = set(exclude_cols) | set(binary_cols)
    scale_cols = [col for col in X_train.columns if col not in skip]

    scaler = StandardScaler()
    X_tr_sc = X_train.copy()
    X_te_sc = X_test.copy()

    X_tr_sc[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_te_sc[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_tr_sc, X_te_sc, scaler


FEATURE_BUILDERS = {
    "baseline": build_features_baseline,
    "standard": build_features_standard,
    "advanced": build_features_advanced,
    "rfm_only": build_features_rfm_only,
    "behavioral_only": build_features_behavioral_only,
    "rfm_behavioral": build_features_rfm_behavioral,
    "rfm_temporal": build_features_rfm_temporal,
    "full": build_features_full,
}
