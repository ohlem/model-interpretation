"""
features.py — Инженерия признаков на уровне клиента.

Функции строят три набора признаков:
  build_features_baseline  → только RFM (3 признака)
  build_features_standard  → RFM + поведенческие (12 признаков)
  build_features_advanced  → RFM + поведенческие + временны́е + страна (20+ признаков)

Все функции принимают очищенный DataFrame (строки = транзакции)
и возвращают DataFrame (строки = клиенты) с колонкой target_repurchase.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CUTOFF_DATE, TARGET_COL, TARGET_END_DATE

logger = logging.getLogger(__name__)


# ─── Целевая переменная ───────────────────────────────────────────────────────

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    target_repurchase = 1, если клиент совершил покупку
    в окне [CUTOFF_DATE, TARGET_END_DATE).
    """
    future = df[(df["date"] >= CUTOFF_DATE) & (df["date"] < TARGET_END_DATE)]
    return future.groupby("customer")["invoice"].nunique().gt(0).astype(int)


# ─── Вспомогательные агрегаты ────────────────────────────────────────────────

def _rfm(df_train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Базовые RFM-признаки."""
    snap = cutoff
    grp  = df_train.groupby("customer")

    recency   = (snap - grp["date"].max()).dt.days.rename("rfm_recency")
    frequency = grp["invoice"].nunique().rename("rfm_frequency")
    monetary  = grp["revenue"].sum().rename("rfm_monetary")

    return pd.concat([recency, frequency, monetary], axis=1)


def _behavioral(df_train: pd.DataFrame) -> pd.DataFrame:
    """Поведенческие признаки клиента."""
    grp = df_train.groupby("customer")

    avg_order_value   = (df_train.groupby(["customer", "invoice"])["revenue"]
                         .sum()
                         .groupby("customer")
                         .mean()
                         .rename("beh_avg_order_value"))

    unique_products   = grp["stock_code"].nunique().rename("beh_unique_products")
    unique_countries  = grp["country"].nunique().rename("beh_unique_countries")
    avg_qty_per_line  = grp["quantity"].mean().rename("beh_avg_qty_per_line")
    avg_price         = grp["price"].mean().rename("beh_avg_price")

    # Доля строк с количеством = 1 (мелкий ритейл vs оптовик)
    single_unit_ratio = (
        (df_train["quantity"] == 1)
        .groupby(df_train["customer"])
        .mean()
        .rename("beh_single_unit_ratio")
    )

    return pd.concat(
        [avg_order_value, unique_products, unique_countries,
         avg_qty_per_line, avg_price, single_unit_ratio],
        axis=1,
    )


def _temporal(df_train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Временны́е признаки клиента."""
    grp = df_train.groupby("customer")

    # Активный период (дни между первой и последней покупкой)
    tenure = (grp["date"].max() - grp["date"].min()).dt.days.rename("tmp_tenure_days")

    # Средний интервал между заказами
    def avg_interval(dates: pd.Series) -> float:
        sorted_d = dates.sort_values()
        if len(sorted_d) < 2:
            return np.nan
        return sorted_d.diff().dt.days.dropna().mean()

    avg_gap = (df_train.groupby("customer")["date"]
               .apply(avg_interval)
               .rename("tmp_avg_gap_days"))

    # Тренд активности: количество заказов в последние 90 дней / всего
    last_90d = (df_train[df_train["date"] >= cutoff - pd.Timedelta(days=90)]
                .groupby("customer")["invoice"]
                .nunique()
                .rename("tmp_orders_last_90d"))

    # Любимый день недели (мода)
    fav_dow = (df_train.assign(dow=df_train["date"].dt.dayofweek)
               .groupby("customer")["dow"]
               .agg(lambda x: x.mode().iat[0])
               .rename("tmp_fav_dow"))

    return pd.concat([tenure, avg_gap, last_90d, fav_dow], axis=1)


def _country_feature(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Топ-5 стран → dummy-колонки, остальные → 'Other'.
    Для логистической регрессии это важно — иначе страна неинтерпретируема.
    """
    top_countries = (
        df_train.groupby("country")["customer"]
        .nunique()
        .nlargest(5)
        .index.tolist()
    )

    def dominant_country(countries: pd.Series) -> str:
        mode = countries.mode()
        c = mode.iat[0] if len(mode) else "Other"
        return c if c in top_countries else "Other"

    dom_country = (df_train.groupby("customer")["country"]
                   .apply(dominant_country)
                   .rename("country_dominant"))

    dummies = pd.get_dummies(dom_country, prefix="ctry").astype(int)
    return dummies


# ─── Публичные строители признаков ────────────────────────────────────────────

def build_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline: только RFM.
    Намеренно простой набор — для демонстрации минимума.
    """
    df_train = df[df["date"] < CUTOFF_DATE].copy()

    rfm    = _rfm(df_train, CUTOFF_DATE)
    target = build_target(df)

    result = rfm.copy()
    result[TARGET_COL] = target.reindex(result.index).fillna(0).astype(int)

    logger.info("Baseline features: %d клиентов, %d признаков (+target)",
                len(result), result.shape[1] - 1)
    return result.dropna(subset=["rfm_recency"])


def build_features_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard: RFM + поведенческие признаки.
    """
    df_train = df[df["date"] < CUTOFF_DATE].copy()

    rfm  = _rfm(df_train, CUTOFF_DATE)
    beh  = _behavioral(df_train)

    result = rfm.join(beh, how="left")

    target = build_target(df)
    result[TARGET_COL] = target.reindex(result.index).fillna(0).astype(int)

    logger.info("Standard features: %d клиентов, %d признаков (+target)",
                len(result), result.shape[1] - 1)
    return result.dropna(subset=["rfm_recency"])


def build_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced: RFM + поведенческие + временны́е + страна.
    """
    df_train = df[df["date"] < CUTOFF_DATE].copy()

    rfm  = _rfm(df_train, CUTOFF_DATE)
    beh  = _behavioral(df_train)
    tmp  = _temporal(df_train, CUTOFF_DATE)
    ctry = _country_feature(df_train)

    result = rfm.join(beh, how="left").join(tmp, how="left").join(ctry, how="left")

    # Заполняем tmp_orders_last_90d нулями (клиент не покупал в период)
    result["tmp_orders_last_90d"] = result["tmp_orders_last_90d"].fillna(0)

    target = build_target(df)
    result[TARGET_COL] = target.reindex(result.index).fillna(0).astype(int)

    logger.info("Advanced features: %d клиентов, %d признаков (+target)",
                len(result), result.shape[1] - 1)
    return result.dropna(subset=["rfm_recency"])


# ─── Масштабирование (только для линейных моделей) ────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    StandardScaler, fit на train, transform на test.
    Бинарные столбцы (0/1) не масштабируем.
    """
    exclude_cols = exclude_cols or []
    binary_cols  = [c for c in X_train.columns
                    if set(X_train[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    skip         = set(exclude_cols) | set(binary_cols)
    scale_cols   = [c for c in X_train.columns if c not in skip]

    scaler     = StandardScaler()
    X_tr_sc    = X_train.copy()
    X_te_sc    = X_test.copy()

    X_tr_sc[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_te_sc[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_tr_sc, X_te_sc, scaler


# ─── Реестр строителей признаков ─────────────────────────────────────────────

FEATURE_BUILDERS = {
    "baseline" : build_features_baseline,
    "standard" : build_features_standard,
    "advanced" : build_features_advanced,
}
