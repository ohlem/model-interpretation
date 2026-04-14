"""
config.py — Центральная конфигурация эксперимента.
Все пути, даты и гиперпараметры меняются только здесь.
"""

from pathlib import Path
import pandas as pd

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR   = OUTPUT_DIR / "plots"
SHAP_DIR   = PLOT_DIR / "shap"
LIME_DIR   = PLOT_DIR / "lime"

for d in [DATA_DIR, OUTPUT_DIR, PLOT_DIR, SHAP_DIR, LIME_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Файл данных ─────────────────────────────────────────────────────────────
# Положите online_retail_II.csv в папку data/
RAW_FILE_CSV  = DATA_DIR / "online_retail_II.csv"

# ─── Временные границы ───────────────────────────────────────────────────────
CUTOFF_DATE      = pd.Timestamp("2011-07-01")   # дата среза (train / target)
TARGET_END_DATE  = pd.Timestamp("2011-10-01")   # конец окна предсказания (3 мес.)
TRAIN_START_DATE = pd.Timestamp("2009-12-01")   # начало истории

# ─── Целевая переменная ───────────────────────────────────────────────────────
TARGET_COL = "target_repurchase"

# ─── Колонки исходного датасета ───────────────────────────────────────────────
RAW_COLS = {
    "invoice"     : "Invoice",
    "stock_code"  : "StockCode",
    "description" : "Description",
    "quantity"    : "Quantity",
    "date"        : "InvoiceDate",
    "price"       : "Price",
    "customer"    : "Customer ID",
    "country"     : "Country",
}

# ─── Параметры обучения ───────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2   # доля клиентов в hold-out внутри train-периода

CV_FOLDS = 5

# ─── Гиперпараметры моделей ───────────────────────────────────────────────────
MODEL_PARAMS = {
    "logistic_regression": {
        "C"            : 1.0,
        "max_iter"     : 1000,
        "random_state" : RANDOM_STATE,
        "class_weight" : "balanced",
    },
    "decision_tree": {
        "max_depth"    : 5,
        "min_samples_leaf": 20,
        "random_state" : RANDOM_STATE,
        "class_weight" : "balanced",
    },
    "xgboost": {
        "n_estimators"     : 300,
        "max_depth"        : 4,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "eval_metric"      : "logloss",
        "random_state"     : RANDOM_STATE,
        "use_label_encoder": False,
    },
}

# ─── Названия пайплайнов предобработки ───────────────────────────────────────
PIPELINE_NAMES = {
    "baseline" : "Baseline (минимальная очистка)",
    "standard" : "Standard (очистка + базовые признаки)",
    "advanced" : "Advanced (полная очистка + расширенные признаки)",
}
