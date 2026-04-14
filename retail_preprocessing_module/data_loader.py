"""
data_loader.py — Загрузка и первичный анализ сырых данных.
"""
import pandas as pd
import logging
from retail_preprocessing_module.config import  RAW_FILE_CSV, RAW_COLS

logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    """Загружает данные из CSV."""
    
    if not RAW_FILE_CSV.exists():
        raise FileNotFoundError(
            f"Файл данных не найден по пути: {RAW_FILE_CSV}\n"
            f"Пожалуйста, положите файл 'online_retail_II.csv' в папку data."
        )

    logger.info(f"Загрузка CSV: {RAW_FILE_CSV}")

    try:
        df = pd.read_csv(RAW_FILE_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Ошибка utf-8, пробую ISO-8859-1...")
        df = pd.read_csv(RAW_FILE_CSV, encoding="ISO-8859-1")

    # Переименование колонок в системные имена
    inv_map = {v: k for k, v in RAW_COLS.items()}
    df = df.rename(columns=inv_map)
    
    # Приведение типов
    df["date"] = pd.to_datetime(df["date"])
    df["customer"] = df["customer"].astype(object) # ID как объект/строка
    
    return df

def print_eda_report(df: pd.DataFrame):
    """Базовый статистический отчет."""
    print("\n--- ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ ---")
    print(f"Строк: {len(df):,}")
    print(f"Колонок: {list(df.columns)}")
    print(f"Период: с {df['date'].min()} по {df['date'].max()}")
    print(f"Уникальных клиентов: {df['customer'].nunique()}")
    print(f"Пропуски в Customer ID: {df['customer'].isna().sum()}")
    print(f"Отрицательные цены: {(df['price'] < 0).sum()}")
    print(f"Отрицательные кол-ва: {(df['quantity'] < 0).sum()}")
    print("-" * 30 + "\n")