# Модуль предобработки данных с оценкой влияния на интерпретируемость
## ВКР: Online Retail II → прогноз повторной покупки

---

## Структура проекта

```
retail_preprocessing_module/
├── config.py            # Константы: даты, пути, гиперпараметры
├── data_loader.py       # Загрузка данных + EDA-отчёт
├── preprocessing.py     # 3 пайплайна очистки (baseline / standard / advanced)
├── features.py          # Инженерия признаков: RFM + поведенческие + временны́е
├── models.py            # Обучение: LogReg / DT / XGBoost + метрики
├── interpretability.py  # SHAP, LIME, Feature Importance
├── experiment.py        # Главный оркестратор — запускает все 9 экспериментов
├── report.py            # Итоговые графики для ВКР
├── requirements.txt
└── data/                # ← положите сюда online_retail_II.xlsx или .csv
```

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Положить данные
mkdir data
cp ~/Downloads/online_retail_II.xlsx data/

# 3. Запустить полный эксперимент
python experiment.py

# 4. Сгенерировать итоговые графики
python report.py
```

Все результаты появятся в папке `outputs/`.

---

## Что делают три пайплайна

| Пайплайн  | Шаги очистки                                        | Признаков |
|-----------|-----------------------------------------------------|-----------|
| baseline  | Только удаление NaN в Customer ID                   | 3 (RFM)   |
| standard  | + отмены, отриц. qty/price, служебные коды           | ~12       |
| advanced  | + выбросы по IQR (Price, Quantity)                  | ~20       |

---

## Эксперимент: 9 комбинаций

```
3 пайплайна × 3 модели = 9 экспериментов

Модели: LogisticRegression | DecisionTree | XGBoost
```

Для каждой комбинации вычисляются:
- **Метрики**: ROC-AUC, F1, Precision, Recall, CV-AUC
- **Feature Importance**: abs(coef) / gain
- **SHAP**: TreeExplainer / LinearExplainer + summary plot
- **LIME**: объяснение для одного клиента

---

## Ключевые выходные файлы

| Файл | Описание |
|------|----------|
| `outputs/metrics_summary.csv` | Все метрики всех 9 экспериментов |
| `outputs/cleaning_steps.csv` | Сколько строк/клиентов удалил каждый шаг |
| `outputs/interpretability_summary.csv` | Топ-признаки по пайплайнам |
| `outputs/plots/metrics_comparison.png` | AUC/F1 по пайплайнам |
| `outputs/plots/cleaning_impact.png` | Влияние очистки на данные |
| `outputs/plots/fi_compare_*.png` | Тепловые карты FI по пайплайнам |
| `outputs/plots/shap/shap_compare_*.png` | Тепловые карты SHAP |
| `outputs/plots/lime/lime_*.png` | LIME объяснения |

---

## Даты эксперимента

```
Данные:        2009-12-01 — 2011-12-09
Дата среза:    2011-07-01  (train / target split)
Target-окно:   2011-07-01 — 2011-10-01 (3 месяца)
Target = 1:    клиент совершил хотя бы одну покупку в target-окне
```

---

## Настройка

Все параметры — в `config.py`:
- `CUTOFF_DATE`, `TARGET_END_DATE` — временны́е окна
- `MODEL_PARAMS` — гиперпараметры моделей
- `RANDOM_STATE` — воспроизводимость

Запуск только части экспериментов:

```python
from experiment import run_experiment

# Только baseline + XGBoost, без LIME
run_experiment(
    pipelines = ["baseline", "standard"],
    models    = ["xgboost"],
    run_lime  = False,
)
```
