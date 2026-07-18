# CarValue AI — прогноз стоимости автомобиля

[![Live Demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://autopricepredictiongit-hse.streamlit.app)
[![CI](https://github.com/McChainiy/auto_price_prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/McChainiy/auto_price_prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)

**End-to-end ML-проект для оценки подержанных автомобилей на индийском рынке:** от исследования данных и feature engineering до сериализованного `scikit-learn Pipeline`, автоматических тестов и интерактивного веб-приложения.

### [Открыть live demo →](https://autopricepredictiongit-hse.streamlit.app)

> Цены и ошибки в проекте указаны в индийских рупиях — INR (`₹`).

## Результат

| Метрика | Значение на holdout-выборке |
|---|---:|
| **R²** | **0.887** |
| **MAE** | **₹127,522** |
| RMSE | ₹254,925 |
| Размер тестовой выборки | 1,000 автомобилей |

Feature engineering поднял R² с **0.600 у baseline Linear Regression до 0.887** у финального пайплайна. Это больший прирост, чем дала одна только смена регуляризации: главной силой решения стала работа с представлением данных.

## Почему проект интересен

- **Полный sklearn Pipeline.** Сырые признаки, масштабирование и Ridge Regression объединены в один воспроизводимый объект: одинаковая обработка используется при обучении и инференсе.
- **Парсинг реальных объявлений.** Кастомный `TransformerMixin` извлекает числа из строк вроде `19.7 kmpl`, `1248 CC`, `74 bhp` и `190 Nm at 2000 rpm`, включая перевод `kgm → Nm`.
- **Feature engineering вместо “магии модели”.** Добавлены квадрат года, логарифм пробега, отношение мощности к объёму двигателя, оценка расхода топлива, страна бренда и средняя цена модели.
- **Аккуратная работа с пропусками и категориями.** Медианная импутация дополняется missing-value флагом, OHE умеет обрабатывать неизвестные значения, а для незнакомой модели есть fallback к среднему target на train.
- **Защита от data leakage.** Mean target encoding модели автомобиля строится только по обучающей выборке и хранится внутри fitted-трансформера.
- **Интерпретируемость.** Коэффициенты Ridge визуализируются после стандартизации признаков, поэтому их можно сравнивать между собой.
- **Продуктовый интерфейс.** Streamlit-приложение поддерживает одиночную оценку, пакетный прогноз из CSV, расчёт метрик, быстрый EDA, φk-корреляции и выгрузку результата.
- **Engineering-обвязка.** Версии зависимостей зафиксированы, метаданные модели версионируются, а GitHub Actions проверяет компиляцию и регрессионные тесты.

## Как устроено решение

```mermaid
flowchart LR
    A[Сырое объявление] --> B[Regex-парсинг]
    B --> C[Импутация + missing flag]
    C --> D[Feature engineering]
    D --> E[OHE + target encoding]
    E --> F[StandardScaler]
    F --> G[Ridge Regression]
    G --> H[Цена в INR]
```

Финальный пайплайн превращает 12 сырых полей в 31 модельный признак:

```text
raw listing → MyTransormer → StandardScaler → Ridge(alpha=100) → prediction
```

Название `MyTransormer` содержит историческую опечатку: оно намеренно сохранено для обратной совместимости с обученным pickle-артефактом. В приложении используется отдельный безопасный mapping legacy-класса при загрузке.

## Эксперименты

| Модель | R² | MAE, INR | Что изменилось |
|---|---:|---:|---|
| Linear Regression | 0.600 | 220,873 | Только числовые признаки |
| Lasso | 0.600 | 220,872 | L1-регуляризация |
| Ridge + OHE | 0.781 | 168,628 | Добавлены категориальные признаки |
| **Final Ridge Pipeline** | **0.887** | **127,522** | Feature engineering + scaling + Ridge |

В исследовательском ноутбуке также реализованы и разобраны:

- ручной расчёт R² и adjusted R²;
- собственная L0-регрессия;
- GridSearchCV с 10-fold cross-validation для Lasso, ElasticNet и Ridge;
- Pearson, Spearman, Kendall и φk-корреляции;
- бизнес-метрика с разными штрафами за переоценку и недооценку;
- анализ влияния выбросов и визуализация весов моделей.

Полный ход экспериментов находится в [Jupyter Notebook](AI_HW1_Regression_with_inference_pro_pt1.ipynb).

## Возможности приложения

1. **Оценить один автомобиль** через форму с понятными единицами измерения.
2. **Загрузить CSV** и получить прогноз для каждой строки.
3. **Проверить качество** — при наличии `selling_price` автоматически считаются R², MAE и RMSE.
4. **Исследовать данные** — распределения, пропуски, дубликаты и φk-корреляции.
5. **Объяснить модель** — увидеть самые сильные положительные и отрицательные коэффициенты.
6. **Скачать результат** в готовом CSV с прогнозами и абсолютной ошибкой.

## Стек

- **ML:** scikit-learn, NumPy, pandas
- **Модель:** Ridge Regression, StandardScaler, OneHotEncoder, custom TransformerMixin
- **Аналитика:** φk, Plotly; в исследовательском ноутбуке — seaborn, matplotlib, ydata-profiling
- **Приложение:** Streamlit
- **Quality:** unittest, GitHub Actions

## Структура репозитория

```text
.
├── .github/workflows/ci.yml       # CI: compile + regression tests
├── .streamlit/config.toml         # тема и настройки приложения
├── models/
│   ├── metadata.json              # версия, схема и метрики модели
│   └── ridge_new_features.pkl     # fitted sklearn Pipeline
├── src/
│   └── model.py                   # трансформер, загрузка и инференс
├── tests/
│   └── test_model.py              # численная регрессия и edge cases
├── app.py                         # Streamlit UI
├── test.csv                       # пример и holdout-выборка
├── requirements.txt               # минимальные runtime-зависимости
└── AI_HW1_Regression_...ipynb     # EDA, эксперименты и обучение
```

## Локальный запуск

Требуется Python 3.12.

```bash
git clone https://github.com/McChainiy/auto_price_prediction.git
cd auto_price_prediction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

streamlit run app.py
```

Модель уже лежит в `models/`, поэтому для запуска приложения повторное обучение не требуется.

## Тесты

```bash
python -m unittest discover -s tests -v
```

Тесты фиксируют контрольные прогнозы, воспроизводят опубликованные R²/MAE, проверяют схему входных данных, парсинг физических величин, неизвестного производителя и ограничение цены неотрицательной областью.

## Формат входного CSV

Обязательные поля:

```text
name, year, km_driven, fuel, seller_type, transmission, owner,
mileage, engine, max_power, torque, seats
```

Колонка `selling_price` необязательна. Если она присутствует, приложение использует её только для расчёта метрик. Готовый пример — [test.csv](test.csv).

## Ограничения

- Модель обучена на исторических данных индийского рынка и не должна напрямую переноситься на другие страны и валюты.
- Она не видит состояние кузова, историю ДТП, регион, комплектацию и динамику рынка.
- Линейная модель может выдавать отрицательные значения на объектах вне обучающего распределения; UI ограничивает такие прогнозы нулём, а исследовательские метрики рассчитаны по исходным значениям.
- Pickle безопасно загружать только из доверенного источника. Приложение использует исключительно артефакт из этого репозитория и никогда не принимает модели от пользователя.

## Данные

Исходные train/test-таблицы взяты из учебного набора [HSE MLDS](https://github.com/Murcha1990/MLDS_ML_2022/tree/main/Hometasks/HT1). В репозитории сохранена тестовая выборка для демонстрации и проверки воспроизводимости.

---

Если проект оказался полезен, можно поставить ⭐ — это помогает ему быть заметнее.
