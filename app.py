"""Streamlit portfolio app for used-car price estimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model import (
    REQUIRED_FEATURES,
    load_model_metadata,
    load_pipeline,
    predict_prices,
    prepare_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "test.csv"
TARGET_COLUMN = "selling_price"

FEATURE_LABELS = {
    "year": "Год выпуска²",
    "mileage": "Экономичность",
    "engine": "Объём двигателя",
    "max_power": "Максимальная мощность",
    "torque": "Крутящий момент",
    "seats": "Количество мест",
    "skipped_flag": "Были пропуски",
    "max_torque_rpm": "Обороты максимального момента",
    "enigne_over_power": "Мощность / объём двигателя",
    "fuel_spent": "Расчётный расход топлива",
    "model_avg_price": "Средняя цена модели",
    "log_km_driven": "Логарифм пробега",
}


st.set_page_config(
    page_title="CarValue · Оценка автомобиля",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
        .app-header {
            border-bottom: 1px solid #2b3038;
            padding: .4rem 0 1.25rem;
            margin-bottom: 1.25rem;
        }
        .app-header h1 {font-size: 2rem; margin: 0 0 .45rem; line-height: 1.15;}
        .app-header p {color: #b6bbc4; max-width: 780px; margin: 0;}
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .75rem;
            margin-bottom: 1rem;
        }
        .metric-item {
            background: #171a1f;
            border: 1px solid #2b3038;
            border-radius: 6px;
            min-height: 92px;
            padding: .85rem 1rem;
        }
        .metric-label {color: #b6bbc4; font-size: .78rem; font-weight: 600;}
        .metric-value {font-size: 1.55rem; line-height: 1.2; margin-top: .3rem;}
        .metric-note {color: #42d392; font-size: .76rem; margin-top: .25rem;}
        .feature-card {
            border: 1px solid #2b3038;
            border-radius: 6px;
            padding: 1rem 1.1rem;
            min-height: 128px;
            background: #171a1f;
        }
        .feature-card h4 {margin: .1rem 0 .45rem;}
        .feature-card p {color: #b6bbc4; font-size: .92rem; margin: 0;}
        .eyebrow {color: #42d392; font-size: .82rem; font-weight: 700; text-transform: uppercase;}
        div.stButton > button, div.stDownloadButton > button {border-radius: 6px;}
        @media (max-width: 640px) {
            .block-container {padding-top: 1rem;}
            .metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem;}
            .metric-item {min-height: 88px; padding: .7rem .8rem;}
            .metric-value {font-size: 1.35rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Загружаем модель…")
def get_model():
    return load_pipeline()


@st.cache_data
def get_model_metadata() -> dict[str, object]:
    return load_model_metadata()


@st.cache_data
def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def format_price(value: float) -> str:
    return f"{max(float(value), 0):,.0f} INR"


def options_for(transformer, column: str) -> list[str]:
    return sorted(map(str, transformer.cat_columns_unique[column]))


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    categorical_prefixes = {
        "fuel_": "Топливо",
        "seller_type_": "Продавец",
        "transmission_": "Коробка передач",
        "owner_": "Владелец",
        "country_": "Страна бренда",
    }
    for prefix, label in categorical_prefixes.items():
        if name.startswith(prefix):
            return f"{label}: {name.removeprefix(prefix)}"
    return name


try:
    model = get_model()
    model_metadata = get_model_metadata()
    test_metrics = model_metadata["test_set"]
    model_version = model_metadata["version"]
    feature_count = model_metadata["engineered_feature_count"]
except Exception as error:
    st.error("Не удалось проверить и загрузить модель или её метаданные.")
    st.exception(error)
    st.stop()

transformer = model.named_steps["transformer"]

with st.sidebar:
    st.markdown("### CarValue")
    st.caption("Оценка стоимости подержанных автомобилей")
    st.divider()
    st.markdown("**Модель**")
    st.caption(f"Ridge Regression · v{model_version}")
    st.caption(f"Признаков после обработки · {feature_count}")
    st.caption("Целевая валюта · INR")
    st.caption(f"Тестовый R² · {test_metrics['r2']:.3f}")
    st.divider()
    st.link_button(
        "Открыть репозиторий ↗",
        "https://github.com/McChainiy/auto_price_prediction",
        width="stretch",
    )
    st.caption("Демо предназначено для исследования модели, а не для финансовой оценки.")

st.markdown(
    """
    <header class="app-header">
        <h1>CarValue</h1>
        <p>
            Оценка стоимости подержанных автомобилей на индийском рынке с объяснением
            факторов прогноза и поддержкой пакетной обработки.
        </p>
    </header>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="metric-grid" aria-label="Метрики модели">
        <div class="metric-item">
            <div class="metric-label">R² на тесте</div>
            <div class="metric-value">{test_metrics['r2']:.3f}</div>
            <div class="metric-note">+0.287 к baseline</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{format_price(test_metrics['mae'])}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Признаков после обработки</div>
            <div class="metric-value">{feature_count}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Holdout-выборка</div>
            <div class="metric-value">{test_metrics['rows']:,}</div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab, analysis_tab, interpretation_tab, overview_tab = st.tabs(
    [
        "Оценить автомобиль",
        "Пакетный прогноз",
        "Анализ данных",
        "Интерпретация",
        "О проекте",
    ]
)

with overview_tab:
    st.subheader("От сырой строки объявления до прогноза")
    st.write(
        "Модель принимает характеристики в том же виде, в котором они встречаются "
        "в объявлениях, и воспроизводимо проводит всю обработку внутри одного "
        "`scikit-learn Pipeline`."
    )

    cards = st.columns(4)
    card_content = [
        (
            "01 · Парсинг",
            "Регулярные выражения извлекают мощность, объём, расход, момент и RPM из смешанных строк.",
        ),
        (
            "02 · Feature engineering",
            "Полиномиальные, логарифмические и interaction-признаки раскрывают нелинейные зависимости.",
        ),
        (
            "03 · Кодирование",
            "OHE с поддержкой неизвестных категорий и train-only mean target encoding модели авто.",
        ),
        (
            "04 · Оценка",
            "StandardScaler и Ridge Regression дают устойчивый прогноз с контролем регуляризации.",
        ),
    ]
    for column, (title, description) in zip(cards, card_content):
        column.markdown(
            f'<div class="feature-card"><div class="eyebrow">{title}</div>'
            f"<p>{description}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Что особенно интересно")
    st.markdown(
        """
        - Медианная импутация дополняется отдельным флагом пропуска.
        - Для неизвестной модели автомобиля используется средняя цена train-выборки.
        - Пробег логарифмируется, а год выпуска и взаимодействия признаков моделируют нелинейность.
        - Интерфейс поддерживает одиночный прогноз, CSV-пакеты, диагностику качества и объяснение коэффициентов.
        """
    )

with single_tab:
    st.subheader("Параметры автомобиля")
    st.caption("Диапазоны формы ориентированы на данные, на которых обучалась модель.")

    names = options_for(transformer, "name")
    fuels = options_for(transformer, "fuel")
    sellers = options_for(transformer, "seller_type")
    transmissions = options_for(transformer, "transmission")
    owners = options_for(transformer, "owner")
    seat_options = sorted({int(float(value)) for value in options_for(transformer, "seats")})

    with st.form("single_prediction"):
        left, right = st.columns(2)
        with left:
            name = st.selectbox(
                "Марка и модель",
                names,
                index=default_index(names, "Maruti Swift"),
            )
            year = st.number_input("Год выпуска", 1994, 2020, 2017, 1)
            km_driven = st.number_input("Пробег, км", 0, 1_500_000, 50_000, 5_000)
            fuel = st.selectbox("Тип топлива", fuels, index=default_index(fuels, "Petrol"))
            transmission = st.selectbox(
                "Коробка передач",
                transmissions,
                index=default_index(transmissions, "Manual"),
            )
            owner = st.selectbox(
                "История владения",
                owners,
                index=default_index(owners, "First Owner"),
            )
        with right:
            seller_type = st.selectbox(
                "Тип продавца",
                sellers,
                index=default_index(sellers, "Individual"),
            )
            mileage = st.number_input("Экономичность, kmpl", 1.0, 50.0, 19.0, 0.5)
            engine = st.number_input("Объём двигателя, CC", 500, 5_000, 1_200, 50)
            max_power = st.number_input("Максимальная мощность, bhp", 20.0, 500.0, 85.0, 5.0)
            torque = st.number_input("Крутящий момент, Nm", 20.0, 1_000.0, 150.0, 10.0)
            torque_rpm = st.number_input("Обороты максимального момента, rpm", 500, 8_000, 3_000, 100)
            seats = st.selectbox("Количество мест", seat_options, index=default_index(list(map(str, seat_options)), "5"))

        submitted = st.form_submit_button(
            "Рассчитать стоимость",
            type="primary",
            width="stretch",
        )

    if submitted:
        input_frame = pd.DataFrame(
            [
                {
                    "name": name,
                    "year": year,
                    "km_driven": km_driven,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                    "mileage": f"{mileage} kmpl",
                    "engine": f"{engine} CC",
                    "max_power": f"{max_power} bhp",
                    "torque": f"{torque} Nm at {torque_rpm} rpm",
                    "seats": seats,
                }
            ]
        )
        try:
            raw_prediction = predict_prices(model, input_frame)[0]
            prediction = max(raw_prediction, 0)
            st.success(f"Оценочная стоимость: **{format_price(prediction)}**")
            if raw_prediction < 0:
                st.warning(
                    "Линейная модель вышла за допустимую область; в интерфейсе "
                    "результат ограничен нулём."
                )
            st.caption(
                "Это исследовательский прогноз для индийского рынка. Он не учитывает "
                "состояние кузова, регион, комплектацию и текущую конъюнктуру."
            )
        except Exception as error:
            st.error(f"Не удалось построить прогноз: {error}")

with batch_tab:
    st.subheader("Прогноз для CSV-файла")
    st.write(
        "Загрузите таблицу с характеристиками автомобилей. Если в ней есть "
        "`selling_price`, приложение дополнительно рассчитает метрики качества."
    )
    download_column, upload_column = st.columns([1, 2])
    with download_column:
        st.download_button(
            "Скачать пример CSV",
            data=SAMPLE_DATA_PATH.read_bytes(),
            file_name="cars_test.csv",
            mime="text/csv",
            width="stretch",
        )
    with upload_column:
        batch_file = st.file_uploader(
            "CSV для пакетной оценки",
            type="csv",
            key="batch_file",
            label_visibility="collapsed",
        )

    with st.expander("Ожидаемая схема данных"):
        st.code(", ".join(REQUIRED_FEATURES), language="text")

    if batch_file is not None:
        try:
            batch = read_csv(batch_file)
            features = prepare_features(batch)
            raw_predictions = predict_prices(model, features)
            display_predictions = np.maximum(raw_predictions, 0)

            result = batch.copy()
            result["predicted_price_inr"] = display_predictions.round().astype(int)

            if TARGET_COLUMN in batch.columns:
                target = pd.to_numeric(batch[TARGET_COLUMN], errors="raise")
                metric_row = st.columns(3)
                metric_row[0].metric("R²", f"{r2_score(target, raw_predictions):.3f}")
                metric_row[1].metric(
                    "MAE",
                    format_price(mean_absolute_error(target, raw_predictions)),
                )
                metric_row[2].metric(
                    "RMSE",
                    format_price(mean_squared_error(target, raw_predictions) ** 0.5),
                )
                result["absolute_error_inr"] = np.abs(target - raw_predictions).round().astype(int)

            st.dataframe(result, width="stretch", hide_index=True)
            st.download_button(
                "Скачать прогнозы",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="car_price_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as error:
            st.error(f"Не удалось обработать файл: {error}")

with analysis_tab:
    st.subheader("Быстрый EDA")
    st.caption("Загрузите CSV, чтобы изучить распределения, пропуски и зависимости.")
    analysis_file = st.file_uploader("CSV для анализа", type="csv", key="analysis_file")

    if analysis_file is not None:
        try:
            analysis = read_csv(analysis_file)
            index_columns = [
                column
                for column in analysis.columns
                if str(column).startswith("Unnamed")
            ]
            analysis = analysis.drop(columns=index_columns)
            stats = st.columns(4)
            stats[0].metric("Строк", f"{len(analysis):,}")
            stats[1].metric("Признаков", analysis.shape[1])
            stats[2].metric("Пропусков", f"{int(analysis.isna().sum().sum()):,}")
            stats[3].metric("Дубликатов", f"{int(analysis.duplicated().sum()):,}")
            st.dataframe(analysis.head(100), width="stretch", hide_index=True)

            numeric_columns = analysis.select_dtypes(include=np.number).columns.tolist()
            if numeric_columns:
                selected_feature = st.selectbox("Распределение признака", numeric_columns)
                histogram = px.histogram(
                    analysis,
                    x=selected_feature,
                    nbins=40,
                    marginal="box",
                    color_discrete_sequence=["#3ecf8e"],
                    template="plotly_dark",
                )
                histogram.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(histogram, width="stretch")

            if st.button("Рассчитать φk-корреляции", width="stretch"):
                with st.spinner("Строим матрицу зависимостей…"):
                    import phik  # noqa: F401 - registers the pandas accessor

                    correlation_data = analysis.drop(columns=["name"], errors="ignore").head(5_000)
                    interval_columns = correlation_data.select_dtypes(include=np.number).columns.tolist()
                    correlation = correlation_data.phik_matrix(interval_cols=interval_columns)
                    heatmap = px.imshow(
                        correlation,
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        text_auto=".2f",
                        aspect="auto",
                        template="plotly_dark",
                    )
                    heatmap.update_layout(height=650, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(heatmap, width="stretch")
        except Exception as error:
            st.error(f"Не удалось проанализировать файл: {error}")

with interpretation_tab:
    st.subheader("Что влияет на прогноз")
    st.write(
        "Перед Ridge Regression признаки стандартизируются, поэтому абсолютные "
        "значения коэффициентов можно использовать для сравнительной интерпретации."
    )

    coefficients = pd.DataFrame(
        {
            "feature": transformer.feature_names_,
            "coefficient": model.named_steps["model"].coef_,
        }
    )
    coefficients["label"] = coefficients["feature"].map(feature_label)
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    top_n = st.slider("Сколько признаков показать", 8, len(coefficients), 15)
    top_coefficients = coefficients.nlargest(top_n, "absolute_coefficient").sort_values("coefficient")

    coefficient_chart = px.bar(
        top_coefficients,
        x="coefficient",
        y="label",
        orientation="h",
        color="coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"coefficient": "Коэффициент", "label": "Признак"},
        template="plotly_dark",
    )
    coefficient_chart.update_layout(
        height=max(480, top_n * 34),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(coefficient_chart, width="stretch")
    st.caption(
        "Знак показывает направление связи при прочих равных, но не доказывает "
        "причинность. Mean target encoding закономерно делает среднюю цену модели "
        "одним из наиболее сильных факторов."
    )

st.divider()
st.caption(
    "CarValue · scikit-learn + Streamlit · Цены и ошибки указаны в INR."
)
