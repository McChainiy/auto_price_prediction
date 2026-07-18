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
    "year": "Year squared",
    "mileage": "Fuel efficiency",
    "engine": "Engine displacement",
    "max_power": "Maximum power",
    "torque": "Torque",
    "seats": "Seat count",
    "skipped_flag": "Had missing values",
    "max_torque_rpm": "Peak torque RPM",
    "enigne_over_power": "Power / engine displacement",
    "fuel_spent": "Fuel-use proxy",
    "model_avg_price": "Model average price",
    "log_km_driven": "Log distance driven",
}


st.set_page_config(
    page_title="CarValue · Used Car Valuation",
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


@st.cache_resource(show_spinner="Loading model...")
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
        "fuel_": "Fuel",
        "seller_type_": "Seller type",
        "transmission_": "Transmission",
        "owner_": "Ownership",
        "country_": "Manufacturer country",
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
    st.error("The model or its metadata could not be verified and loaded.")
    st.exception(error)
    st.stop()

transformer = model.named_steps["transformer"]

with st.sidebar:
    st.markdown("### CarValue")
    st.caption("Used-car price estimation")
    st.divider()
    st.markdown("**Model**")
    st.caption(f"Ridge Regression · v{model_version}")
    st.caption(f"Engineered features · {feature_count}")
    st.caption("Target currency · INR")
    st.caption(f"Holdout R² · {test_metrics['r2']:.3f}")
    st.divider()
    st.link_button(
        "Open repository ↗",
        "https://github.com/McChainiy/auto_price_prediction",
        width="stretch",
    )
    st.caption("This demo is for model exploration, not financial valuation.")

st.markdown(
    """
    <header class="app-header">
        <h1>CarValue</h1>
        <p>
            Used-car price estimation for the Indian market, with prediction-factor
            analysis and batch processing.
        </p>
    </header>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="metric-grid" aria-label="Model metrics">
        <div class="metric-item">
            <div class="metric-label">Holdout R²</div>
            <div class="metric-value">{test_metrics['r2']:.3f}</div>
            <div class="metric-note">+0.287 vs. baseline</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{format_price(test_metrics['mae'])}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Engineered features</div>
            <div class="metric-value">{feature_count}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Holdout samples</div>
            <div class="metric-value">{test_metrics['rows']:,}</div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab, analysis_tab, interpretation_tab, overview_tab = st.tabs(
    [
        "Estimate a car",
        "Batch prediction",
        "Data analysis",
        "Interpretation",
        "About",
    ]
)

with overview_tab:
    st.subheader("From a raw listing to a price estimate")
    st.write(
        "The model accepts vehicle attributes in the same format used by real "
        "listings and applies every processing step reproducibly inside a single "
        "`scikit-learn Pipeline`."
    )

    cards = st.columns(4)
    card_content = [
        (
            "01 · Parsing",
            "Regular expressions extract power, displacement, mileage, torque, and RPM from mixed strings.",
        ),
        (
            "02 · Feature engineering",
            "Polynomial, logarithmic, and interaction features expose nonlinear relationships.",
        ),
        (
            "03 · Encoding",
            "OHE handles unseen categories, while model-level mean target encoding is learned on train only.",
        ),
        (
            "04 · Estimation",
            "StandardScaler and Ridge Regression provide stable estimates with controlled regularization.",
        ),
    ]
    for column, (title, description) in zip(cards, card_content):
        column.markdown(
            f'<div class="feature-card"><div class="eyebrow">{title}</div>'
            f"<p>{description}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Engineering highlights")
    st.markdown(
        """
        - Median imputation is paired with an explicit missing-value indicator.
        - Unseen vehicle models fall back to the training-set target mean.
        - Log distance, squared year, and interaction features model nonlinear effects.
        - The UI supports single estimates, CSV batches, quality diagnostics, and coefficient analysis.
        """
    )

with single_tab:
    st.subheader("Vehicle details")
    st.caption("Input ranges reflect the data distribution used to train the model.")

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
                "Make and model",
                names,
                index=default_index(names, "Maruti Swift"),
            )
            year = st.number_input("Model year", 1994, 2020, 2017, 1)
            km_driven = st.number_input("Distance driven, km", 0, 1_500_000, 50_000, 5_000)
            fuel = st.selectbox("Fuel type", fuels, index=default_index(fuels, "Petrol"))
            transmission = st.selectbox(
                "Transmission",
                transmissions,
                index=default_index(transmissions, "Manual"),
            )
            owner = st.selectbox(
                "Ownership history",
                owners,
                index=default_index(owners, "First Owner"),
            )
        with right:
            seller_type = st.selectbox(
                "Seller type",
                sellers,
                index=default_index(sellers, "Individual"),
            )
            mileage = st.number_input("Fuel efficiency, kmpl", 1.0, 50.0, 19.0, 0.5)
            engine = st.number_input("Engine displacement, CC", 500, 5_000, 1_200, 50)
            max_power = st.number_input("Maximum power, bhp", 20.0, 500.0, 85.0, 5.0)
            torque = st.number_input("Torque, Nm", 20.0, 1_000.0, 150.0, 10.0)
            torque_rpm = st.number_input("Peak torque, rpm", 500, 8_000, 3_000, 100)
            seats = st.selectbox("Seats", seat_options, index=default_index(list(map(str, seat_options)), "5"))

        submitted = st.form_submit_button(
            "Estimate price",
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
            st.success(f"Estimated price: **{format_price(prediction)}**")
            if raw_prediction < 0:
                st.warning(
                    "The linear model predicted a value outside the valid price "
                    "domain, so the displayed result was clipped to zero."
                )
            st.caption(
                "This is an experimental estimate for the Indian market. It does not "
                "account for vehicle condition, region, trim, or current market dynamics."
            )
        except Exception as error:
            st.error(f"Could not generate a prediction: {error}")

with batch_tab:
    st.subheader("CSV batch prediction")
    st.write(
        "Upload a table of vehicle attributes. When `selling_price` is present, "
        "the application also calculates evaluation metrics."
    )
    download_column, upload_column = st.columns([1, 2])
    with download_column:
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_DATA_PATH.read_bytes(),
            file_name="cars_test.csv",
            mime="text/csv",
            width="stretch",
        )
    with upload_column:
        batch_file = st.file_uploader(
            "CSV file for batch prediction",
            type="csv",
            key="batch_file",
            label_visibility="collapsed",
        )

    with st.expander("Expected data schema"):
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
                "Download predictions",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="car_price_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as error:
            st.error(f"Could not process the file: {error}")

with analysis_tab:
    st.subheader("Quick exploratory analysis")
    st.caption("Upload a CSV file to inspect distributions, missing values, and relationships.")
    analysis_file = st.file_uploader("CSV file for analysis", type="csv", key="analysis_file")

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
            stats[0].metric("Rows", f"{len(analysis):,}")
            stats[1].metric("Features", analysis.shape[1])
            stats[2].metric("Missing values", f"{int(analysis.isna().sum().sum()):,}")
            stats[3].metric("Duplicates", f"{int(analysis.duplicated().sum()):,}")
            st.dataframe(analysis.head(100), width="stretch", hide_index=True)

            numeric_columns = analysis.select_dtypes(include=np.number).columns.tolist()
            if numeric_columns:
                selected_feature = st.selectbox("Feature distribution", numeric_columns)
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

            if st.button("Calculate phi-k correlations", width="stretch"):
                with st.spinner("Building the relationship matrix..."):
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
            st.error(f"Could not analyze the file: {error}")

with interpretation_tab:
    st.subheader("What drives the estimate")
    st.write(
        "Features are standardized before Ridge Regression, so coefficient "
        "magnitudes can be compared for relative interpretation."
    )

    coefficients = pd.DataFrame(
        {
            "feature": transformer.feature_names_,
            "coefficient": model.named_steps["model"].coef_,
        }
    )
    coefficients["label"] = coefficients["feature"].map(feature_label)
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    top_n = st.slider("Number of features to display", 8, len(coefficients), 15)
    top_coefficients = coefficients.nlargest(top_n, "absolute_coefficient").sort_values("coefficient")

    coefficient_chart = px.bar(
        top_coefficients,
        x="coefficient",
        y="label",
        orientation="h",
        color="coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"coefficient": "Coefficient", "label": "Feature"},
        template="plotly_dark",
    )
    coefficient_chart.update_layout(
        height=max(480, top_n * 34),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(coefficient_chart, width="stretch")
    st.caption(
        "The sign indicates the direction of association, all else equal, but does "
        "not establish causality. Mean target encoding naturally makes a vehicle "
        "model's average price one of the strongest factors."
    )

st.divider()
st.caption(
    "CarValue · scikit-learn + Streamlit · Prices and errors are reported in INR."
)
