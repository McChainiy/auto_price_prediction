# CarValue - Used Car Price Prediction

[![CI](https://github.com/McChainiy/auto_price_prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/McChainiy/auto_price_prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)

**An end-to-end machine learning project that estimates used-car prices on the
Indian market:** from exploratory data analysis and feature engineering to a
versioned `scikit-learn` pipeline, automated tests, and an interactive Streamlit
application.

> Prices and error metrics are reported in Indian rupees (INR).

## Results at a Glance

| Metric | Holdout result |
|---|---:|
| **R²** | **0.887** |
| **MAE** | **127,522 INR** |
| RMSE | 254,925 INR |
| Holdout size | 1,000 vehicles |

Feature engineering improved R² from **0.600 for the baseline Linear Regression
to 0.887 for the final pipeline**. The largest gain came from representing the
data better, not simply switching estimators.

For intended use, evaluation details, and risks, see the
[Model Card](MODEL_CARD.md).

## Why This Project Stands Out

- **A single reproducible sklearn pipeline.** Raw feature processing, scaling,
  and Ridge Regression are packaged into one fitted artifact, keeping training
  and inference behavior consistent.
- **Parsing real-world listing data.** A custom `TransformerMixin` extracts
  values from strings such as `19.7 kmpl`, `1248 CC`, `74 bhp`, and
  `190 Nm at 2000 rpm`, including `kgm -> Nm` conversion.
- **Purposeful feature engineering.** The model uses squared model year,
  log-transformed distance driven, power-to-engine ratio, a fuel-use proxy,
  manufacturer country, and model-level average price.
- **Robust missing-value and category handling.** Median imputation is paired
  with a missing-value flag; one-hot encoding supports unseen categories; and
  unknown vehicle models fall back to the training-set target mean.
- **Leakage-aware target encoding.** Model-level mean prices are learned only
  from the training data and stored inside the fitted transformer.
- **Interpretable predictions.** Standardized Ridge coefficients are exposed in
  the application for direct comparison.
- **A usable product interface.** The Streamlit app supports single predictions,
  CSV batch inference, evaluation metrics, lightweight EDA, phi-k correlations,
  coefficient analysis, and result export.
- **Engineering safeguards.** Dependencies are pinned, model metadata is
  versioned, the pickle artifact is verified with SHA-256 before loading, and
  GitHub Actions runs compilation and regression tests.

## Solution Architecture

```mermaid
flowchart LR
    A[Raw vehicle listing] --> B[Regex parsing]
    B --> C[Imputation + missing flag]
    C --> D[Feature engineering]
    D --> E[OHE + target encoding]
    E --> F[StandardScaler]
    F --> G[Ridge Regression]
    G --> H[Price in INR]
```

The final pipeline turns 12 raw input fields into 31 model features:

```text
raw listing -> MyTransormer -> StandardScaler -> Ridge(alpha=100) -> prediction
```

`MyTransormer` intentionally keeps a historical spelling mistake for backward
compatibility with the trained pickle artifact. A dedicated unpickler maps the
legacy notebook class to the maintained implementation in `src/model.py`.

## Experiment Summary

| Model | R² | MAE, INR | Main change |
|---|---:|---:|---|
| Linear Regression | 0.600 | 220,873 | Numeric features only |
| Lasso | 0.600 | 220,872 | L1 regularization |
| Ridge + OHE | 0.781 | 168,628 | Added categorical features |
| **Final Ridge Pipeline** | **0.887** | **127,522** | Feature engineering + scaling + Ridge |

The research notebook also covers:

- manual R² and adjusted R² calculation;
- a custom L0 regression implementation;
- 10-fold `GridSearchCV` for Lasso, ElasticNet, and Ridge;
- Pearson, Spearman, Kendall, and phi-k correlations;
- an asymmetric business metric for overpricing and underpricing;
- outlier analysis and model-weight visualization.

The complete experiment log is available in the
[Jupyter Notebook](AI_HW1_Regression_with_inference_pro_pt1.ipynb). The notebook
retains the original Russian course prompts; all public-facing documentation and
the application UI are provided in English.

## Application Capabilities

1. **Estimate one vehicle** through a form with explicit measurement units.
2. **Upload a CSV file** and generate a prediction for every row.
3. **Evaluate model quality** with R², MAE, and RMSE when `selling_price` is present.
4. **Explore a dataset** through distributions, missing values, duplicates, and phi-k correlations.
5. **Interpret the model** through its strongest positive and negative coefficients.
6. **Export results** as a CSV file with predictions and absolute errors.

## Technology Stack

- **Machine learning:** scikit-learn, NumPy, pandas
- **Modeling:** Ridge Regression, StandardScaler, OneHotEncoder, custom TransformerMixin
- **Analytics:** phi-k, Plotly; seaborn, matplotlib, and ydata-profiling in the notebook
- **Application:** Streamlit
- **Quality:** unittest, Streamlit AppTest, GitHub Actions, versioned metadata, SHA-256 verification

## Repository Structure

```text
.
├── .github/workflows/ci.yml       # Compilation and regression-test workflow
├── .streamlit/config.toml         # Application theme and server settings
├── models/
│   ├── metadata.json              # Model version, schema, metrics, and checksum
│   └── ridge_new_features.pkl     # Fitted sklearn Pipeline
├── src/
│   └── model.py                   # Transformer, validation, loading, and inference
├── tests/
│   ├── test_app.py                # Headless Streamlit smoke test
│   └── test_model.py              # Numerical regression and edge-case tests
├── .python-version                # Shared Python version for local tooling
├── app.py                         # Streamlit application
├── MODEL_CARD.md                  # Intended use, evaluation, and model risks
├── LICENSE                        # MIT License
├── test.csv                       # Example data and reproducible holdout set
├── requirements.txt               # Pinned runtime dependencies
└── AI_HW1_Regression_...ipynb     # EDA, experiments, and model training
```

## Quick Start

Python 3.12 is required.

```bash
git clone https://github.com/McChainiy/auto_price_prediction.git
cd auto_price_prediction

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

streamlit run app.py
```

The fitted model is included in `models/`, so retraining is not required to run
the application.

For Streamlit Community Cloud, select `app.py` as the entry point and Python 3.12
in Advanced settings. All Python dependencies are declared in the root
`requirements.txt`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite locks down reference predictions and published metrics, validates the
input schema and measurement parsing, checks unknown manufacturers and price
clipping, verifies metadata and SHA-256 integrity, and starts the complete app
through `streamlit.testing`.

## CSV Input Schema

Required columns:

```text
name, year, km_driven, fuel, seller_type, transmission, owner,
mileage, engine, max_power, torque, seats
```

The `selling_price` column is optional. When present, it is used only to compute
evaluation metrics. A ready-to-use example is provided in [test.csv](test.csv).

## Limitations

- The model was trained on historical data from the Indian market and should not
  be transferred directly to other countries or currencies.
- It does not observe vehicle condition, accident history, region, trim level,
  or current market dynamics.
- A linear model may produce negative values for inputs far outside the training
  distribution. The UI clips displayed predictions at zero, while published
  research metrics use the original outputs.
- Pickle files must only be loaded from trusted sources. The application loads
  the repository-owned artifact and never accepts user-supplied models.

## Data

The original train and test tables come from the educational
[HSE MLDS dataset](https://github.com/Murcha1990/MLDS_ML_2022/tree/main/Hometasks/HT1).
The holdout set is included for demonstration and reproducibility. Review the
upstream dataset terms before reusing or redistributing the data.

## License

The project code is released under the [MIT License](LICENSE). Dataset usage is
subject to the terms of its original source.
