# Model Card: CarValue

## Model Overview

CarValue estimates used-car prices on the Indian market from 12 raw listing
fields. The final model is a serialized `scikit-learn Pipeline` that combines a
custom feature transformer, `StandardScaler`, and `Ridge Regression` with
`alpha=100`.

| Property | Value |
|---|---|
| Version | 1.0.0 |
| Artifact | `models/ridge_new_features.pkl` |
| Framework | scikit-learn 1.7.1 |
| Python | 3.12 |
| Target | `selling_price`, INR |
| Raw inputs | 12 features |
| Model inputs | 31 engineered features |

Machine-readable model details are stored in
[`models/metadata.json`](models/metadata.json).

## Intended Use

The model is intended for:

- demonstrating an end-to-end ML workflow and reproducible inference;
- educational experiments with regression and feature engineering;
- approximate evaluation of samples drawn from the original data distribution.

It is not intended for financial decisions, professional vehicle valuation,
credit scoring, or deployment outside the Indian market without retraining and
independent validation.

## Inputs and Outputs

Required input fields:

```text
name, year, km_driven, fuel, seller_type, transmission, owner,
mileage, engine, max_power, torque, seats
```

The `mileage`, `engine`, `max_power`, and `torque` fields may include measurement
units in the listing string. The output is one floating-point price in INR per
row. Calling `predict_prices(..., clip=True)` constrains the returned values to
the non-negative price domain.

## Evaluation

Metrics were calculated on a holdout set of 1,000 vehicles:

| Metric | Value |
|---|---:|
| R² | 0.8869457592 |
| MAE | 127,521.59 INR |
| RMSE | 254,925.10 INR |

Reference predictions and metrics are locked down in
[`tests/test_model.py`](tests/test_model.py) to detect behavioral changes caused
by inference-code or environment updates.

## Feature Processing

- Regex parsing for power, engine displacement, mileage, torque, and RPM
- Median imputation with an additional missing-value indicator
- Squared year, log-transformed distance, and numeric interaction features
- One-hot encoding with support for unseen categories
- Vehicle-model mean target encoding learned only from the training data
- Training-target mean fallback for previously unseen vehicle models

## Limitations and Risks

- The data represents historical listings from the Indian market; inflation and
  current market dynamics are not modeled.
- The model does not observe region, trim level, technical condition, accident
  history, or legal restrictions.
- Error is not uniform across price segments and rare manufacturers; aggregate
  MAE alone does not describe performance for every subgroup.
- Target encoding can reinforce the relationship between a vehicle model's
  popularity and its typical price segment.
- Linear regression can produce negative or unrealistic values for samples far
  outside the training distribution.

## Reproducibility and Security

Runtime dependencies are pinned in `requirements.txt`, and the Python version is
declared in `.python-version`. Before deserialization,
`src.model.load_pipeline()` checks the artifact SHA-256 against the value stored
in `models/metadata.json`.

Pickle files can execute arbitrary code. Only load the model artifact from a
trusted copy of this repository, and never pass user-supplied models to
`load_pipeline()`.

## Data Provenance

The train and test tables originate from the educational
[HSE MLDS dataset](https://github.com/Murcha1990/MLDS_ML_2022/tree/main/Hometasks/HT1).
Review the upstream dataset terms before reusing or redistributing the data.
