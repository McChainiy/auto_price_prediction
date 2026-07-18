"""Model loading, feature preparation, and inference helpers.

The serialized pipeline was trained in a notebook where the custom transformer
lived in ``__main__``. ``TrustedModelUnpickler`` maps that legacy reference to
the maintained implementation below, so the artifact can be loaded from the
app, tests, or another Python module.

Only the model artifact committed to this repository should be loaded. Pickle
files can execute arbitrary code and must never be accepted from app users.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_new_features.pkl"

REQUIRED_FEATURES = (
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
)

MANUFACTURER_COUNTRY = {
    "Ambassador": "India",
    "Ashok": "India",
    "Audi": "Germany",
    "BMW": "Germany",
    "Chevrolet": "USA",
    "Daewoo": "South Korea",
    "Datsun": "Japan",
    "Fiat": "Italy",
    "Force": "India",
    "Ford": "USA",
    "Honda": "Japan",
    "Hyundai": "South Korea",
    "Isuzu": "Japan",
    "Jaguar": "United Kingdom",
    "Jeep": "USA",
    "Kia": "South Korea",
    "Land": "United Kingdom",
    "Lexus": "Japan",
    "MG": "United Kingdom",
    "Mahindra": "India",
    "Maruti": "India",
    "Mercedes-Benz": "Germany",
    "Mitsubishi": "Japan",
    "Nissan": "Japan",
    "Opel": "Germany",
    "Peugeot": "France",
    "Renault": "France",
    "Skoda": "Czech Republic",
    "Tata": "India",
    "Toyota": "Japan",
    "Volkswagen": "Germany",
    "Volvo": "Sweden",
}


class MyTransormer(BaseEstimator, TransformerMixin):
    """Transform raw car listings into the features expected by the model.

    The misspelling in the class name is intentionally preserved for backward
    compatibility with the serialized model.
    """

    def __init__(self) -> None:
        self.regex_pattern = r"[-+]?([0-9]*\.[0-9]+|\d+)"

    def fit(self, X_real: pd.DataFrame, y_real: pd.Series) -> "MyTransormer":
        X = X_real.copy(deep=True)
        y = pd.Series(y_real).copy(deep=True)

        unique_rows = ~X.duplicated(keep="first")
        X = X.loc[unique_rows].reset_index(drop=True)
        y = y.loc[unique_rows].reset_index(drop=True)

        for column in ("mileage", "engine", "max_power"):
            X[column] = self.extract_number(X[column])
        X["max_torque_rpm"] = X["torque"].apply(self.extract_rpm)
        X["torque"] = X["torque"].apply(self.extract_torque)

        self.cols_with_na = X.columns[X.isna().any()].tolist()
        self.train_medians = {
            column: X[column].median() for column in self.cols_with_na
        }
        for column, median in self.train_medians.items():
            X[column] = X[column].fillna(median)

        X["engine"] = X["engine"].astype(int)
        X["seats"] = X["seats"].astype(int)
        X["name"] = X["name"].apply(self._short_model_name)
        X["year"] = X["year"] ** 2
        # The legacy feature name is kept because the fitted scaler expects it.
        X["enigne_over_power"] = X["max_power"] / X["engine"]
        X["fuel_spent"] = X["km_driven"] / 100 * X["mileage"]

        target_frame = pd.concat(
            [X, y.rename("selling_price")], axis=1
        )
        self.means = target_frame.groupby("name")["selling_price"].mean()
        self.train_mean_y = float(y.mean())
        X["model_avg_price"] = X["name"].map(self.means).fillna(self.train_mean_y)
        X["country"] = X["name"].apply(self.get_country)

        self.cat_columns = [
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "country",
        ]
        self.ohc_cols = {}
        for column in self.cat_columns:
            encoder = OneHotEncoder(
                sparse_output=False,
                drop="first",
                handle_unknown="ignore",
            )
            encoder.fit(X[[column]])
            self.ohc_cols[column] = encoder

        self.raw_feature_names_ = X_real.columns.tolist()
        self.cat_columns_unique = {
            column: X[column].unique()
            for column in (
                "name",
                "fuel",
                "seller_type",
                "transmission",
                "owner",
                "seats",
            )
        }
        return self

    def transform(self, X_real: pd.DataFrame) -> pd.DataFrame:
        X = X_real.copy(deep=True)
        missing_value_rows = X.isna().any(axis=1)
        X["skipped_flag"] = missing_value_rows

        for column in ("mileage", "engine", "max_power"):
            X[column] = self.extract_number(X[column])
        X["max_torque_rpm"] = X["torque"].apply(self.extract_rpm)
        X["torque"] = X["torque"].apply(self.extract_torque)

        for column in self.cols_with_na:
            X[column] = X[column].fillna(self.train_medians[column])

        X["engine"] = X["engine"].astype(int)
        X["seats"] = X["seats"].astype(int)
        X["name"] = X["name"].apply(self._short_model_name)
        X["year"] = X["year"] ** 2
        X["enigne_over_power"] = X["max_power"] / X["engine"]
        X["fuel_spent"] = X["km_driven"] / 100 * X["mileage"]
        X["model_avg_price"] = X["name"].map(self.means).fillna(self.train_mean_y)
        X["country"] = X["name"].apply(self.get_country)

        X = self.add_ohe_features(X)
        X = X.drop(columns=self.cat_columns)
        X["log_km_driven"] = np.log1p(X["km_driven"])
        X = X.drop(columns=["km_driven", "name"])
        self.feature_names_ = X.columns.tolist()
        return X

    def extract_number(self, series: pd.Series) -> pd.Series:
        def parse(value: object) -> float:
            if pd.isna(value):
                return np.nan
            matches = re.findall(self.regex_pattern, str(value))
            return float(matches[0]) if matches else np.nan

        return series.apply(parse)

    @staticmethod
    def extract_rpm(value: object) -> float:
        if pd.isna(value):
            return np.nan
        match = re.findall(
            r"(@|at|/)\s*[^0-9]*(?:[0-9,.]+-)?([0-9,.]*)",
            str(value),
            re.IGNORECASE,
        )
        if not match or not match[0][1]:
            return np.nan
        return float(match[0][1].replace(",", ""))

    @staticmethod
    def extract_torque(value: object) -> float:
        if pd.isna(value):
            return np.nan
        match = re.findall(
            r"([-+]?(?:\d*\.\d+|\d+))[^a-zA-Z]*(kgm|nm)?",
            str(value),
            re.IGNORECASE,
        )
        if not match:
            return np.nan
        number, unit = match[0]
        return float(number) * (9.8 if unit.lower() == "kgm" else 1)

    @staticmethod
    def _short_model_name(value: object) -> str:
        return " ".join(str(value).split()[:2])

    @staticmethod
    def get_country(name: object) -> str:
        manufacturer = str(name).split()[0] if str(name).split() else ""
        return MANUFACTURER_COUNTRY.get(manufacturer, "Other")

    def add_ohe_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in self.cat_columns:
            encoder = self.ohc_cols[column]
            encoded = encoder.transform(X[[column]])
            X[encoder.get_feature_names_out([column])] = encoded
        return X


class TrustedModelUnpickler(pickle.Unpickler):
    """Resolve the transformer's legacy notebook module path."""

    def find_class(self, module: str, name: str) -> type:
        if module == "__main__" and name == "MyTransormer":
            return MyTransormer
        return super().find_class(module, name)


def _load(file: BinaryIO):
    return TrustedModelUnpickler(file).load()


def load_pipeline(model_path: str | Path = MODEL_PATH):
    """Load the trusted, repository-owned sklearn pipeline."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as model_file:
        return _load(model_file)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw dataset and return columns in the trained schema order."""

    clean = frame.copy(deep=True)
    unnamed_columns = clean.columns[clean.columns.str.match(r"^Unnamed")]
    clean = clean.drop(columns=unnamed_columns)

    missing = [column for column in REQUIRED_FEATURES if column not in clean.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_list}")

    return clean.loc[:, REQUIRED_FEATURES].copy()


def predict_prices(model, frame: pd.DataFrame, *, clip: bool = False) -> np.ndarray:
    """Predict prices in INR; optionally enforce the non-negative price domain."""

    predictions = np.asarray(model.predict(prepare_features(frame)), dtype=float)
    return np.maximum(predictions, 0) if clip else predictions
