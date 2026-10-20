"""Scikit-learn preprocessing and full model pipelines.

All preprocessing (imputation, scaling, one-hot encoding) lives inside the
saved pipeline and is fitted **only** on training data, which prevents
train/test leakage and guarantees that the application applies exactly the
same transformations the model was trained with.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import config


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + standard-scale numerics; impute + one-hot categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the shared preprocessing steps."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
