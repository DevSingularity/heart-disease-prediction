"""Preprocessing pipeline and train/test-leakage tests (synthetic data)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessing.pipeline import build_pipeline, build_preprocessor
from src.utils import config


def _synthetic(n: int = 90, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(30, 80, n).astype(float),
            "sex": rng.integers(0, 2, n).astype(float),
            "cp": rng.integers(1, 5, n).astype(float),
            "trestbps": rng.integers(100, 180, n).astype(float),
            "chol": rng.integers(180, 400, n).astype(float),
            "fbs": rng.integers(0, 2, n).astype(float),
            "restecg": rng.integers(0, 3, n).astype(float),
            "thalach": rng.integers(90, 200, n).astype(float),
            "exang": rng.integers(0, 2, n).astype(float),
            "oldpeak": rng.uniform(0, 4, n).round(1),
            "slope": rng.integers(1, 4, n).astype(float),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
        }
    )
    frame.loc[frame.index[:4], "age"] = np.nan  # force missing values
    target = pd.Series(
        ((frame["age"].fillna(55) + frame["cp"] * 5) > 62).astype(int),
        name=config.TARGET_COLUMN,
    )
    return frame, target


def test_preprocessor_fits_and_transforms():
    X, _ = _synthetic()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()
    names = list(preprocessor.get_feature_names_out())
    assert len(names) == transformed.shape[1]
    assert any(name.startswith("num__") for name in names)
    assert any(name.startswith("cat__") for name in names)


def test_no_train_test_leakage():
    X, _ = _synthetic()
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
    preprocessor = build_preprocessor().fit(X_train)

    numeric = preprocessor.named_transformers_["num"]
    age_median = float(np.nanmedian(X_train["age"].to_numpy()))
    assert numeric.named_steps["imputer"].statistics_[0] == pytest.approx(age_median)
    # The scaler must be fitted on median-imputed TRAIN data only.
    imputed_train_age = X_train["age"].fillna(age_median).to_numpy()
    assert numeric.named_steps["scaler"].mean_[0] == pytest.approx(
        imputed_train_age.mean(), rel=1e-9
    )
    # And it must differ from what a leaky full-data fit would produce.
    full_median = float(np.nanmedian(X["age"].to_numpy()))
    full_mean = X["age"].fillna(full_median).mean()
    assert imputed_train_age.mean() != pytest.approx(full_mean, rel=1e-9)


def test_all_13_features_are_used():
    X, _ = _synthetic()
    preprocessor = build_preprocessor().fit(X)
    used = set()
    for _, _, columns in preprocessor.transformers_:
        if isinstance(columns, list):
            used.update(columns)
    assert used == set(config.FEATURES)


def test_pipeline_trains_predicts_and_ignores_column_order():
    X, y = _synthetic()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(5))
    assert set(np.unique(preds)).issubset({0, 1})
    # ColumnTransformer selects by name, so row ordering of columns is irrelevant.
    reordered = X[list(reversed(config.FEATURES))]
    assert (pipeline.predict(X) == pipeline.predict(reordered)).all()


def test_unknown_categorical_level_is_ignored():
    X, y = _synthetic()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    pipeline.fit(X, y)
    row = X.head(1).copy()
    row["thal"] = 99.0  # unseen level
    assert int(pipeline.predict(row)[0]) in {0, 1}
