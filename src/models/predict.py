"""Loading saved models and producing predictions + explanations.

The application uses these helpers; they never train or overwrite anything.
All model loading failures are translated into clear, actionable errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import artifacts, config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """A loaded pipeline plus its saved metadata."""

    name: str
    pipeline: Any
    metadata: dict


@dataclass(frozen=True)
class PredictionResult:
    """Prediction output for a single record."""

    predicted_class: int
    label: str
    probability: float | None  # None when the model exposes no scores


def available_models(artifacts_dir: Path | str | None = None) -> list[dict]:
    """Registry entries whose model files exist on the artifact volume."""
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    registry = artifacts.read_registry(artifacts_dir)
    if not registry:
        return []
    entries = []
    for name, info in registry.get("models", {}).items():
        model_path = artifacts_dir / info.get("model_file", "")
        if model_path.exists():
            entries.append({"name": name, **info})
    return entries


def load_model_bundle(
    name: str, artifacts_dir: Path | str | None = None
) -> ModelBundle:
    """Load a saved pipeline + metadata, with clear errors on any failure."""
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    registry = artifacts.read_registry(artifacts_dir) or {}
    info = registry.get("models", {}).get(name)
    if info is None:
        raise artifacts.ModelNotFoundError(
            f"Model '{name}' is not in the registry "
            f"({artifacts_dir / config.REGISTRY_PATH.name}). "
            f"Train the models first: {artifacts.TRAINER_HINT}"
        )
    pipeline = artifacts.load_joblib(artifacts_dir / info["model_file"])
    required_steps = {"preprocess", "model"}
    if not required_steps.issubset(getattr(pipeline, "named_steps", {})):
        raise artifacts.ModelError(
            f"Model file for '{name}' does not contain the expected "
            "'preprocess' + 'model' pipeline steps; the artifact is incompatible "
            f"with this application. Re-train: {artifacts.TRAINER_HINT}"
        )
    metadata = artifacts.read_json(artifacts_dir / info["metadata_file"]) or {}
    schema = metadata.get("feature_schema", {})
    expected = schema.get("features")
    if expected and list(expected) != list(config.FEATURES):
        raise artifacts.ModelError(
            f"Model '{name}' was trained with a different feature schema "
            f"({expected}); expected {config.FEATURES}. Re-train with the "
            f"current code: {artifacts.TRAINER_HINT}"
        )
    logger.info("Loaded model '%s' from %s", name, info["model_file"])
    return ModelBundle(name=name, pipeline=pipeline, metadata=metadata)


def build_input_frame(record: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with the exact training feature order."""
    missing = [f for f in config.FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    values = [float(record[f]) for f in config.FEATURES]
    return pd.DataFrame([values], columns=config.FEATURES)


def predict_record(pipeline, record: dict) -> PredictionResult:
    """Predict one record; feature order/schema is enforced from config."""
    frame = build_input_frame(record)
    predicted = int(pipeline.predict(frame)[0])
    probability: float | None = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(frame)[0, 1])
    elif hasattr(pipeline, "decision_function"):
        probability = None  # raw decision values are not probabilities
    label = (
        "Positive - heart disease indicated by the dataset's target definition"
        if predicted == 1
        else "Negative - no heart disease indicated by the dataset's target definition"
    )
    return PredictionResult(
        predicted_class=predicted, label=label, probability=probability
    )


def baseline_record(pipeline) -> dict:
    """Typical values per feature = the fitted imputer statistics.

    Numeric features use the training-set median; categorical features use the
    training-set most-frequent value. Missing predictors are imputed with these
    values inside the pipeline, so this is the natural "neutral" reference.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    numeric = preprocessor.named_transformers_["num"]
    categorical = preprocessor.named_transformers_["cat"]
    num_stats = numeric.named_steps["imputer"].statistics_
    cat_stats = categorical.named_steps["imputer"].statistics_
    baseline = dict(zip(config.NUMERIC_FEATURES, [float(v) for v in num_stats]))
    baseline.update(dict(zip(config.CATEGORICAL_FEATURES, [float(v) for v in cat_stats])))
    return baseline


def local_contributions(pipeline, record: dict, baseline: dict | None = None) -> list[dict]:
    """Single-prediction explanation via feature ablation.

    For each feature, its actual value is replaced by the typical
    (training-median / most-frequent) value while all other features keep
    their actual values; the contribution is the drop in the predicted
    probability of class 1. Contributions are not causal effects and do not
    necessarily sum to the difference from the baseline prediction.
    """
    if not hasattr(pipeline, "predict_proba"):
        return []
    baseline = baseline or baseline_record(pipeline)
    full = float(pipeline.predict_proba(build_input_frame(record))[0, 1])
    contributions = []
    for feature in config.FEATURES:
        ablated = dict(record)
        ablated[feature] = float(baseline[feature])
        ablated_p = float(pipeline.predict_proba(build_input_frame(ablated))[0, 1])
        contributions.append(
            {
                "feature": feature,
                "value": float(record[feature]),
                "typical_value": float(baseline[feature]),
                "contribution": full - ablated_p,
            }
        )
    contributions.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return contributions
