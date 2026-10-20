"""Metrics, artifact persistence and prediction tests."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.loader import load_heart_dataset
from src.models import evaluate, predict as predict_mod
from src.preprocessing.pipeline import build_pipeline
from src.utils import artifacts, config


def test_metrics_computed_correctly():
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0]
    y_score = [0.1, 0.6, 0.9, 0.7, 0.4, 0.2]
    metrics = evaluate.compute_metrics(y_true, y_pred, y_score)
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(8 / 9)


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    """Train a small pipeline on the real dataset and persist it atomically."""
    tmp = tmp_path_factory.mktemp("artifacts")
    dataset = load_heart_dataset()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(dataset.features, dataset.target)

    artifacts.atomic_joblib_dump(pipeline, tmp / "models" / "logistic_regression.joblib")
    artifacts.atomic_write_json(
        tmp / "metadata" / "logistic_regression.json",
        {
            "model_name": "logistic_regression",
            "model_type": "LogisticRegression",
            "feature_schema": {"features": config.FEATURES},
            "random_seed": 42,
        },
    )
    artifacts.write_registry(
        {
            "schema_version": 1,
            "models": {
                "logistic_regression": {
                    "model_file": "models/logistic_regression.joblib",
                    "metadata_file": "metadata/logistic_regression.json",
                    "model_type": "LogisticRegression",
                }
            },
        },
        tmp,
    )
    return tmp


def test_registry_lists_available_models(artifacts_dir):
    entries = predict_mod.available_models(artifacts_dir)
    assert [entry["name"] for entry in entries] == ["logistic_regression"]


def test_saved_artifact_reloads_with_metadata(artifacts_dir):
    bundle = predict_mod.load_model_bundle("logistic_regression", artifacts_dir)
    assert bundle.metadata["model_type"] == "LogisticRegression"
    assert bundle.metadata["feature_schema"]["features"] == config.FEATURES
    assert hasattr(bundle.pipeline, "predict")


def test_prediction_uses_expected_schema_and_valid_labels(artifacts_dir):
    dataset = load_heart_dataset()
    bundle = predict_mod.load_model_bundle("logistic_regression", artifacts_dir)
    record = {feature: float(dataset.features.iloc[0][feature]) for feature in config.FEATURES}
    result = predict_mod.predict_record(bundle.pipeline, record)
    assert result.predicted_class in {0, 1}
    assert result.label.startswith(("Positive", "Negative"))
    assert result.probability is None or 0.0 <= result.probability <= 1.0


def test_feature_order_is_enforced(artifacts_dir):
    dataset = load_heart_dataset()
    bundle = predict_mod.load_model_bundle("logistic_regression", artifacts_dir)
    record = {feature: float(dataset.features.iloc[3][feature]) for feature in config.FEATURES}
    reversed_record = dict(reversed(list(record.items())))
    a = predict_mod.predict_record(bundle.pipeline, record)
    b = predict_mod.predict_record(bundle.pipeline, reversed_record)
    assert a.predicted_class == b.predicted_class
    assert a.probability == pytest.approx(b.probability)

    frame = predict_mod.build_input_frame(record)
    assert list(frame.columns) == config.FEATURES
    with pytest.raises(ValueError, match="Missing features"):
        predict_mod.build_input_frame({k: 1.0 for k in config.FEATURES[:-1]})


def test_local_contributions_are_ordered_and_complete(artifacts_dir):
    dataset = load_heart_dataset()
    bundle = predict_mod.load_model_bundle("logistic_regression", artifacts_dir)
    record = {feature: float(dataset.features.iloc[7][feature]) for feature in config.FEATURES}
    contributions = predict_mod.local_contributions(bundle.pipeline, record)
    assert [row["feature"] for row in contributions] == sorted(
        (row["feature"] for row in contributions),
        key=lambda f: abs(dict((c["feature"], c["contribution"]) for c in contributions)[f]),
        reverse=True,
    )
    assert {row["feature"] for row in contributions} == set(config.FEATURES)
    baseline = predict_mod.baseline_record(bundle.pipeline)
    assert set(baseline) == set(config.FEATURES)


def test_missing_model_raises_actionable_error(artifacts_dir):
    with pytest.raises(artifacts.ModelNotFoundError, match="trainer"):
        predict_mod.load_model_bundle("does_not_exist", artifacts_dir)


def test_corrupt_model_raises_clear_error(artifacts_dir):
    model_path = artifacts_dir / "models" / "logistic_regression.joblib"
    backup = model_path.read_bytes()
    try:
        model_path.write_bytes(b"not a real joblib file")
        with pytest.raises(artifacts.ModelError, match="corrupt|incompatible"):
            predict_mod.load_model_bundle("logistic_regression", artifacts_dir)
    finally:
        model_path.write_bytes(backup)


def test_atomic_writes_leave_no_temp_files(artifacts_dir):
    artifacts.atomic_joblib_dump(
        {"a": 1}, artifacts_dir / "models" / "atomic_check.joblib"
    )
    artifacts.atomic_write_json(artifacts_dir / "metadata" / "atomic_check.json", {"ok": True})
    leftovers = [
        path.name
        for path in (artifacts_dir / "models").iterdir()
        if path.name.endswith(".tmp")
    ] + [
        path.name
        for path in (artifacts_dir / "metadata").iterdir()
        if path.name.endswith(".tmp")
    ]
    assert leftovers == []
    payload = json.loads((artifacts_dir / "metadata" / "atomic_check.json").read_text())
    assert payload == {"ok": True}


def test_incompatible_registry_model_is_rejected(artifacts_dir, tmp_path):
    import shutil

    clone = tmp_path / "clone"
    shutil.copytree(artifacts_dir, clone)
    bundle_meta = json.loads(
        (clone / "metadata" / "logistic_regression.json").read_text()
    )
    bundle_meta["feature_schema"]["features"] = ["something", "else"]
    (clone / "metadata" / "logistic_regression.json").write_text(json.dumps(bundle_meta))
    with pytest.raises(artifacts.ModelError, match="feature schema"):
        predict_mod.load_model_bundle("logistic_regression", clone)


def test_random_seed_is_fixed_for_reproducibility():
    assert config.RANDOM_SEED == int(__import__("os").environ.get("RANDOM_SEED", "42"))
