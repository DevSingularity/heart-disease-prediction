"""Training and evaluation entry point (the Docker ``trainer`` service).

Run with::

    python -m src.models.train
    docker compose run --rm trainer

The pipeline is: download/validate the dataset (into the persistent data
volume), stratified 80/20 split with a fixed seed, per-model ``GridSearchCV``
on the training split only, held-out test evaluation, then atomic persistence
of pipelines, metadata, registry and all reports to the persistent volumes.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_heart_dataset
from src.models import evaluate
from src.preprocessing.pipeline import build_pipeline
from src.utils import artifacts, config

logger = logging.getLogger(__name__)


def model_grids(seed: int) -> dict[str, tuple]:
    """Model instances + deliberately small search grids (fast on CPU)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed),
            {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [1, 5]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": [200],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "svm": (
            SVC(probability=True, random_state=seed),
            {
                "model__C": [0.5, 1.0, 5.0],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", 0.01],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [5, 9, 15],
                "model__weights": ["uniform", "distance"],
            },
        ),
    }


def train_all(
    seed: int | None = None,
    test_size: float | None = None,
    data_dir=None,
    artifacts_dir=None,
    reports_dir=None,
    download: bool = True,
) -> dict:
    """Full training + evaluation + persistence run. Returns evaluated models."""
    seed = config.RANDOM_SEED if seed is None else seed
    test_size = config.TEST_SIZE if test_size is None else test_size
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    reports_dir = reports_dir or config.REPORTS_DIR
    config.ensure_project_dirs(
        data_dir=data_dir, artifacts_dir=artifacts_dir, reports_dir=reports_dir
    )

    dataset = load_heart_dataset(data_dir=data_dir, download=download)
    X, y = dataset.features, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    logger.info(
        "Split: train=%d test=%d (stratified, test_size=%.2f, seed=%d)",
        len(X_train),
        len(X_test),
        test_size,
        seed,
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    results: dict[str, evaluate.ModelEvaluation] = {}
    trained_at = datetime.now(timezone.utc).isoformat()

    for name, (estimator, grid) in model_grids(seed).items():
        logger.info("Training '%s' (grid: %s) ...", name, grid)
        search = GridSearchCV(
            build_pipeline(estimator),
            grid,
            cv=cv,
            scoring=config.CV_SCORING,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        result = evaluate.evaluate_model(
            name, search.best_estimator_, X_train, X_test, y_train, y_test
        )
        result.best_params = search.best_params_
        result.cv_best_score = float(search.best_score_)
        results[name] = result
        logger.info(
            "'%s': accuracy=%.4f recall=%.4f specificity=%.4f roc_auc=%s "
            "(best params: %s)",
            name,
            result.metrics["accuracy"],
            result.metrics["recall"],
            result.metrics["specificity"],
            (
                f"{result.metrics['roc_auc']:.4f}"
                if result.metrics["roc_auc"] is not None
                else "n/a"
            ),
            search.best_params_,
        )

        artifacts.atomic_joblib_dump(
            result.pipeline, Path(artifacts_dir) / "models" / f"{name}.joblib"
        )
        metadata = {
            "model_name": name,
            "model_type": result.model_type,
            "trained_at": trained_at,
            "random_seed": seed,
            "test_size": test_size,
            "cv_folds": config.CV_FOLDS,
            "cv_scoring": config.CV_SCORING,
            "cv_best_score": round(result.cv_best_score, 4),
            "best_params": search.best_params_,
            "feature_schema": {
                "features": config.FEATURES,
                "numeric_features": config.NUMERIC_FEATURES,
                "categorical_features": config.CATEGORICAL_FEATURES,
                "target": config.TARGET_COLUMN,
                "target_definition": "num == 0 -> 0; num > 0 (1-4) -> 1",
            },
            "dataset": {
                "version": config.DATASET_VERSION,
                "source_url": config.DATASET_SOURCE_PAGE,
                "sha256": dataset.sha256,
                "rows": dataset.n_rows,
                "dropped_rows": dataset.dropped_rows,
            },
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in result.metrics.items()
            },
            "train_accuracy": round(result.train_accuracy, 4),
            "library_versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "model_file": f"models/{name}.joblib",
        }
        artifacts.atomic_write_json(
            Path(artifacts_dir) / "metadata" / f"{name}.json", metadata
        )

    run_meta = {
        "trained_at": trained_at,
        "random_seed": seed,
        "test_size": test_size,
        "cv_folds": config.CV_FOLDS,
        "cv_scoring": config.CV_SCORING,
        "library_versions": metadata["library_versions"],
        "dataset": metadata["dataset"],
    }
    evaluate.write_all_reports(results, X_test, y_test, run_meta, reports_dir=reports_dir)

    registry = {
        "schema_version": 1,
        "generated_at": trained_at,
        "random_seed": seed,
        "test_size": test_size,
        "dataset_sha256": dataset.sha256,
        "models": {
            name: {
                "model_file": f"models/{name}.joblib",
                "metadata_file": f"metadata/{name}.json",
                "model_type": r.model_type,
                "trained_at": trained_at,
                "best_params": r.best_params,
                "metrics": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.metrics.items()
                },
            }
            for name, r in results.items()
        },
    }
    artifacts.write_registry(registry, artifacts_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    started = time.time()
    logger.info("=== Heart disease model training starting ===")
    logger.info("data=%s artifacts=%s reports=%s",
                config.DATA_DIR, config.ARTIFACTS_DIR, config.REPORTS_DIR)
    try:
        results = train_all()
    except Exception:
        logger.exception("Training failed")
        return 1
    summary = evaluate.comparison_dataframe(results)
    logger.info("Training finished in %.1fs. Summary:\n%s", time.time() - started, summary.to_string(index=False))
    logger.info("Artifacts: %s | Reports: %s", config.ARTIFACTS_DIR, config.REPORTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
