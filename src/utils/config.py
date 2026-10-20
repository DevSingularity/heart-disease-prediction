"""Central configuration and feature schema for the heart-disease project.

Every path is environment-driven so the exact same code runs inside Docker
(where ``/app/data``, ``/app/artifacts`` and ``/app/reports`` are persistent
named volumes) and in a local checkout (relative folders).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.environ.get("CV_FOLDS", "5"))
CV_SCORING = "roc_auc"

# --------------------------------------------------------------------------- #
# Paths (Docker volume mount points when the *_DIR environment variables exist)
# --------------------------------------------------------------------------- #


def _dir_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


DATA_DIR = _dir_from_env("DATA_DIR", "data")
ARTIFACTS_DIR = _dir_from_env("ARTIFACTS_DIR", "artifacts")
REPORTS_DIR = _dir_from_env("REPORTS_DIR", "reports")

MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"


def ensure_project_dirs(
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> None:
    """Create every writable project directory (idempotent)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    for directory in (
        data_dir,
        artifacts_dir / "models",
        artifacts_dir / "metadata",
        reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DATASET_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
DATASET_URL = os.environ.get(
    "DATASET_URL",
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
)
#: member file extracted from the official UCI zip archive
DATASET_MEMBER = "processed.cleveland.data"
#: filename stored inside the dataset volume
DATASET_FILE = "processed.cleveland.data"
#: human-readable dataset version recorded in model metadata
DATASET_VERSION = "UCI-45/heart-disease/cleveland-processed"
EXPECTED_ROWS = 303

# --------------------------------------------------------------------------- #
# Feature schema (the 13 predictors required by the project specification)
# --------------------------------------------------------------------------- #
FEATURES: list[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"
RAW_TARGET_COLUMN = "num"

#: Fallback input ranges (derived from the Cleveland dataset; used only when
#: the dataset itself cannot be read for range discovery).
NUMERIC_RANGES: dict[str, tuple[float, float, float, float]] = {
    # feature: (min, max, step, default)
    "age": (29, 77, 1, 54),
    "trestbps": (94, 200, 1, 130),
    "chol": (126, 564, 1, 250),
    "thalach": (71, 202, 1, 150),
    "oldpeak": (0.0, 6.2, 0.1, 1.0),
}

#: Readable labels for every categorical predictor value.
CATEGORICAL_LABELS: dict[str, dict[int, str]] = {
    "sex": {0: "Female (0)", 1: "Male (1)"},
    "cp": {
        1: "Typical angina (1)",
        2: "Atypical angina (2)",
        3: "Non-anginal pain (3)",
        4: "Asymptomatic (4)",
    },
    "fbs": {
        0: "Fasting blood sugar <= 120 mg/dL (0)",
        1: "Fasting blood sugar > 120 mg/dL (1)",
    },
    "restecg": {
        0: "Normal (0)",
        1: "ST-T wave abnormality (1)",
        2: "Left ventricular hypertrophy (2)",
    },
    "exang": {0: "No (0)", 1: "Yes (1)"},
    "slope": {1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"},
    "ca": {
        0: "0 major vessels (0)",
        1: "1 major vessel (1)",
        2: "2 major vessels (2)",
        3: "3 major vessels (3)",
        4: "4 major vessels (4)",
    },
    "thal": {
        3: "Normal (3)",
        6: "Fixed defect (6)",
        7: "Reversible defect (7)",
    },
}

#: Human-readable descriptions shown in the Dataset Explorer page.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Age in years.",
    "sex": "Sex: 1 = male, 0 = female.",
    "cp": "Chest pain type: 1 = typical angina, 2 = atypical angina, "
    "3 = non-anginal pain, 4 = asymptomatic.",
    "trestbps": "Resting blood pressure in mm Hg on admission to hospital.",
    "chol": "Serum cholesterol in mg/dL.",
    "fbs": "Fasting blood sugar > 120 mg/dL (1 = true, 0 = false).",
    "restecg": "Resting electrocardiographic results: 0 = normal, "
    "1 = ST-T wave abnormality, 2 = left ventricular hypertrophy.",
    "thalach": "Maximum heart rate achieved during exercise.",
    "exang": "Exercise-induced angina: 1 = yes, 0 = no.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the peak exercise ST segment: 1 = upsloping, "
    "2 = flat, 3 = downsloping.",
    "ca": "Number of major vessels (0-4) coloured by fluoroscopy.",
    "thal": "Thalassemia category: 3 = normal, 6 = fixed defect, "
    "7 = reversible defect.",
    "target": "Binary target: 0 = no heart disease (num == 0), "
    "1 = heart disease present (num > 0).",
}

DISCLAIMER = (
    "Educational demonstration only. This application is **not** a clinically "
    "validated diagnostic tool and must not be used to diagnose, treat, or "
    "make medical decisions. Always consult a qualified healthcare professional."
)
