"""Atomic persistence helpers for model artifacts, metadata and the registry.

All writes go through a temporary file in the destination directory followed
by ``os.replace`` so readers (the Streamlit app) never observe a half-written
artifact.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

TRAINER_HINT = "docker compose run --rm trainer"


class ModelNotFoundError(FileNotFoundError):
    """A requested model artifact does not exist (trainer not run yet)."""


class ModelError(RuntimeError):
    """A model artifact exists but is corrupt or incompatible."""


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_joblib_dump(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    logger.info("Saved artifact: %s", path)


def read_json(path: Path | str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(
            f"Metadata file {path} is unreadable or corrupt ({exc}). "
            f"Re-train to regenerate it: {TRAINER_HINT}"
        ) from exc


def load_joblib(path: Path | str) -> Any:
    """Load a Joblib artifact, translating failures into helpful errors.

    SECURITY NOTE: Joblib/pickle files can execute arbitrary code when loaded.
    Only load model files produced by this project's own trainer.
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found: {path}. "
            f"Train the models first: {TRAINER_HINT}"
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surface any unpickling failure clearly
        raise ModelError(
            f"Model file {path} could not be loaded (corrupt or incompatible "
            f"with the installed scikit-learn version): {exc}. "
            f"Re-train to regenerate all artifacts: {TRAINER_HINT}"
        ) from exc


def write_registry(registry: dict, artifacts_dir: Path | str | None = None) -> Path:
    from src.utils import config

    artifacts_dir = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    path = artifacts_dir / config.REGISTRY_PATH.name
    atomic_write_json(path, registry)
    return path


def read_registry(artifacts_dir: Path | str | None = None) -> dict | None:
    from src.utils import config

    artifacts_dir = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR
    return read_json(artifacts_dir / config.REGISTRY_PATH.name)
