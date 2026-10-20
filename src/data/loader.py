"""Download, validate and load the UCI Cleveland heart-disease dataset.

The official UCI archive (``heart+disease.zip``) is downloaded into the
persistent dataset directory (``/app/data`` inside Docker) and the
``processed.cleveland.data`` member is extracted from it.  A manual
``processed.cleveland.data`` file placed in the same directory is used
without downloading.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import config

logger = logging.getLogger(__name__)

COLUMNS = config.FEATURES + [config.RAW_TARGET_COLUMN]

MANUAL_INSTRUCTIONS = """\
Manual dataset installation:
  1. On any machine with internet access, download
     {page}
     (direct file: {url})
  2. Extract the member "{member}" from the zip archive.
  3. Copy it into the persistent dataset volume, e.g.:

       docker run --rm -v heart_dataset_data:/data -v "$PWD":/host \\
           alpine cp /host/{member} /data/{member}

  4. Re-run the trainer:  docker compose run --rm trainer
"""


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be downloaded, located or validated."""


@dataclass(frozen=True)
class HeartDataset:
    """Validated Cleveland dataset (13 predictors + binary target)."""

    features: pd.DataFrame  # 13 predictor columns, float dtype
    target: pd.Series  # binary {0, 1}, name = "target"
    raw: pd.DataFrame  # predictors + original "num" column
    sha256: str  # checksum of the source file (dataset version identifier)
    source: str  # file path used
    dropped_rows: int  # rows removed by documented cleaning rules

    @property
    def n_rows(self) -> int:
        return len(self.raw)

    @property
    def positive_rate(self) -> float:
        return float(self.target.mean())


def dataset_path(data_dir: Path | str | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    return data_dir / config.DATASET_FILE


def download_dataset(
    data_dir: Path | str | None = None,
    url: str | None = None,
    timeout: int = 60,
) -> Path:
    """Ensure ``processed.cleveland.data`` exists in ``data_dir``.

    Downloads the official UCI zip archive when the file is absent and writes
    it atomically into the (persistent) data directory.  If the download fails
    but a previously downloaded/manual copy exists, that copy is reused.
    """
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    url = url or config.DATASET_URL
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / config.DATASET_FILE

    if target.exists() and target.stat().st_size > 0:
        logger.info("Dataset already present: %s", target)
        return target

    temp_path = target.parent / (target.name + ".part")
    try:
        logger.info("Downloading dataset from %s ...", url)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next(
                (n for n in archive.namelist() if n.endswith(config.DATASET_MEMBER)),
                None,
            )
            if member is None:
                raise DatasetError(
                    f"'{config.DATASET_MEMBER}' not found inside archive "
                    f"(members: {archive.namelist()})"
                )
            csv_bytes = archive.read(member)
        with open(temp_path, "wb") as handle:
            handle.write(csv_bytes)
        os.replace(temp_path, target)  # atomic on POSIX/NTFS
        logger.info("Dataset saved to %s (%d bytes)", target, len(csv_bytes))
        return target
    except (urllib.error.URLError, OSError, zipfile.BadZipFile, DatasetError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        if target.exists() and target.stat().st_size > 0:
            logger.warning(
                "Download failed (%s); reusing existing file %s", exc, target
            )
            return target
        raise DatasetError(
            f"Could not download the dataset from {url}: {exc}\n"
            + MANUAL_INSTRUCTIONS.format(
                page=config.DATASET_SOURCE_PAGE,
                url=config.DATASET_URL,
                member=config.DATASET_FILE,
            )
        ) from exc


def _read_raw_frame(path: Path) -> pd.DataFrame:
    """Read the Cleveland CSV (headerless, '?' = missing) with header detection."""
    peek = pd.read_csv(path, header=None, dtype=str, na_values=["?"])
    first_cell = str(peek.iloc[0, 0]).strip().lower() if len(peek) else ""
    has_header = not _is_number(first_cell)
    if has_header:
        header = [str(c).strip().lower() for c in peek.iloc[0]]
        expected = set(COLUMNS)
        provided = set(header)
        if not expected.issubset(provided):
            raise DatasetError(
                "The supplied dataset has unexpected column names.\n"
                f"  Expected columns: {sorted(expected)}\n"
                f"  Found columns:    {sorted(provided)}\n"
                "Only the official UCI Cleveland processed dataset is supported; "
                "a different heart-disease dataset is never silently substituted."
            )
        frame = pd.read_csv(path, dtype=str, na_values=["?"])
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        return frame[list(COLUMNS)]
    peek.columns = COLUMNS
    return peek


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_heart_dataset(
    data_dir: Path | str | None = None,
    url: str | None = None,
    download: bool = True,
) -> HeartDataset:
    """Load and validate the Cleveland dataset.

    Documented cleaning rules:
      * ``?`` markers become missing values (NaN); missing predictor values are
        imputed later inside the scikit-learn pipeline (fit on training data).
      * Rows whose raw target ``num`` is missing or outside ``0..4`` are dropped.
      * ``num == 0`` maps to class 0; ``num > 0`` (severity 1-4) maps to class 1.
    """
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    path = dataset_path(data_dir)
    if not (path.exists() and path.stat().st_size > 0):
        if not download:
            raise DatasetError(
                f"Dataset not found at {path}.\n"
                + MANUAL_INSTRUCTIONS.format(
                    page=config.DATASET_SOURCE_PAGE,
                    url=config.DATASET_URL,
                    member=config.DATASET_FILE,
                )
            )
        path = download_dataset(data_dir=data_dir, url=url)

    frame = _read_raw_frame(path)
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.empty:
        raise DatasetError(f"Dataset at {path} contains no rows.")

    invalid_target = ~frame[config.RAW_TARGET_COLUMN].isin([0, 1, 2, 3, 4])
    dropped = int(invalid_target.sum())
    if dropped:
        logger.warning(
            "Dropping %d row(s) with missing/invalid target 'num' values.", dropped
        )
    frame = frame[~invalid_target].reset_index(drop=True)

    if config.RAW_TARGET_COLUMN in frame.columns:
        raw_num = frame[config.RAW_TARGET_COLUMN].astype(int)
    else:  # pragma: no cover - defensive
        raise DatasetError("Raw target column 'num' is missing after validation.")

    features = frame[config.FEATURES].astype(float)
    target = (raw_num > 0).astype(int).rename(config.TARGET_COLUMN)

    if len(frame) != config.EXPECTED_ROWS:
        logger.warning(
            "Expected %d Cleveland records, found %d - check the dataset source.",
            config.EXPECTED_ROWS,
            len(frame),
        )
    if not np.isfinite(features.to_numpy()).all() and features.isna().sum().sum() == 0:
        raise DatasetError("Dataset contains non-finite predictor values.")

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    raw_frame = features.copy()
    raw_frame[config.RAW_TARGET_COLUMN] = raw_num

    logger.info(
        "Loaded %d records (%d features) | positives=%d negatives=%d | sha256=%s",
        len(features),
        len(config.FEATURES),
        int(target.sum()),
        int((target == 0).sum()),
        sha256[:12],
    )
    return HeartDataset(
        features=features,
        target=target,
        raw=raw_frame,
        sha256=sha256,
        source=str(path),
        dropped_rows=dropped,
    )
