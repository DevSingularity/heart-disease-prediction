"""Dataset loading, validation and target-mapping tests."""

import pytest

from src.data.loader import DatasetError, download_dataset, load_heart_dataset
from src.utils import config


@pytest.fixture(scope="module")
def dataset():
    """Download (into the persistent data dir once) and load the dataset."""
    return load_heart_dataset()


def test_expected_row_count(dataset):
    assert dataset.n_rows == config.EXPECTED_ROWS == 303


def test_feature_schema(dataset):
    assert list(dataset.features.columns) == config.FEATURES
    assert len(config.FEATURES) == 13
    assert list(dataset.raw.columns) == config.FEATURES + [config.RAW_TARGET_COLUMN]


def test_binary_target_mapping(dataset):
    assert set(dataset.target.unique()) == {0, 1}
    raw_num = dataset.raw[config.RAW_TARGET_COLUMN]
    assert ((raw_num > 0).astype(int) == dataset.target).all()
    assert ((raw_num == 0).astype(int) == (dataset.target == 0)).all()
    # both classes present
    assert dataset.target.sum() > 0
    assert (dataset.target == 0).sum() > 0


def test_missing_values_become_nan(dataset):
    # Cleveland data uses '?' in ca/thal; those must load as numeric NaN so the
    # pipeline imputer can handle them.
    assert dataset.features.isna().sum().sum() >= 1
    for dtype in dataset.features.dtypes:
        assert dtype.kind == "f"


def test_sha256_recorded(dataset):
    assert len(dataset.sha256) == 64


def test_download_is_idempotent(tmp_path):
    first = download_dataset(data_dir=tmp_path)
    second = download_dataset(data_dir=tmp_path)
    assert first == second
    assert first.exists()
    assert first.stat().st_size > 0


def test_headerless_rejection_of_unknown_columns(tmp_path):
    target = tmp_path / config.DATASET_FILE
    target.write_text("age,sex\n57,1\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="unexpected column names"):
        load_heart_dataset(data_dir=tmp_path, download=False)


def test_documented_target_cleaning_rule(tmp_path):
    # One valid row (num=0) and one row with an out-of-range target (9).
    target = tmp_path / config.DATASET_FILE
    target.write_text(
        "57,1,1,130,236,0,1,174,0,0.0,2,0,3,0\n"
        "63,1,1,145,233,1,2,150,0,2.3,3,0,6,9\n",
        encoding="utf-8",
    )
    loaded = load_heart_dataset(data_dir=tmp_path, download=False)
    assert loaded.dropped_rows == 1
    assert loaded.n_rows == 1
    assert loaded.target.tolist() == [0]


def test_header_detection_with_named_columns(tmp_path):
    target = tmp_path / config.DATASET_FILE
    header = ",".join(config.FEATURES + [config.RAW_TARGET_COLUMN])
    row = "57,1,1,130,236,0,1,174,0,0.0,2,0,3,1"
    target.write_text(f"{header}\n{row}\n", encoding="utf-8")
    loaded = load_heart_dataset(data_dir=tmp_path, download=False)
    assert loaded.n_rows == 1
    assert loaded.target.tolist() == [1]
