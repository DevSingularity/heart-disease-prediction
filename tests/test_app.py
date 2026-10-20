"""Application smoke tests using Streamlit's built-in AppTest runner.

Verifies that the app starts, all pages render without exceptions, and the
prediction interface returns a result for the default form values.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=120)


def test_app_starts_and_renders_overview():
    at = _app()
    at.run()
    assert not at.exception
    assert any("Heart Disease Risk Prediction" in t.value for t in at.title)


def test_every_page_renders_without_exceptions():
    at = _app()
    at.run()
    for page in (
        "Dataset Explorer",
        "Model Training & Comparison",
        "Risk Prediction Demo",
        "Research & Documentation",
        "Project Overview",
    ):
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception, f"page '{page}' raised: {at.exception}"
        assert at.title, f"page '{page}' rendered no title"


def test_prediction_interface_returns_result():
    from src.models.predict import available_models

    at = _app()
    at.run()
    at.sidebar.radio[0].set_value("Risk Prediction Demo")
    at.run()
    assert not at.exception

    if not available_models():
        # Artifacts absent: the page must show trainer instructions, not crash.
        assert at.info, "expected instructions when no models exist"
        assert "No trained models" in at.info[0].value
        pytest.skip("no trained artifacts in this environment (instruction path checked)")

    run_button = next(b for b in at.button if b.label == "Run prediction")
    run_button.click().run()
    assert not at.exception

    results = list(at.success) + list(at.error)
    assert results, "prediction produced neither a success nor an error result"
    assert any(
        "Predicted class" in m.value for m in results
    ), f"unexpected prediction output: {[m.value for m in results]}"


def test_model_training_page_shows_instructions_or_metrics():
    at = _app()
    at.run()
    at.sidebar.radio[0].set_value("Model Training & Comparison")
    at.run()
    assert not at.exception
    # With trained artifacts present we expect a metrics table; without them
    # we expect clear instructions. Both are acceptable, crashes are not.
    assert (at.success or at.warning), "no status shown on the training page"
    if not at.success:
        # instruction path: must name the exact command to run
        joined = " ".join(w.value for w in at.warning) + " ".join(
            c.value for c in at.code
        )
        assert "docker compose run --rm trainer" in joined
