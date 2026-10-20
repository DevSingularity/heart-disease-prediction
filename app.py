"""Heart Disease Risk Prediction — Streamlit entry point.

Run locally with ``streamlit run app.py`` or (recommended) via Docker:

    docker compose up -d app     # http://localhost:8501

The app only *loads* artifacts produced by the `trainer` service; it never
trains or overwrites models on rerun.
"""

from __future__ import annotations

import streamlit as st

from src.utils import config
from views import dataset_explorer, model_training, overview, prediction, research

st.set_page_config(
    page_title="Heart Disease Risk Prediction (Educational)",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Project Overview": overview.render,
    "Dataset Explorer": dataset_explorer.render,
    "Model Training & Comparison": model_training.render,
    "Risk Prediction Demo": prediction.render,
    "Research & Documentation": research.render,
}

st.sidebar.title("🫀 Heart Disease Risk")
st.sidebar.caption("Educational ML demonstration — not a medical device.")
page = st.sidebar.radio("Navigation", list(PAGES), key="nav_page")
st.sidebar.divider()
st.sidebar.markdown(
    f"""
**Storage (Docker volumes)**
- data: `{config.DATA_DIR}`
- artifacts: `{config.ARTIFACTS_DIR}`
- reports: `{config.REPORTS_DIR}`

Retrain models:

```bash
docker compose run --rm trainer
```

Reference: [Karthick et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9085310/)
(article retracted — used only as an approximate benchmark).
"""
)

PAGES[page]()
