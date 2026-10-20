"""Page 2: Dataset explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.loader import DatasetError, load_heart_dataset
from src.utils import config
from src.visualization import charts


@st.cache_data(show_spinner="Loading dataset …", ttl=120)
def _load_dataset() -> tuple:
    dataset = load_heart_dataset(download=True)
    return dataset, dataset.features.copy(), dataset.target.copy(), dataset.raw.copy()


def get_dataset():
    """Load the dataset, returning ``(dataset, error_message_or_None)``."""
    try:
        dataset, _, _, _ = _load_dataset()
        return dataset, None
    except DatasetError as exc:
        return None, str(exc)


def render() -> None:
    st.title("Dataset Explorer")
    st.caption(
        f"UCI Heart Disease (Cleveland processed subset) — loaded from "
        f"`{config.DATA_DIR}` · source: {config.DATASET_SOURCE_PAGE}"
    )

    dataset, error = get_dataset()
    if error:
        st.error(
            "The dataset could not be loaded.\n\n" + error
        )
        return

    features, target, raw = dataset.features, dataset.target, dataset.raw

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", dataset.n_rows)
    c2.metric("Predictors", len(config.FEATURES))
    c3.metric("Class 1 (disease)", int(target.sum()))
    c4.metric("Missing cells (predictors)", int(features.isna().sum().sum()))

    st.markdown("### Preview")
    st.dataframe(raw.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Target class distribution")
        fig = charts.fig_class_distribution(target)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            f"Class 1 = `num > 0` ({target.mean():.1%} of records). "
            "Roughly balanced classes — still, accuracy alone is not enough "
            "(see Research page)."
        )
    with right:
        st.markdown("### Missing values per feature")
        missing = features.isna().sum().sort_values(ascending=False)
        missing_df = pd.DataFrame(
            {"feature": missing.index, "missing": missing.values}
        )
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        st.caption(
            "Missing values ('?' in the raw file) are imputed **inside** the "
            "model pipeline (median for numeric, most-frequent for categorical "
            "features), fitted on training data only."
        )

    st.markdown("### Age distribution by target class")
    exploratory = features.copy()
    exploratory[config.TARGET_COLUMN] = target
    fig = charts.fig_age_distribution_by_class(exploratory)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Feature correlation matrix")
    fig = charts.fig_correlation_heatmap(exploratory)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Correlation is descriptive only — it does not imply causation."
    )

    st.markdown("### Feature descriptions")
    described = {
        feature: config.FEATURE_DESCRIPTIONS.get(feature, "")
        for feature in config.FEATURES + [config.TARGET_COLUMN]
    }
    st.dataframe(
        pd.DataFrame(
            {"feature": described.keys(), "description": described.values()}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.expander("Dataset file details").markdown(
        f"- Path: `{dataset.source}`\n"
        f"- SHA-256: `{dataset.sha256}`\n"
        f"- Version tag: `{config.DATASET_VERSION}`\n"
        f"- Rows dropped by target validation: {dataset.dropped_rows}"
    )
