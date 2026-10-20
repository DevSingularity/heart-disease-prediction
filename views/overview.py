"""Page 1: Project overview."""

from __future__ import annotations

import platform

import streamlit as st

import src
from src.utils import config, paper


def render() -> None:
    st.title("Heart Disease Risk Prediction")
    st.subheader("An educational machine-learning demonstration on the UCI Cleveland dataset")
    st.markdown(config.DISCLAIMER)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Objective")
        st.markdown(
            """
            Build and compare **five classical classifiers** (logistic regression,
            decision tree, random forest, SVM, KNN) that predict whether a patient
            record belongs to the *heart-disease-positive* or *heart-disease-negative*
            class defined by the UCI Cleveland dataset, then explain each prediction.

            The full pipeline — preprocessing, stratified splitting, cross-validated
            hyperparameter tuning, evaluation and persistence — is implemented with
            scikit-learn and runs entirely inside Docker.
            """
        )
        st.markdown("### Research reference")
        st.markdown(
            f"""
            **{paper.PAPER['title']}**

            {paper.PAPER['authors']}

            *{paper.PAPER['journal']}*, {paper.PAPER['year']}

            DOI: [{paper.PAPER['doi']}](https://doi.org/{paper.PAPER['doi']}) ·
            [Full text ({paper.PAPER['pmcid']})]({paper.PAPER['url']})
            """
        )
        if paper.PAPER["retracted"]:
            st.warning(
                "**Note:** the reference article is marked as **retracted** by the "
                "publisher. Its reported numbers are used here only as an "
                "approximate external benchmark (see the Research page)."
            )
        st.markdown("### Dataset")
        st.markdown(
            f"""
            **UCI Heart Disease — Cleveland processed subset**
            ({config.EXPECTED_ROWS} patient records, {len(config.FEATURES)} predictors)

            Source: [{config.DATASET_SOURCE_PAGE}]({config.DATASET_SOURCE_PAGE})

            Target: `num == 0` → class 0 (no heart disease indicated);
            `num > 0` (severity 1–4) → class 1 (heart disease present).
            """
        )

    with col2:
        st.markdown("### Technologies")
        st.markdown(
            f"""
            - Python {platform.python_version()} (pinned in `requirements.txt`)
            - pandas · NumPy · scikit-learn · Joblib
            - Matplotlib · Seaborn (evaluation plots)
            - Streamlit (web interface)
            - Docker & Docker Compose (containerized execution, persistent volumes)
            - Fixed random seed `{config.RANDOM_SEED}` for reproducible experiments
            """
        )
        st.markdown("### How to use this app")
        st.markdown(
            """
            1. **Dataset Explorer** — inspect the raw Cleveland data.
            2. **Model Training & Comparison** — locally measured metrics vs. the
               reference paper's published benchmark.
            3. **Risk Prediction Demo** — enter clinical measurements and obtain a
               dataset-defined classification with an explanation.
            4. **Research & Documentation** — methodology, comparison, limitations
               and ethical considerations.
            """
        )
        st.info(
            "Models are **not** trained inside the web app. They are produced by the "
            "`trainer` service and loaded from the persistent artifact volume:\n\n"
            "```bash\ndocker compose run --rm trainer\n```"
        )
