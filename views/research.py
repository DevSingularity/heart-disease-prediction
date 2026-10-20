"""Page 5: research paper, methodology, comparison, limitations, ethics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils import config, paper


def render() -> None:
    st.title("Research & Documentation")

    st.markdown("### Reference paper")
    st.markdown(
        f"""
        **{paper.PAPER['title']}**

        {paper.PAPER['authors']}

        *{paper.PAPER['journal']}* ({paper.PAPER['year']})

        DOI: [{paper.PAPER['doi']}](https://doi.org/{paper.PAPER['doi']}) ·
        Full text: [{paper.PAPER['pmcid']}]({paper.PAPER['url']})
        """
    )
    if paper.PAPER["retracted"]:
        st.error("**Retraction notice:** " + paper.PAPER["retraction_notice"])
    st.markdown(f"**Reported protocol:** {paper.PAPER['protocol']}")

    st.markdown("### Dataset citation")
    st.markdown(
        f"""
        - Detrich R. et al. (1988), *Heart Disease Data Set*, UCI Machine
          Learning Repository: [{config.DATASET_SOURCE_PAGE}]({config.DATASET_SOURCE_PAGE})
        - Cleveland subset: {config.EXPECTED_ROWS} instances, 13 predictive
          attributes + diagnosis (`num`, here binarised).
        - Target definition used in this project: `num == 0` → 0,
          `num > 0` (categories 1–4) → 1.
        - Version identifier (SHA-256 of the downloaded file) is recorded in
          every model's metadata so experiments are traceable to the exact data.
        """
    )

    st.markdown("### Methodology (this project)")
    st.markdown(
        f"""
        1. Download the official UCI Cleveland processed file into the
           persistent data volume; validate schema and target values.
        2. Stratified 80/20 train/test split with fixed seed
           `{config.RANDOM_SEED}`.
        3. Preprocessing inside each scikit-learn pipeline (fit on training
           data only): median imputation + standard scaling for numeric
           features; most-frequent imputation + one-hot encoding for
           categorical features.
        4. Five classifiers — logistic regression, decision tree, random
           forest, SVM (RBF), KNN — tuned with `GridSearchCV`
           ({config.CV_FOLDS}-fold stratified CV, ROC-AUC) on the training
           split only.
        5. Final evaluation on the untouched test split: accuracy, precision,
           recall/sensitivity, specificity, F1, ROC-AUC, confusion matrices,
           ROC curves, classification reports, feature importance.
        6. Pipelines + metadata saved atomically to the artifact volume;
           all reports saved to the reports volume.
        """
    )
    st.caption(
        "**Deviations from the paper:** no chi-square feature-selection stage "
        "(all 13 standard predictors are used); one-hot encoding + explicit "
        "imputation instead of the paper's undocumented preprocessing; a "
        "documented, seeded split instead of the paper's unpublished split; "
        "cross-validated hyperparameter tuning. Because the paper's exact "
        "split and preprocessing cannot be reproduced, the comparison below "
        "is **approximate**."
    )

    st.markdown("### Model comparison — published vs. locally measured")
    comparison = config.REPORTS_DIR / "paper_comparison.csv"
    if comparison.exists():
        st.dataframe(
            pd.read_csv(comparison), use_container_width=True, hide_index=True
        )
    else:
        st.info(
            "No local results yet. Run the trainer to generate the comparison "
            "table:"
        )
        st.code("docker compose run --rm trainer", language="bash")
    st.caption(paper.PAPER["missing_metric_note"])

    st.markdown("### Why accuracy alone is insufficient")
    st.markdown(
        """
        In a medical-risk setting the two error types are **not** equally
        costly. A **false negative** — a sick patient told they are healthy —
        can delay diagnosis and treatment with severe consequences, while a
        false positive typically leads to additional (usually non-invasive)
        testing. Accuracy weights both errors identically and can look high
        even when a dangerous share of positives is missed; recall
        (sensitivity), specificity, precision, F1 and ROC-AUC expose the
        trade-offs that accuracy hides. This is also why a model score must
        never be presented as a calibrated medical probability without
        calibration studies.
        """
    )

    st.markdown("### Known limitations")
    st.markdown(
        """
        - The Cleveland dataset is tiny (303 records, single centre, 1988),
          so all metrics have wide confidence intervals and models can
          overfit; single-split numbers vary noticeably with the seed.
        - Features are limited to the 13 classical predictors; modern
          biomarkers, imaging and patient history are absent.
        - No calibration, fairness or external-validation study was performed;
          performance on other populations is unknown.
        - The reference article is retracted, so even the external benchmark
          must be treated with caution.
        - One-hot encoding expands categorical levels; unseen levels at
          prediction time are ignored (`handle_unknown='ignore'`), which can
          silently reduce information.
        - Loading Joblib/pickle artifacts executes code — only artifacts from
          this project's own trainer should ever be loaded.
        """
    )

    st.markdown("### Ethical considerations")
    st.markdown(
        """
        - This application is an **educational demonstration**, not a medical
          device; it has no regulatory approval and must not guide clinical
          decisions.
        - Model outputs could encode historical biases (e.g. sex imbalance in
          the dataset: ~68% male) and must not be used to deny care.
        - Patient data in the dataset is de-identified and public, but any
          real deployment would require consent, privacy safeguards and
          clinical validation.
        - Explainability outputs describe model behaviour, not causation, and
          must not be used to justify treatment decisions.
        """
    )

    st.markdown("### Reproducing the experiment (Docker-only)")
    st.code(
        """# 1. Build the image (host needs only Docker + Docker Compose)
docker compose build

# 2. Train + evaluate (finite job; writes dataset/models/reports to volumes)
docker compose run --rm trainer

# 3. Start the web app at http://localhost:8501
docker compose up -d app

# 4. Tests inside the container
docker compose run --rm trainer python -m pytest -q

# 5. Stop containers WITHOUT deleting the volumes
docker compose down""",
        language="bash",
    )
    st.caption(
        f"Paths inside the containers — data: `{config.DATA_DIR}`, "
        f"artifacts: `{config.ARTIFACTS_DIR}`, reports: `{config.REPORTS_DIR}` "
        "(named volumes `heart_dataset_data`, `heart_model_artifacts`, "
        "`heart_experiment_reports`)."
    )
