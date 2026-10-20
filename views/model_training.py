"""Page 3: model training status, metrics comparison and paper benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils import artifacts, config, paper


def render() -> None:
    st.title("Model Training & Comparison")

    registry = artifacts.read_registry()
    comparison_path = config.REPORTS_DIR / "model_comparison.csv"

    if not registry or not comparison_path.exists():
        st.warning(
            "No trained models or evaluation reports were found in the "
            "persistent volumes yet."
        )
        st.markdown("Run the finite training job once from the project root:")
        st.code("docker compose run --rm trainer", language="bash")
        st.caption(
            f"Artifacts volume path: `{config.ARTIFACTS_DIR}` · "
            f"Reports volume path: `{config.REPORTS_DIR}`. "
            "The web app never trains models by itself and never overwrites "
            "saved artifacts."
        )
        return

    models = registry.get("models", {})
    st.success(
        f"{len(models)} trained model(s) available — trained "
        f"{registry.get('generated_at', 'unknown time')} · "
        f"random seed `{registry.get('random_seed')}` · "
        f"dataset sha256 `{str(registry.get('dataset_sha256', ''))[:12]}…`"
    )

    table = pd.read_csv(comparison_path)
    display = table.drop(columns=["best_params"], errors="ignore")
    st.markdown("### Locally measured metrics (held-out test set)")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "Test split = 20% of 303 records, stratified, produced with the fixed "
        "seed above. Hyperparameters were selected with 5-fold stratified CV "
        "on the **training** split only (no test-set tuning)."
    )

    st.markdown("### Model comparison chart")
    chart = config.REPORTS_DIR / "model_comparison_chart.png"
    if chart.exists():
        st.image(str(chart), use_container_width=True)
    else:
        st.info("Chart not found in the reports volume — re-run the trainer.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion matrices")
        path = config.REPORTS_DIR / "confusion_matrices.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("Run the trainer to generate this chart.")
    with col2:
        st.markdown("### ROC curves")
        path = config.REPORTS_DIR / "roc_curves.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("Run the trainer to generate this chart.")

    st.markdown("### Feature importance (global)")
    tabs = st.tabs([name.replace("_", " ") for name in models])
    for tab, name in zip(tabs, models):
        with tab:
            path = config.REPORTS_DIR / f"feature_importance_{name}.png"
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Run the trainer to generate this chart.")
            st.caption(
                "Global importance is a model-behaviour summary, not proof of "
                "causation and not medical advice."
            )

    st.markdown("### Reference-paper benchmark")
    st.warning(
        f"Published values only — from {paper.PAPER['title']} "
        f"({paper.PAPER['doi']}). {paper.PAPER['retraction_notice']}"
    )
    comparison = config.REPORTS_DIR / "paper_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), use_container_width=True, hide_index=True)
    else:
        st.info("Run the trainer to generate the paper-comparison table.")
    st.caption(
        "N/A = metric not reported by the paper (or model not trained locally). "
        "Published and locally measured numbers are never mixed in one column."
    )

    with st.expander("Per-model training metadata"):
        for name, info in models.items():
            meta_path = config.ARTIFACTS_DIR / info.get("metadata_file", "")
            st.markdown(f"**{name}**")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    st.json(
                        {
                            k: meta[k]
                            for k in (
                                "model_type",
                                "trained_at",
                                "random_seed",
                                "best_params",
                                "cv_best_score",
                                "metrics",
                                "library_versions",
                                "dataset",
                            )
                            if k in meta
                        }
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    st.error(f"Metadata unreadable: {exc}")
            else:
                st.caption("Metadata file missing — re-run the trainer.")

    with st.expander("Report files in the reports volume"):
        reports = sorted(p.name for p in Path(config.REPORTS_DIR).glob("*") if p.is_file())
        st.code("\n".join(reports) or "(empty)", language="text")
