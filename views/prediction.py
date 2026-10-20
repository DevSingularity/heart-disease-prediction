"""Page 4: interactive risk-prediction demonstration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models import predict as predict_mod
from src.utils import artifacts, config
from src.utils.artifacts import ModelError, ModelNotFoundError
from src.visualization import charts
from views.dataset_explorer import get_dataset


def _example_record_controls(dataset) -> None:
    """Example-record loader; rendered BEFORE the inputs so that setting
    widget state followed by ``st.rerun()`` is legal in Streamlit."""
    with st.expander("Load an example record from the dataset"):
        st.caption(
            "Rows are taken verbatim from the Cleveland dataset for "
            "demonstration — they are **not** a real-time clinical assessment."
        )
        idx = st.number_input(
            "Dataset row index",
            min_value=0,
            max_value=max(dataset.n_rows - 1, 0),
            value=0,
            step=1,
            key="example_idx",
        )
        row = dataset.features.iloc[int(idx)]
        shown = pd.DataFrame(
            {
                "feature": list(config.FEATURES) + [config.TARGET_COLUMN],
                "value": [float(row[f]) for f in config.FEATURES]
                + [int(dataset.target.iloc[int(idx)])],
            }
        )
        st.dataframe(shown, use_container_width=True, hide_index=True)
        if st.button("Fill the form with this record", key="fill_example"):
            for feature in config.FEATURES:
                st.session_state[f"feat_{feature}"] = float(row[feature])
            st.session_state["last_example_index"] = int(idx)
            st.rerun()


def render() -> None:
    st.title("Risk Prediction Demo")
    st.warning(config.DISCLAIMER)

    entries = predict_mod.available_models()
    if not entries:
        st.info(
            "No trained models are available in the artifact volume yet — "
            "the prediction interface is disabled until training has run at "
            "least once."
        )
        st.code("docker compose run --rm trainer", language="bash")
        return

    names = [entry["name"] for entry in entries]
    model_name = st.selectbox(
        "Trained model",
        names,
        format_func=lambda n: n.replace("_", " ").title(),
        help="All models were trained on the same split and preprocessing.",
    )

    try:
        bundle = predict_mod.load_model_bundle(model_name)
    except (ModelNotFoundError, ModelError) as exc:
        st.error(str(exc))
        return

    dataset, data_error = get_dataset()
    if dataset is not None:
        _example_record_controls(dataset)
    else:
        st.caption(
            "Example records are unavailable because the dataset could not be "
            "loaded (see the Dataset Explorer page for details)."
        )

    st.markdown("### Patient measurements")
    observed: dict[str, list] | None = None
    if dataset is not None:
        observed = {
            feature: sorted(dataset.features[feature].dropna().tolist())
            for feature in config.CATEGORICAL_FEATURES
        }

    columns = st.columns(3)
    record: dict[str, float] = {}
    for position, feature in enumerate(config.FEATURES):
        with columns[position % 3]:
            key = f"feat_{feature}"
            if feature in config.NUMERIC_FEATURES:
                fmin, fmax, step, _default = config.NUMERIC_RANGES[feature]
                if dataset is not None:
                    series = dataset.features[feature].dropna()
                    fmin, fmax = float(series.min()), float(series.max())
                default = (fmin + fmax) / 2
                if st.session_state.get(key) is not None:
                    default = float(st.session_state[key])
                label = f"{feature} — {config.FEATURE_DESCRIPTIONS.get(feature, '')}"
                if feature == "oldpeak":
                    record[feature] = float(
                        st.slider(
                            label, fmin, fmax, float(default), step=0.1, key=key
                        )
                    )
                else:
                    record[feature] = float(
                        st.slider(
                            label,
                            int(fmin),
                            int(fmax),
                            int(default),
                            step=int(step) or 1,
                            key=key,
                        )
                    )
            else:
                labels = config.CATEGORICAL_LABELS.get(feature, {})
                options = observed.get(feature) if observed else None
                if not options:
                    options = sorted(labels)
                options = [int(v) for v in options]
                index = None
                current = st.session_state.get(key)
                if current is not None and int(current) in options:
                    index = options.index(int(current))
                label = f"{feature} — {config.FEATURE_DESCRIPTIONS.get(feature, '')}"
                kwargs = {
                    "format_func": lambda v, f=feature, l=labels: l.get(
                        int(v), f"{f} = {v}"
                    ),
                    "key": key,
                }
                if index is not None:
                    kwargs["index"] = index
                choice = st.selectbox(label, options, **kwargs)
                record[feature] = float(options[0] if choice is None else choice)

    st.caption(
        "All 13 predictors use the schema the models were trained on; the "
        "submitted values are encoded by the exact preprocessing pipeline "
        "saved with the selected model."
    )

    if st.button("Run prediction", type="primary"):
        try:
            result = predict_mod.predict_record(bundle.pipeline, record)
        except Exception as exc:  # noqa: BLE001 - show any prediction failure clearly
            st.error(f"The prediction could not be computed: {exc}")
            return

        st.markdown("### Result")
        if result.predicted_class == 1:
            st.error(f"**Predicted class: 1 — {result.label}**")
        else:
            st.success(f"**Predicted class: 0 — {result.label}**")

        if result.probability is not None:
            st.progress(
                result.probability,
                text=f"Model score (predicted probability of class 1): "
                f"{result.probability:.2f}",
            )
            st.caption(
                "This score is a raw model output. It is **not** a calibrated "
                "medical probability — calibration has not been evaluated for "
                "these models."
            )
        else:
            st.caption("The selected model does not expose probabilities/scores.")

        contributions = predict_mod.local_contributions(bundle.pipeline, record)
        if contributions:
            st.markdown("### Why this prediction?")
            fig = charts.fig_local_contributions(contributions)
            st.pyplot(fig, clear_figure=True)
            st.dataframe(
                pd.DataFrame(contributions), use_container_width=True, hide_index=True
            )
            st.caption(
                "Method: each feature is replaced one at a time by its typical "
                "training value (median / most-frequent), keeping the others "
                "actual; the bars show how much the predicted probability of "
                "class 1 changes. These contributions are **not** causal "
                "effects and must not be used as medical advice."
            )

    st.markdown("### Model context")
    imp_path = config.REPORTS_DIR / f"feature_importance_{model_name}.png"
    if imp_path.exists():
        st.image(
            str(imp_path),
            caption=f"Global feature importance — {model_name}",
            use_container_width=True,
        )
    with st.expander("Model metadata (saved with the artifact)"):
        if bundle.metadata:
            st.json(bundle.metadata)
        else:
            st.caption("No metadata found for this model.")
