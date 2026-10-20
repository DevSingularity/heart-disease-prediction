"""Chart builders for evaluation reports and the Streamlit application.

Every ``fig_*`` function returns a matplotlib ``Figure`` so callers can either
save it (``save_figure``) or render it inline with ``st.pyplot``.
``save_figure`` writes atomically via a temp file in the target directory.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from src.utils import config  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
METRIC_COLUMNS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def save_figure(fig, path: Path | str) -> Path:
    """Save a figure atomically (temp file + os.replace) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        fig.savefig(tmp_name, format="png", dpi=150, bbox_inches="tight")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    finally:
        plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path


def fig_model_comparison(results) -> "plt.Figure":
    """Grouped bar chart of the locally measured metrics for every model."""
    rows = []
    for name, result in results.items():
        row = {"model": name}
        for metric in METRIC_COLUMNS:
            value = result.metrics.get(metric)
            row[metric] = value if value is not None else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot(kind="bar", ax=ax, colormap="viridis", width=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (held-out test set)")
    ax.set_xlabel("Model")
    ax.set_title("Model comparison - locally measured test metrics")
    ax.legend(title="Metric", fontsize=8, loc="lower right")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def fig_confusion_matrices(results, y_true=None) -> "plt.Figure | None":
    """Confusion-matrix grid for every model (values from each result)."""
    if not results:
        return None
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, result) in zip(axes, results.items()):
        cm = result.metrics.get("confusion_matrix")
        if cm:
            matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        else:
            truth = y_true if y_true is not None else result.y_pred
            matrix = confusion_matrix(truth, result.y_pred, labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["pred 0", "pred 1"],
            yticklabels=["true 0", "true 1"],
            ax=ax,
        )
        ax.set_title(f"{name}", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (held-out test set)", y=1.02)
    fig.tight_layout()
    return fig


def fig_roc_curves(results, y_true) -> "plt.Figure | None":
    """ROC curves for every model that exposes scores."""
    usable = {k: v for k, v in results.items() if v.y_score is not None}
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, result in usable.items():
        fpr, tpr, _ = roc_curve(y_true, result.y_score)
        auc = result.metrics.get("roc_auc")
        label = f"{name} (AUC = {auc:.3f})" if auc is not None else name
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall / sensitivity)")
    ax.set_title("ROC curves - held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def transformed_feature_names(pipeline) -> list[str]:
    """Feature names after preprocessing (one-hot expanded)."""
    preprocessor = pipeline.named_steps["preprocess"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def feature_importance_values(pipeline, X_test=None, y_test=None):
    """Importance vector + human-readable method label for a fitted pipeline.

    Tree ensembles expose native impurity-based importances; for other models
    (logistic regression, SVM, KNN) permutation importance on the held-out
    test split is computed with a fixed seed.
    """
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float), (
            f"{type(model).__name__}: impurity-based feature importance"
        )
    if X_test is None or y_test is None:
        return None, "permutation importance needs a held-out sample"
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=config.RANDOM_SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return result.importances_mean, (
        f"{type(model).__name__}: permutation importance "
        "(mean ROC-AUC decrease, 10 repeats, held-out test set)"
    )


def fig_feature_importance(pipeline, X_test=None, y_test=None) -> "plt.Figure | None":
    """Horizontal bar chart of global feature importance (one-hot features)."""
    values, label = feature_importance_values(pipeline, X_test, y_test)
    if values is None:
        return None
    names = transformed_feature_names(pipeline)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.32 * len(names))))
    ax.barh([names[i] for i in order], values[order], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Global feature importance\n{label}", fontsize=10)
    fig.tight_layout()
    return fig


def fig_class_distribution(y) -> "plt.Figure":
    """Target class distribution bar chart."""
    series = pd.Series(np.asarray(y))
    counts = series.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["0 - no disease", "1 - disease present"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["#4c78a8", "#e45756"],
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(counts.max() * 1.15, 1))
    ax.set_title("Target class distribution")
    fig.tight_layout()
    return fig


def fig_age_distribution_by_class(df) -> "plt.Figure":
    """Age histogram split by target class."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, group in df.groupby(config.TARGET_COLUMN):
        ax.hist(
            group["age"],
            bins=12,
            alpha=0.6,
            label=f"class {label}",
            color="#4c78a8" if label == 0 else "#e45756",
        )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Patients")
    ax.set_title("Age distribution by target class")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(df) -> "plt.Figure":
    """Correlation heatmap of the 13 predictors + target."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature correlation matrix (incl. target)")
    fig.tight_layout()
    return fig


def fig_local_contributions(contributions) -> "plt.Figure":
    """Diverging bar chart of single-prediction contribution scores."""
    frame = pd.DataFrame(contributions)
    frame = frame.reindex(frame["contribution"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(frame) + 1.6))
    colors = ["#e45756" if v > 0 else "#4c78a8" for v in frame["contribution"]]
    ax.barh(frame["feature"], frame["contribution"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in predicted probability of class 1\n"
                  "(actual value vs. typical/median-missing value)")
    ax.set_title("Individual prediction explanation", fontsize=10)
    fig.tight_layout()
    return fig
