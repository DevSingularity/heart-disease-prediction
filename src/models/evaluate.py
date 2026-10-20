"""Model evaluation: metrics, reports and paper-comparison artifacts.

All evaluation outputs (CSV, JSON, TXT, Markdown) are written to the
persistent reports directory via atomic writes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import artifacts, config, paper

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluation:
    """Everything needed to report and visualise one trained model."""

    name: str
    pipeline: object
    y_pred: np.ndarray
    y_score: np.ndarray | None
    metrics: dict
    train_accuracy: float
    best_params: dict = field(default_factory=dict)
    cv_best_score: float | None = None
    model_type: str = ""


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Accuracy, precision, recall, specificity, F1 and ROC-AUC (class 1 = positive)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_score)) if y_score is not None else None
    )
    return metrics


def score_for_roc(pipeline, X) -> np.ndarray | None:
    """Positive-class probabilities, falling back to decision scores."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(X), dtype=float)
    return None


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test) -> ModelEvaluation:
    """Evaluate a fitted pipeline on the held-out test split."""
    y_pred = pipeline.predict(X_test)
    y_score = score_for_roc(pipeline, X_test)
    metrics = compute_metrics(y_test, y_pred, y_score)
    train_accuracy = float(accuracy_score(y_train, pipeline.predict(X_train)))
    model = pipeline.named_steps["model"]
    return ModelEvaluation(
        name=name,
        pipeline=pipeline,
        y_pred=np.asarray(y_pred),
        y_score=y_score,
        metrics=metrics,
        train_accuracy=train_accuracy,
        model_type=type(model).__name__,
    )


def comparison_dataframe(results: dict[str, ModelEvaluation]) -> pd.DataFrame:
    """Local metrics table (what this experiment actually measured)."""
    rows = []
    for name, result in results.items():
        m = result.metrics
        rows.append(
            {
                "model": name,
                "model_type": result.model_type,
                "train_accuracy": round(result.train_accuracy, 4),
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "specificity": round(m["specificity"], 4),
                "f1": round(m["f1"], 4),
                "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
                "best_params": result.best_params,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("accuracy", ascending=False).reset_index(drop=True)
    return frame


def paper_comparison_dataframe(results: dict[str, ModelEvaluation]) -> pd.DataFrame:
    """Paper-reported values (verified against the article) vs local measurements."""
    local = comparison_dataframe(results).set_index("model")
    rows = []
    all_models = list(local.index) + [
        m for m in paper.PAPER["models_not_trained_locally"] if m not in local.index
    ]
    for name in all_models:
        reported_acc = paper.PAPER["reported_test_accuracy"].get(name)
        reported_train = paper.PAPER["reported_train_accuracy"].get(name)
        reported_auc = paper.PAPER["reported_roc_auc"].get(name)
        row = {
            "model": name,
            "paper_reported_test_accuracy": (
                reported_acc if reported_acc is not None else paper.NA
            ),
            "paper_reported_train_accuracy": (
                reported_train if reported_train is not None else paper.NA
            ),
            "paper_reported_roc_auc": (
                reported_auc if reported_auc is not None else paper.NA
            ),
        }
        if name in local.index:
            loc = local.loc[name]
            row.update(
                {
                    "local_test_accuracy": loc["accuracy"],
                    "local_precision": loc["precision"],
                    "local_recall": loc["recall"],
                    "local_f1": loc["f1"],
                    "local_specificity": loc["specificity"],
                    "local_roc_auc": (
                        loc["roc_auc"] if loc["roc_auc"] is not None else paper.NA
                    ),
                }
            )
        else:
            row.update(
                {
                    "local_test_accuracy": paper.NA,
                    "local_precision": paper.NA,
                    "local_recall": paper.NA,
                    "local_f1": paper.NA,
                    "local_specificity": paper.NA,
                    "local_roc_auc": paper.NA,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def classification_report_markdown(results: dict[str, ModelEvaluation], y_true) -> str:
    lines = []
    for name, result in results.items():
        lines.append(f"### {name} ({result.model_type})")
        lines.append("")
        lines.append(
            "```\n"
            + classification_report(
                y_true, result.y_pred, target_names=["no disease (0)", "disease (1)"]
            )
            + "```"
        )
        lines.append("")
    return "\n".join(lines)


def experiment_notes_markdown(results: dict[str, ModelEvaluation], run_meta: dict, y_true) -> str:
    table = comparison_dataframe(results)
    paper_table = paper_comparison_dataframe(results)
    md_table = paper_table.to_markdown(index=False)
    local_md = table.drop(columns=["best_params"]).to_markdown(index=False)
    best = table.iloc[0] if not table.empty else None

    return f"""# Experiment Notes

Generated (UTC): {run_meta['trained_at']}

## Reproduction context

| Item | Value |
| --- | --- |
| Python | {run_meta['library_versions']['python']} |
| scikit-learn | {run_meta['library_versions']['scikit-learn']} |
| pandas | {run_meta['library_versions']['pandas']} |
| NumPy | {run_meta['library_versions']['numpy']} |
| Random seed | {run_meta['random_seed']} |
| Train/test split | {int((1 - run_meta['test_size']) * 100)}% / {int(run_meta['test_size'] * 100)}% stratified |
| Cross-validation | {run_meta['cv_folds']}-fold stratified, scoring = {run_meta['cv_scoring']} |
| Dataset | {run_meta['dataset']['version']} (rows={run_meta['dataset']['rows']}) |
| Dataset SHA-256 | `{run_meta['dataset']['sha256']}` |
| Dataset source | {run_meta['dataset']['source_url']} |

## Target definition

`num == 0` -> class 0 (no heart disease in the dataset),
`num > 0` (original severity categories 1-4) -> class 1 (heart disease present).
Rows with a missing or out-of-range `num` value are dropped
({run_meta['dataset'].get('dropped_rows', 0)} rows dropped in this run).

## Preprocessing

All preprocessing lives inside each saved scikit-learn pipeline and is fitted
only on the training split (no train/test leakage):

- numeric features ({', '.join(config.NUMERIC_FEATURES)}): median imputation + standard scaling
- categorical features ({', '.join(config.CATEGORICAL_FEATURES)}): most-frequent imputation + one-hot encoding (`handle_unknown='ignore'`)

## Models and tuning

{table[['model', 'model_type']].to_markdown(index=False)}

Each model was tuned with `GridSearchCV` ({run_meta['cv_folds']}-fold stratified
CV, scoring `{run_meta['cv_scoring']}`) on the **training split only**; the
held-out test split was never used for tuning.

## Locally measured results (this experiment)

{local_md}

## Comparison with the reference paper

The reference paper ({paper.PAPER['doi']}) reports a Random Forest test accuracy
of **{paper.PAPER['reported_test_accuracy']['random_forest']}%** and ROC-AUC of
**{paper.PAPER['reported_roc_auc']['random_forest']}** on 303 Cleveland instances
with a chi-square-selected 13-feature subset and an 80:20 split.

{md_table}

**IMPORTANT - the reference article is RETRACTED** (see notice below), so its
numbers are used only as an approximate external benchmark:

> {paper.PAPER['retraction_notice']}

### Why the comparison is approximate

- The paper does not publish its exact train/test indices or random seed, so the
  splits are not identical; a different split alone can shift accuracy by several
  percentage points on 303 records.
- The paper applies chi-square feature selection over the full UCI attribute set;
  this project uses all 13 standard Cleveland predictors with one-hot encoding.
- Preprocessing differs (the paper does not document imputation/scaling details).
- Hyperparameter search spaces and CV protocols differ.
- Precision, recall and F1 are **not reported** by the paper and are shown as
  `N/A`; they are never invented.

Best local model this run: **{best['model'] if best is not None else 'n/a'}**
(test accuracy {best['accuracy'] if best is not None else 'n/a'},
ROC-AUC {best['roc_auc'] if best is not None else 'n/a'}).

## Why accuracy alone is insufficient here

Heart-disease screening is asymmetric: a **false negative** (predicted "no
disease" for a sick patient) can delay treatment and lead to severe harm, while
a false positive leads to additional, usually non-invasive, testing. Accuracy
weights both error types equally and hides the behaviour that matters most in a
clinical context, so this experiment reports precision, recall/sensitivity,
specificity, F1 and ROC-AUC alongside accuracy. On a roughly balanced dataset a
model can score high accuracy while missing a dangerous fraction of positives;
recall (sensitivity) makes that failure visible.

## Reproduce this experiment

```bash
docker compose build
docker compose run --rm trainer     # writes /app/artifacts and /app/reports
docker compose up -d app            # Streamlit app at http://localhost:8501
```
"""


def write_all_reports(
    results: dict[str, ModelEvaluation],
    X_test: pd.DataFrame,
    y_true,
    run_meta: dict,
    reports_dir: Path | str | None = None,
) -> list[Path]:
    """Write every evaluation artifact (CSV/JSON/TXT/MD + plots) atomically."""
    from src.visualization import charts

    reports_dir = Path(reports_dir) if reports_dir else config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save_text(name: str, text: str) -> Path:
        path = reports_dir / name
        artifacts.atomic_write_text(path, text)
        written.append(path)
        return path

    table = comparison_dataframe(results)
    _save_text("model_comparison.csv", table.to_csv(index=False))
    _save_text(
        "paper_comparison.csv",
        paper_comparison_dataframe(results).to_csv(index=False),
    )
    _save_text(
        "classification_report.txt",
        "# Classification reports (held-out test set)\n\n"
        + classification_report_markdown(results, y_true),
    )
    metrics_payload = {
        "generated_at": run_meta["trained_at"],
        "random_seed": run_meta["random_seed"],
        "test_size": run_meta["test_size"],
        "dataset": run_meta["dataset"],
        "models": {
            name: {
                "model_type": r.model_type,
                "train_accuracy": round(r.train_accuracy, 4),
                "best_params": r.best_params,
                "cv_best_score": (
                    round(r.cv_best_score, 4) if r.cv_best_score is not None else None
                ),
                "metrics": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.metrics.items()
                },
            }
            for name, r in results.items()
        },
    }
    _save_text("metrics.json", json.dumps(metrics_payload, indent=2) + "\n")
    _save_text(
        "experiment_notes.md", experiment_notes_markdown(results, run_meta, y_true)
    )

    predictions = {
        "y_true": [int(v) for v in np.asarray(y_true)],
        "models": {
            name: {
                "y_pred": [int(v) for v in r.y_pred],
                "y_score": (
                    [float(v) for v in r.y_score] if r.y_score is not None else None
                ),
            }
            for name, r in results.items()
        },
    }
    _save_text("test_predictions.json", json.dumps(predictions, indent=2) + "\n")

    # Plots -----------------------------------------------------------------
    fig = charts.fig_model_comparison(results)
    path = reports_dir / "model_comparison_chart.png"
    charts.save_figure(fig, path)
    written.append(path)

    fig = charts.fig_confusion_matrices(results)
    if fig is not None:
        path = reports_dir / "confusion_matrices.png"
        charts.save_figure(fig, path)
        written.append(path)

    fig = charts.fig_roc_curves(results, y_true)
    if fig is not None:
        path = reports_dir / "roc_curves.png"
        charts.save_figure(fig, path)
        written.append(path)

    fig = charts.fig_class_distribution(y_true)
    path = reports_dir / "dataset_class_distribution.png"
    charts.save_figure(fig, path)
    written.append(path)

    for name, result in results.items():
        fig = charts.fig_feature_importance(result.pipeline, X_test, y_true)
        if fig is None:
            continue
        path = reports_dir / f"feature_importance_{name}.png"
        charts.save_figure(fig, path)
        written.append(path)

    logger.info("Wrote %d report files to %s", len(written), reports_dir)
    return written
