# Heart Disease Risk Prediction Using Machine Learning

*A fully containerized ML mini-project: five classifiers trained on the UCI
Cleveland heart-disease dataset, served through a Streamlit dashboard — all
running inside Docker with persistent volumes. The host needs only Docker and
Docker Compose; no host Python environment is required.*

> ⚠️ **Educational demonstration only.** This application is **not** a
> clinically validated diagnostic tool. It must never be used to diagnose,
> treat, or make medical decisions. Always consult a qualified healthcare
> professional.

---

## 1. Project Abstract

Cardiovascular risk assessment is a classic, well-documented machine-learning
classification problem. This project implements the **entire ML pipeline** —
dataset acquisition and validation, leakage-safe preprocessing, stratified
splitting, cross-validated hyperparameter tuning, evaluation with
medical-context metrics, explainability, and artifact persistence — for five
classical classifiers (Logistic Regression, Decision Tree, Random Forest,
SVM, KNN) on the UCI Cleveland dataset (303 patient records, 13 predictors).
A Streamlit dashboard provides a dataset explorer, a model-comparison view
with a research-paper benchmark, and an interactive prediction demo with
per-prediction explanations. Everything runs in Docker: a finite `trainer`
service writes models/metrics/plots to named volumes, and the `app` service
loads them.

## 2. Problem Statement

Given 13 clinical measurements for a patient (age, chest-pain type, resting
blood pressure, cholesterol, …), classify the record into one of two
classes **as defined by the dataset**:

- `num == 0` → class 0 (no heart disease indicated)
- `num > 0` (severity categories 1–4) → class 1 (heart disease present)

Accuracy alone is misleading for medical-risk screening: a **false negative**
(a sick patient classified as healthy) can delay treatment and cause severe
harm, while a false positive usually leads only to further non-invasive
testing. Therefore every model is evaluated with precision, recall
(sensitivity), specificity, F1 and ROC-AUC alongside accuracy, and all
outputs carry an educational-use disclaimer.

## 3. Objectives

1. Download, validate and document the official UCI Cleveland dataset.
2. Build a leakage-safe scikit-learn preprocessing pipeline (imputation,
   scaling, one-hot encoding) fitted on training data only.
3. Train and tune five classifiers with stratified cross-validation.
4. Evaluate with a complete metric set, confusion matrices and ROC curves.
5. Compare locally measured results with the reference paper's published
   numbers — clearly separating published from measured values.
6. Serve predictions through a polished Streamlit UI with explanations.
7. Package the entire workflow in Docker with persistent volumes so it is
   reproducible from a clean checkout.

## 4. Research Paper Reference

**Karthick K. et al., “Implementation of a Heart Disease Risk Prediction
Model Using Machine Learning,” *Computational and Mathematical Methods in
Medicine*, vol. 2022, article 6517716, 2022.**

- DOI: <https://doi.org/10.1155/2022/6517716>
- Full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9085310/> (PMC9085310)

**Retraction notice:** this article was marked as **retracted** by the
publisher in July 2023 (*Comput Math Methods Med.* 2023;2023:9764021). Its
reported metrics are used in this project **only as an approximate external
benchmark**, with full disclosure, and are never treated as validated ground
truth.

**Metrics actually reported by the paper** (verified against the full text,
Section 5 / Table 3 / abstract):

| Model | Paper train accuracy | Paper test accuracy | Paper ROC-AUC |
| --- | ---: | ---: | ---: |
| Random forest | 100.0% | 88.5% | 0.92 |
| SVM (RBF) | 92.56% | 80.32% | N/A |
| Logistic regression | 85.95% | 80.32% | N/A |
| Gaussian Naive Bayes | 86.77% | 78.68% | N/A |
| LightGBM | 98.76% | 77.04% | N/A |
| XGBoost | 99.58% | 73.77% | N/A |

Precision, recall, F1 and specificity are **not reported** by the paper and
are shown as `N/A` everywhere — they are never invented. The paper's
protocol: 303 instances, chi-square feature selection (13 attributes), 80:20
train/test split (exact indices/seed unpublished).

## 5. Dataset Description

| Item | Value |
| --- | --- |
| Name | UCI Heart Disease — Cleveland processed subset |
| Source page | <https://archive.ics.uci.edu/dataset/45/heart+disease> |
| Direct archive | <https://archive.ics.uci.edu/static/public/45/heart+disease.zip> |
| File used | `processed.cleveland.data` (extracted from the archive) |
| Records | 303 patients |
| Predictors | 13 (see below) |
| Target | `num` (0 = none, 1–4 = severity) → binarised: 0 vs >0 |
| Missing values | `?` markers in `ca` (4) and `thal` (2) → imputed inside the pipeline |
| Version tag | SHA-256 of the file, stored in every model's metadata |

Features: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal` — full descriptions are shown in the app's
Dataset Explorer page.

**No dataset is committed to git.** The trainer downloads it automatically
into the `heart_dataset_data` volume; for manual installation see
[§9.4](#94-supplying-the-dataset-manually).

## 6. Software and Hardware Requirements

**Docker path (recommended, required by the project spec):**

- Docker Engine 24+ or Docker Desktop 4.x with Docker Compose v2
  (tested with Docker 29.x / Compose v5.x)
- ~4 GB free RAM for the build; no GPU required
- Internet access on the first run (base image, pip packages, dataset)

**Versions pinned inside the image:**

- Base image: `python:3.11.10-slim-bookworm` (Python 3.11.10, Debian 12)
- `numpy==1.26.4`, `pandas==2.2.3`, `scikit-learn==1.5.2`,
  `matplotlib==3.9.2`, `seaborn==0.13.2`, `streamlit==1.41.1`,
  `pyarrow==17.0.0`, `joblib==1.4.2`, `tabulate==0.9.0`,
  `pytest==8.3.4`
- Fixed random seed: `42` (configurable via `.env`)

**Optional host-Python path (alternative, not required):** Python 3.11,
`venv`, pip.

## 7. Installation (Docker — recommended)

```bash
# from a clean checkout
docker compose build
```

That is the entire installation. Nothing is installed on the host besides
Docker itself. See [§9](#9-docker-deployment) for the full deployment guide.

<details>
<summary>Optional: local virtualenv instead of Docker</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

</details>

## 8. Quick Start (Docker)

```bash
# 1. Build the image
docker compose build

# 2. Train + evaluate all models (finite job, exits when done)
docker compose run --rm trainer

# 3. Start the web application
docker compose up -d app

# 4. Open the app
#    http://localhost:8501
```

Other everyday commands:

```bash
docker compose logs -f app        # follow application logs
docker compose ps                 # service status + health
docker compose down               # stop containers, KEEP all volumes
docker compose run --rm trainer python -m pytest -q   # run the test suite
```

After code or dependency changes:

```bash
docker compose build --no-cache
docker compose up -d app
```

> ⚠️ **Never run `docker compose down -v` as part of routine workflow** —
> the `-v` flag deletes the named volumes holding your dataset, trained
> models and reports. Use it only when you intentionally want to reset
> everything.

---

## 9. Docker Deployment

### 9.1 Architecture and service responsibilities

```text
                    ┌────────────────────────────────────────────────┐
                    │ docker compose project: heart-disease-ml       │
                    │ network: heart-net (bridge)                    │
                    │                                                │
  host :8501 ──────►│  app (Streamlit)                               │
                    │    - loads models/reports read-only            │
                    │    - restart: unless-stopped                   │
                    │    - healthcheck: /_stcore/health              │
                    │                                                │
                    │  trainer (finite job, on demand, no port)      │
                    │    - downloads dataset, trains, evaluates      │
                    │    - writes artifacts + reports, then exits    │
                    │                                                │
                    │  shared named volumes:                         │
                    │    heart_dataset_data     → /app/data           │
                    │    heart_model_artifacts  → /app/artifacts      │
                    │    heart_experiment_reports → /app/reports      │
                    └────────────────────────────────────────────────┘
```

- **`app`** — runs the Streamlit UI. Publishes only port 8501. It **never**
  trains or overwrites models; it loads whatever the trainer produced.
- **`trainer`** — same image, different command
  (`python -m src.models.train`). Runs only when invoked with
  `docker compose run --rm trainer`, then exits. No published ports.
- **Volumes** — dataset, model pipelines + metadata, and all evaluation
  reports live in named Docker volumes, so they survive `docker compose down`
  and image rebuilds.
- No Docker socket is mounted, no `privileged` mode, no extra services
  (no Kubernetes, no databases).

### 9.2 Prerequisites

- Docker Engine **or** Docker Desktop with Compose v2 (`docker compose …`).
- Verify with:

  ```bash
  docker version
  docker compose version
  ```

### 9.3 Building the image

```bash
docker compose build                 # cached, fast on rebuilds
docker compose build --no-cache      # full rebuild (after base-image changes)
```

The Dockerfile copies `requirements.txt` first and installs dependencies
**before** copying application code, so code edits don't re-run `pip`. The
image runs as a non-root user (`appuser`, uid 1000).

### 9.4 Supplying the dataset

**Automatic (default):** the trainer downloads the official UCI archive into
the `heart_dataset_data` volume on first run and reuses it afterwards.

**Manual (offline/firewalled environments):**

1. Download <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>
   on any machine with internet access.
2. Extract the member `processed.cleveland.data`.
3. Copy it into the named volume:

   ```bash
   docker run --rm -v heart_dataset_data:/data -v "$PWD":/host \
       alpine cp /host/processed.cleveland.data /data/processed.cleveland.data
   ```

4. Run the trainer — it will detect the existing file and skip downloading.

### 9.5 Running model training

```bash
docker compose run --rm trainer
```

This single command downloads/validates the dataset, performs the stratified
80/20 split (seed 42), runs 5-fold `GridSearchCV` per model on the training
split, evaluates on the held-out test split, and atomically writes:

```text
/app/artifacts/models/<model>.joblib        # full pipeline (preprocess + model)
/app/artifacts/metadata/<model>.json        # type, schema, seed, timestamp, metrics
/app/artifacts/model_registry.json          # index of available models
/app/reports/model_comparison.csv           # local metrics
/app/reports/paper_comparison.csv           # published vs measured (N/A where unreported)
/app/reports/classification_report.txt
/app/reports/metrics.json
/app/reports/experiment_notes.md
/app/reports/test_predictions.json
/app/reports/*.png                          # comparison, confusion matrices, ROC, importance
```

Training is **on demand**: starting the app never triggers retraining, and
Streamlit reruns never overwrite saved models.

### 9.6 Starting the application

```bash
docker compose up -d app        # → http://localhost:8501
```

If the trainer has not run yet, every model-dependent page shows clear
instructions instead of crashing.

### 9.7 Logs and troubleshooting

```bash
docker compose logs -f app          # follow app logs
docker compose logs trainer         # trainer output of the last run
docker compose ps                    # status + health (healthy/unhealthy)
```

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| App page: “No trained models available” | Trainer never ran | `docker compose run --rm trainer` |
| App page: model corrupt/incompatible error | Truncated/old artifact | Re-run the trainer (atomic writes prevent this normally) |
| Trainer: dataset download error | No internet in container | Follow [§9.4](#94-supplying-the-dataset-manually) manual copy |
| `healthcheck: unhealthy` | App crashed on start | `docker compose logs app` (look for import/permission errors) |
| Permission errors writing volumes | Stale root-owned volume contents | Re-create volumes intentionally (see §9.9) or `chown` them |
| Port already in use | Another service on 8501 | Set `APP_PORT=8502` in `.env` |

### 9.8 Persistent volumes: locations, lifecycle, inspection

| Volume | Mounted at | Contents |
| --- | --- | --- |
| `heart_dataset_data` | `/app/data` | `processed.cleveland.data` |
| `heart_model_artifacts` | `/app/artifacts` | pipelines, metadata, registry |
| `heart_experiment_reports` | `/app/reports` | CSV/JSON/TXT/MD metrics + plots |

Both services mount all three volumes at identical paths.

**Inspect volume names and host mount points:**

```bash
docker volume ls                                  # list project volumes
docker volume inspect heart_model_artifacts       # includes "Mountpoint"
docker compose config --volumes                   # volumes declared by the project
# list volume contents from a throwaway container:
docker compose run --rm trainer ls -la /app/artifacts /app/reports /app/data
```

On Docker Desktop, the `Mountpoint` path lives inside the Linux VM (visible
with `docker run --rm -v heart_model_artifacts:/v alpine ls /v`); you don't
need host access — use the copy/tar commands below instead.

**Lifecycle rules:**

- `docker compose down` — removes containers/network only. **Data stays.**
- `docker compose build` / `up` after code changes — image layers change,
  **volumes untouched**: trained models survive rebuilds.
- `docker compose down -v` — **deletes the volumes** (all dataset/models/
  reports). Only for a deliberate full reset.

### 9.9 Backing up and restoring volumes

**Backup** (tar the volume contents into `./backups/` on the host):

```bash
mkdir -p backups
docker run --rm -v heart_model_artifacts:/src -v "$PWD/backups":/dst alpine \
    tar czf /dst/heart_model_artifacts.tar.gz -C /src .
docker run --rm -v heart_experiment_reports:/src -v "$PWD/backups":/dst alpine \
    tar czf /dst/heart_experiment_reports.tar.gz -C /src .
docker run --rm -v heart_dataset_data:/src -v "$PWD/backups":/dst alpine \
    tar czf /dst/heart_dataset_data.tar.gz -C /src .
```

**Restore:**

```bash
docker run --rm -v heart_model_artifacts:/dst -v "$PWD/backups":/src alpine \
    tar xzf /src/heart_model_artifacts.tar.gz -C /dst
```

(Restoring into an empty volume re-creates files owned by root; fix ownership
with `alpine sh -c "chown -R 1000:1000 /dst"` if you then run as `appuser`.)

### 9.10 Removing containers vs. resetting data

```bash
docker compose down                     # stop + remove containers/network (DATA KEPT)
docker compose down --rmi local         # also remove the built image (DATA KEPT)
docker compose down -v                  # ⚠ DELETES dataset, models and reports
```

### 9.11 Security notes

- The image contains **no credentials or secrets**; `.env` (if you create one
  from `.env.example`) is git-ignored and needs no secret values.
- Containers run as non-root (`appuser`); no Docker socket mount, no
  `privileged: true`, only port 8501 is published, and no outbound access is
  needed at app runtime (only the trainer downloads the dataset once).
- ⚠️ **Joblib/pickle safety:** loading a model file unpickles Python objects
  and can therefore **execute arbitrary code**. Only ever load artifacts
  produced by this project's own trainer (from your own volumes/backups).
  Never load model files from untrusted sources.

---

## 10. Model Training Commands (summary)

```bash
docker compose run --rm trainer                    # Docker (recommended)
python -m src.models.train                         # local venv alternative
```

## 11. Application Startup Commands (summary)

```bash
docker compose up -d app        # Docker → http://localhost:8501
streamlit run app.py            # local venv alternative
```

## 12. Evaluation Methodology

- **Split:** stratified 80/20 train/test, `random_state=42`.
- **Preprocessing (inside each saved pipeline, fit on train only):**
  numeric → median imputation + standard scaling; categorical →
  most-frequent imputation + one-hot (`handle_unknown='ignore'`).
- **Tuning:** `GridSearchCV`, 5-fold stratified CV, scoring `roc_auc`, small
  deliberate grids; the held-out test set is **never** used for tuning.
- **Metrics:** accuracy, precision, recall/sensitivity, specificity,
  F1, ROC-AUC, confusion matrix, ROC curve, classification report.
- **Importance:** impurity-based importance for tree models; 10-repeat
  permutation importance (ROC-AUC) on the test split for LR/SVM/KNN;
  per-prediction explanations via feature ablation against typical values.
- **Persistence:** atomic writes (temp file + `os.replace`) so the app never
  reads a half-written artifact.

## 13. Results (locally measured, this repository's experiment)

<!-- RESULTS:BEGIN -->
Executed with `docker compose run --rm trainer` on 2026-10-20 (seed 42,
stratified 80/20 split of 303 records → 242 train / 61 test, 5-fold CV on the
training split). Source files: `reports/model_comparison.csv` and
`reports/paper_comparison.csv` (regenerated on every trainer run).

| Model | Train acc. | **Test acc.** | Precision | Recall | Specificity | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8595 | **0.8852** | 0.8387 | 0.9286 | 0.8485 | 0.8814 | 0.9665 |
| KNN | 1.0000 | **0.8852** | 0.8621 | 0.8929 | 0.8788 | 0.8772 | 0.9589 |
| Decision Tree | 0.8430 | **0.8689** | 0.8571 | 0.8571 | 0.8788 | 0.8571 | 0.8712 |
| Random Forest | 0.9256 | **0.8525** | 0.8065 | 0.8929 | 0.8182 | 0.8475 | 0.9502 |
| SVM (RBF) | 0.8554 | **0.8525** | 0.8065 | 0.8929 | 0.8182 | 0.8475 | 0.9545 |

Published vs. locally measured (N/A = not reported by the paper / not trained
locally — never invented):

| Model | Paper test acc. | Local test acc. | Local precision | Local recall | Local F1 | Paper ROC-AUC | Local ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 80.32% | 88.52% | 0.8387 | 0.9286 | 0.8814 | N/A | 0.9665 |
| KNN | N/A | 88.52% | 0.8621 | 0.8929 | 0.8772 | N/A | 0.9589 |
| Decision Tree | N/A | 86.89% | 0.8571 | 0.8571 | 0.8571 | N/A | 0.8712 |
| Random Forest | 88.5% | 85.25% | 0.8065 | 0.8929 | 0.8475 | 0.92 | 0.9502 |
| SVM (RBF) | 80.32% | 85.25% | 0.8065 | 0.8929 | 0.8475 | N/A | 0.9545 |
| Gaussian Naive Bayes | 78.68% | N/A | N/A | N/A | N/A | N/A | N/A |
| LightGBM | 77.04% | N/A | N/A | N/A | N/A | N/A | N/A |
| XGBoost | 73.77% | N/A | N/A | N/A | N/A | N/A | N/A |

Observations (no cherry-picking):

- Locally, Logistic Regression and KNN tie on accuracy (88.52%) with LR having
  the best ROC-AUC (0.9665) and recall (0.9286) — recall matters most here
  because false negatives are the costly error.
- Random Forest reached 85.25% locally vs. the paper's 88.5% (ROC-AUC 0.9502
  vs. 0.92). The comparison is **approximate**: the paper's exact split,
  seed, chi-selected feature set and preprocessing are unpublished, and the
  paper itself is retracted.
- KNN's train accuracy of 1.0 with 88.5% test accuracy illustrates
  overfitting of the distance-weighted variant on this small dataset.
- The paper reports no precision/recall/F1, so those cells stay `N/A`.
<!-- RESULTS:END -->

## 14. Research-Paper Comparison

See [§4](#4-research-paper-reference) for the verified published values and
`reports/paper_comparison.csv` for the machine-generated table. The
comparison is **approximate** because the paper does not publish its exact
split, seed or preprocessing; this is stated in `reports/experiment_notes.md`
and in the app's Research page.

## 15. Testing

```bash
# unit/integration tests inside the container
docker compose run --rm trainer python -m pytest -q
```

Covered: dataset download/validation/target mapping, feature schema,
preprocessing fit and train-only statistics (no leakage), metric correctness,
atomic save → reload → prediction with the expected feature schema,
actionable errors for missing/corrupt/incompatible model files.

Container-level checks (see [§17](#17-execution-log--validation-status)):

```bash
docker compose ps                                   # health status
curl -fsS http://localhost:8501/_stcore/health      # → "ok"
docker compose down && docker compose up -d app     # recreate, volumes kept
```

## 16. Screenshots / how to capture them

1. `docker compose up -d app` and open <http://localhost:8501>.
2. Capture each sidebar page (Overview, Dataset Explorer, Model Training &
   Comparison, Risk Prediction Demo after submitting a prediction,
   Research & Documentation).
3. Capture a terminal screenshot of `docker compose ps` showing `healthy`
   and of `docker compose run --rm trainer` completing.
4. Save them as `docs/screenshots/*.png` if you want to commit them (they
   are optional and git-ignored by default only if added to `.gitignore`).

## 17. Execution log / validation status

All checks below were **actually executed** on 2026-10-20 with Docker Desktop
29.x + Compose v5.x (linux/arm64 LinuxKit VM), from a clean checkout:

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 1 | Image builds | `docker compose build` | ✅ exit 0 (base `python:3.11.10-slim-bookworm`) |
| 2 | Training service | `docker compose run --rm trainer` | ✅ exit 0 — 5 models trained in ~3 s; dataset auto-downloaded to the volume |
| 3 | Artifacts written to named volumes | `ls` inside `/app/artifacts`, `/app/reports` | ✅ 5 `.joblib` + 5 metadata JSON + registry; 15 report files; all owned by `appuser` (uid 1000) |
| 4 | Containers removed & recreated, volumes kept | `docker compose down` → `docker compose up -d app` (performed 4×) | ✅ containers recreated each time, volumes untouched |
| 5 | App loads models after recreation | `docker compose exec app python -c "…"` | ✅ all 5 models listed and loaded from the restored volume |
| 6 | Prediction interface | Streamlit `AppTest` against the **live app container** (`docker compose exec app python -m pytest tests/test_app.py`) | ✅ pages render, prediction button returns a classified result; fresh-volume run verified the "no models → instructions" path |
| 7 | Health checks | `docker compose ps` + `curl http://localhost:8501/_stcore/health` | ✅ status `healthy`, endpoint returns `ok`, homepage HTTP 200 |
| 8 | Log inspection | `docker compose logs app` | ✅ 0 occurrences of error/traceback/permission/module-not-found |
| 9 | Persistence across `down`/`up` | file listing + `sha256sum` before/after | ✅ dataset (sha `a74b7efa…`), models (mtimes unchanged at training time — app reruns never overwrite) and all 15 reports survived |
| 10 | Full test suite (with artifacts) | `docker compose run --rm trainer python -m pytest -q` | ✅ **29 passed** |
| 11 | Full test suite (fresh volumes, no artifacts) | same command before training | ✅ 28 passed, 1 skipped (instruction path asserted instead) |
| 12 | `pip check` inside image | `docker compose run --rm trainer python -m pip check` | ✅ "No broken requirements found." |
| 13 | Backup procedure (README §9.9) | `tar` of all 3 volumes into `./backups/` | ✅ 3 archives created |
| 14 | Deliberate full reset | `docker compose down -v` | ✅ volumes removed; app restarted on fresh volumes showing trainer instructions instead of crashing |
| 15 | Restore procedure (README §9.9) | `tar` back into volumes → restart | ✅ ownership preserved (`appuser:appuser`), all 5 models load again, suite passes, health `ok` |

Bugs found and fixed during validation (recorded for transparency):

1. Loader returned an unnamed headerless frame → `KeyError` on first train.
2. `DataFrame.to_markdown()` needed the missing `tabulate` dependency.
3. Chart temp files used a `.tmp` suffix that matplotlib can't infer a format
   from → explicit `format="png"` + temp cleanup on failure.
4. Unpinned transitive `pyarrow` resolved to a release requiring NumPy ≥ 2 →
   pinned `pyarrow==17.0.0` (caught only because page rendering was tested).
5. Dataset Explorer grouped by a column (`target`) absent from the raw frame.
6. `st.selectbox(index=None)` returned `None` on the prediction form.

Not executed (and therefore **not claimed**): none of the required Docker
integration tests above were skipped — network access and Docker were both
available in this environment. Host-native (non-Docker) execution was not
used for validation, per the Docker-only requirement.

## 18. Limitations and Future Improvements

- 303 records from a single 1988 centre: metrics are noisy, single-split
  variance is high; results need bootstrap confidence intervals.
- No calibration, fairness or external-validation study; scores are not
  calibrated medical probabilities.
- Possible dataset bias (~68% male); performance across demographics unknown.
- The reference paper is retracted — the external benchmark itself is
  questionable.
- Possible improvements: nested cross-validation, calibration curves,
  SHAP-based explanations, threshold tuning for high recall, external
  validation on the other UCI subsets.

## 19. References

1. Karthick K. et al. (2022). *Implementation of a Heart Disease Risk
   Prediction Model Using Machine Learning.* Comput Math Methods Med.
   <https://doi.org/10.1155/2022/6517716> — **retracted** July 2023.
2. UCI Machine Learning Repository, Heart Disease Dataset (id 45).
   <https://archive.ics.uci.edu/dataset/45/heart+disease>
3. Detrano R. et al. (1989). *International application of a new probability
   algorithm for the diagnosis of coronary artery disease.* Am J Cardiol.
4. scikit-learn documentation: pipelines, `ColumnTransformer`,
   `GridSearchCV`, `permutation_importance`.
5. Streamlit documentation: <https://docs.streamlit.io>

## 20. Project Structure

```text
ml_mini_project/
├── app.py                     # Streamlit entry point (sidebar navigation)
├── views/                     # one module per page
│   ├── overview.py
│   ├── dataset_explorer.py
│   ├── model_training.py
│   ├── prediction.py
│   └── research.py
├── src/
│   ├── data/loader.py         # download + validation + target mapping
│   ├── preprocessing/pipeline.py
│   ├── models/train.py        # python -m src.models.train
│   ├── models/evaluate.py     # metrics + reports (atomic writes)
│   ├── models/predict.py      # artifact loading + explanations
│   ├── visualization/charts.py
│   └── utils/                 # config, artifacts, paper metadata
├── tests/                     # pytest suite (runs inside the container)
├── data/                      # volume mount point (dataset README only in git)
├── artifacts/                 # volume mount point (.gitkeep only in git)
├── reports/                   # volume mount point (.gitkeep only in git)
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```
