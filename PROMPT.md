# Project: Heart Disease Risk Prediction Using Machine Learning

## Role

Act as a senior Machine Learning Engineer and Full-Stack Developer. Build a complete, working, research-paper-based Machine Learning mini-project suitable for a university ML practical and project demonstration.

Do not just provide a plan or sample code. Inspect the existing project directory, implement the complete application, run the code, fix errors, and document how to execute it.

## 1. Project Objective

Build a machine learning application that predicts whether a person belongs to a heart-disease-positive or heart-disease-negative class based on clinical measurements.

The application must allow users to enter patient measurements, obtain a model prediction, view prediction confidence where appropriate, and understand the factors that influenced the prediction.

This is an educational demonstration of machine learning, NOT a clinically validated diagnostic tool. Display an appropriate disclaimer and never claim that the application can diagnose heart disease or replace medical advice.

## 2. Research Paper and Dataset

Use the following research paper as the primary reference:

**Paper:** Implementation of a Heart Disease Risk Prediction Model Using Machine Learning

**Paper URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9085310/

The paper reports comparing multiple classification algorithms using heart-disease data.

Dataset:

**UCI Heart Disease Dataset**
https://archive.ics.uci.edu/dataset/45/heart+disease

Use the Cleveland subset, which contains approximately 303 patient records and 13 predictive attributes, excluding the target variable.

The target variable in the commonly used Cleveland data is `num`:
- `0`: No heart disease indicated in the dataset.
- `1–4`: Different positive disease-severity categories in the original dataset.

For the main binary classification experiment, map `num = 0` to class 0 and `num > 0` to class 1.

Dataset requirements:
- Download the dataset from the official UCI repository when possible.
- Prefer the official processed Cleveland dataset.
- Document the source URL, data version, target definition, and feature descriptions.
- Do not fabricate missing records or metrics.
- Handle missing values appropriately.
- If the download fails, provide clear instructions for manually placing the dataset in the project data directory.
- Never silently substitute a different heart-disease dataset.
- If the dataset format or column names differ from expectations, validate them before training.

## 3. Technology Stack

Use a simple, maintainable Python stack:

- Python 3.11 or a compatible installed version
- pandas and NumPy for data processing
- scikit-learn for machine learning
- Matplotlib and Seaborn for visualizations
- Streamlit for the web interface
- Joblib for model persistence
- SHAP or a suitable scikit-learn-based explanation method for model interpretability, if compatible

The application must run on a normal laptop CPU. Do not require a GPU, paid API, cloud service, or external AI API.

Pin or document compatible dependency versions in `requirements.txt`.

## 4. Machine Learning Pipeline

Implement the entire ML pipeline, not just a pretrained model inference interface.

### Data preprocessing

1. Load and inspect the dataset.
2. Display dataset shape, feature types, missing values, and class distribution.
3. Validate target labels and remove invalid target values only according to documented rules.
4. Separate predictors and target.
5. Identify categorical and numerical features correctly.
6. Apply appropriate preprocessing using scikit-learn pipelines and `ColumnTransformer` where practical.
7. Use imputation if required.
8. Apply scaling for models such as Logistic Regression and SVM.
9. Avoid data leakage by fitting all preprocessing steps only on training data.
10. Use a stratified train/test split, such as 80% training and 20% testing, with a fixed random seed.

Document any deviations from the original paper's preprocessing and experimental protocol.

### Models to train

Train and evaluate at least these five models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

Optional: add Gradient Boosting or XGBoost if it can be installed reliably and adds value.

Use appropriate hyperparameter tuning, such as `GridSearchCV` with stratified cross-validation, but avoid unnecessarily expensive searches.

Do not tune models on the held-out test set.

### Evaluation metrics

Calculate the following metrics for every model:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC, when probability or decision scores are available

Also generate:
- Confusion matrix
- ROC curve
- Classification report
- Model comparison bar chart
- Feature importance chart or model explanation

Save evaluation results to CSV and save the trained model pipelines using Joblib.

Explain why accuracy alone is insufficient for medical-risk classification and why false negatives are important.

## 5. Research Paper Comparison

Create a separate research comparison section in the application and project report.

Compare the results from your experiments with the metrics actually reported in the reference paper.

The reference paper reports Random Forest test accuracy of approximately 88.5% and ROC-AUC of approximately 0.92. Verify these values against the paper before using them.

Important rules:
- Clearly distinguish published results from locally measured results.
- Record the evaluation protocol and dataset split used by the paper whenever available.
- If the paper's exact split or preprocessing cannot be reproduced, explicitly state that the comparison is approximate.
- Do not claim that your implementation reproduces the paper exactly unless it does.
- Do not invent precision, recall, F1-score, or any other metric that is not reported in the paper.
- Explain why results can differ due to random splits, preprocessing, feature selection, and hyperparameters.
- Do not artificially manipulate model predictions to achieve the paper's accuracy.

Provide a comparison table with columns for model, paper-reported accuracy, locally measured accuracy, precision, recall, F1-score, and ROC-AUC. Use `N/A` for unavailable published metrics.

## 6. Real-World Use Case

Implement this scenario in the web application:

**Use case: Preliminary Heart Disease Risk Assessment Demonstration**

A user enters a patient's clinical measurements. The application uses the selected trained ML model to classify the record into one of two dataset-defined classes.

The application displays:
- Predicted class: positive or negative according to the dataset's target definition.
- Model probability or score, only when supported by the model.
- A clear explanation that the score is not a calibrated medical probability unless calibration has been evaluated.
- A visualization of relevant features.
- A notice explaining that the output is educational and must not be used for medical decisions.

Include a sample record based on an actual dataset row for demonstration. Label it clearly as a dataset example, not a real-time clinical assessment.

Do not recommend medication, diagnose a patient, or give treatment instructions.

## 7. Streamlit User Interface

Create a polished, professional, responsive Streamlit application.

Use a clean dashboard layout with a sidebar for navigation.

### Page 1: Project Overview
Display:
- Project title and description.
- Research paper details and link.
- Dataset source.
- Project objectives.
- Technologies used.
- Educational-use disclaimer.

### Page 2: Dataset Explorer
Display:
- Dataset preview.
- Dataset dimensions.
- Missing-value summary.
- Target-class distribution.
- Feature descriptions.
- Useful charts for exploring the dataset.

### Page 3: Model Training and Comparison
Display:
- Available models.
- Training and evaluation status.
- A table of model performance metrics.
- A model comparison chart.
- Confusion matrices.
- ROC curves.
- A clearly identified research-paper benchmark.

Training should not repeat unnecessarily on every Streamlit rerun. Use caching where appropriate and persist trained artifacts.

### Page 4: Risk Prediction Demo
Provide an input form for all 13 predictor attributes using understandable labels and sensible ranges derived from the dataset.

The 13 predictors are:
- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

Use readable dropdown labels for categorical variables, including chest-pain type, resting ECG, exercise-induced angina, slope, number of major vessels, and thalassemia category.

Validate all inputs. Ensure the values are transformed into the correct encoding expected by the trained pipeline.

Allow the user to select a trained model, submit the input, and view the predicted class and any supported confidence or score.

Add a button to load an example record from the dataset.

### Page 5: Research and Documentation
Display:
- Reference paper details.
- Dataset citation.
- Methodology.
- Model comparison results.
- Known limitations.
- Ethical considerations.
- Instructions for reproducing the experiment.

## 8. Explainability

Provide an interpretable explanation of model predictions.

Use model feature importances for Random Forest and Decision Tree. For other models, use an appropriate method such as permutation importance or SHAP if it can be integrated reliably.

Distinguish global feature importance from explanations for an individual prediction.

Do not present feature importance as proof of causation or as medical advice.

## 9. Project Structure

Use a clean, feature-oriented organization rather than putting all code in one large file.

Suggested structure:

heart-disease-ml/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── artifacts/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── visualization/
│   │   └── charts.py
│   └── utils/
│       └── config.py
└── tests/
    ├── test_data.py
    ├── test_pipeline.py
    └── test_prediction.py

You may improve the structure if needed, but keep modules focused and maintainable.

Do not commit the dataset if its licensing or redistribution terms are unclear. Do not commit unnecessarily large model files.

## 10. Testing and Validation

Implement tests that verify:

- Dataset loading and target mapping.
- Expected feature count and feature names.
- Preprocessing pipeline fitting and transformation.
- No train/test leakage.
- Every model can be trained.
- Predictions use exactly the expected feature schema.
- Predictions contain only valid class labels.
- Evaluation metrics are computed correctly.
- Saved model artifacts can be reloaded.
- The application can start successfully.

Run the tests and fix errors before considering the project complete.

## 11. Documentation and Final Deliverables

Create a comprehensive README containing:

1. Project abstract.
2. Problem statement.
3. Objectives.
4. Research paper reference.
5. Dataset description.
6. Software and hardware requirements.
7. Installation instructions.
8. Dataset download instructions.
9. Model training commands.
10. Application startup commands.
11. Model evaluation methodology.
12. Results table generated from actual experiments.
13. Research-paper comparison.
14. Screenshots or instructions for capturing screenshots.
15. Limitations and future improvements.
16. References.

Also create:
- `reports/model_comparison.csv`
- `reports/classification_report.txt`
- `reports/experiment_notes.md`
- Saved trained model pipelines in `artifacts/`
- Useful evaluation plots in `reports/`

Do not generate fake results or placeholder accuracy numbers. Generate these files from actual executed experiments.

## 12. Execution Commands

Provide simple commands such as:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.models.train
streamlit run app.py
```

Document the Windows activation equivalent as well.

If training is performed through a separate script, ensure the Streamlit app can load the saved artifacts and clearly explain what to do if they do not exist.

## 13. Final Instructions

Start by inspecting the working directory and the available environment. If an existing project is present, preserve unrelated files and integrate the implementation appropriately.

Then:
1. Implement the project.
2. Download and validate the dataset if network access is available.
3. Train all required models.
4. Generate real evaluation results.
5. Run tests and fix errors.
6. Launch or smoke-test the Streamlit application.
7. Update the README with the actual results and exact commands.
8. Summarize the files created, tests performed, measured model results, and any remaining limitations.

Do not stop after creating files. Execute the implementation and verify it works as far as the available environment permits.

If a step is blocked by unavailable network access, missing dependencies, or execution limitations, document the issue accurately and provide a practical recovery command rather than claiming the step succeeded.

## 14. Docker and Containerized Development (MANDATORY)

The entire Heart Disease Prediction ML mini-project must be developed and executed using Docker containers and persistent Docker volumes.

Do not rely on installing Python dependencies directly on the host machine. The host should only need Docker Engine/Desktop and Docker Compose.

### A. Docker Architecture

Use the following architecture:

- **ML Application Container:** Runs the Streamlit web application.
- **Training Container or Service:** Executes dataset preprocessing, model training, evaluation, and artifact generation. It may share the same Docker image as the application but must support running as a separate service.
- **Persistent Docker Volumes:** Store the dataset, trained models, evaluation results, and experiment reports.
- **Docker Compose:** Orchestrates the services, environment variables, networking, health checks, and volume mounts.

Keep the implementation simple. Do not introduce Kubernetes or unnecessary infrastructure.

### B. Required Docker Files

Create these files in the project root:

```text
Dockerfile
compose.yaml
.dockerignore
.env.example
```

Update the existing `requirements.txt` and `README.md` as necessary.

The Dockerfile must:
- Use a suitable Python base image.
- Install dependencies from `requirements.txt`.
- Set an appropriate working directory.
- Run as a non-root user wherever practical.
- Avoid copying local virtual environments, Git history, cached files, and unnecessary artifacts into the image.
- Expose the Streamlit port.
- Start the application using the correct Streamlit command.
- Use sensible image-layer caching to speed up rebuilds.

### C. Docker Compose Services

Define the following services in `compose.yaml`:

1. `app`
   - Runs the Streamlit application.
   - Publishes port `8501` to the host.
   - Loads trained model artifacts from a persistent volume.
   - Reads the dataset and reports where required.
   - Restarts appropriately unless explicitly stopped.
   - Includes a health check for the application.

2. `trainer`
   - Uses the same project image unless a separate image is genuinely necessary.
   - Runs the training and evaluation pipeline.
   - Must be runnable on demand rather than automatically retraining every time the application starts.
   - Writes trained model pipelines, metrics, charts, and reports to persistent volumes.
   - Exits after training completes successfully.
   - Does not require a published host port.

Do not make the application dependent on the trainer continuously running. The trainer is a finite job; the application should load the artifacts it needs from the shared persistent storage.

Configure a shared Docker network if required. Avoid unnecessary public network exposure.

### D. Required Persistent Volumes

Create named Docker volumes for the following data:

- `heart_dataset_data`: Downloaded or manually supplied dataset files.
- `heart_model_artifacts`: Trained model pipelines and model metadata.
- `heart_experiment_reports`: Evaluation metrics, plots, and research comparison results.

Mount them at clear locations inside the containers, such as:

```text
/app/data
/app/artifacts
/app/reports
```

Use the same corresponding mounts for both `app` and `trainer` where shared access is required.

Requirements:
- Data and trained models must survive normal container recreation and image rebuilds.
- Do not store the only copy of trained models or reports in the container's writable layer.
- Configure appropriate file permissions for the non-root container user.
- Ensure the application and trainer agree on the same paths.
- Do not use `docker compose down -v` in routine instructions because it deletes named volumes.
- Explain how to back up and restore persistent volumes.
- Explain how to inspect volume names and their mount points.

If the dataset is downloaded automatically, save it to the persistent dataset volume. If it must be supplied manually, document the exact procedure for copying it into that volume.

### E. Required Commands

Verify that the following workflow works from a clean checkout.

Build the application image:

```bash
docker compose build
```

Train and evaluate the models:

```bash
docker compose run --rm trainer
```

Start the web application:

```bash
docker compose up -d app
```

Open the application at:

```text
http://localhost:8501
```

View application logs:

```bash
docker compose logs -f app
```

View the status of the services:

```bash
docker compose ps
```

Stop the application without deleting the data:

```bash
docker compose down
```

Rebuild the image after code or dependency changes:

```bash
docker compose build --no-cache
docker compose up -d app
```

The project must remain reproducible after containers are removed and recreated, provided the persistent volumes are retained.

### F. Training and Application Integration

Ensure the application never assumes that training artifacts already exist.

Implement the following behavior:
- If the trained models exist, load them from the artifact volume.
- If models do not exist, display clear instructions for running the trainer service.
- If a model file is corrupt or incompatible, display an understandable error instead of crashing without explanation.
- Save all model evaluation results to the reports volume.
- Use atomic writes where practical so the application does not load partially written artifacts.
- Do not automatically overwrite the current working model during every Streamlit rerun.
- Save model metadata, including model type, feature schema, training timestamp, dataset version, random seed, and evaluation metrics.
- Ensure the application uses the correct preprocessing pipeline and feature ordering associated with each saved model.

### G. Reproducibility and Security

- Pin compatible Python package versions.
- Document the Python and base-image versions.
- Use a fixed random seed for reproducible experiments.
- Never include credentials or secrets in the image.
- Keep `.env` files containing real secrets out of version control.
- Use `.dockerignore` to exclude unnecessary files.
- Run containers as non-root where practical.
- Do not mount the Docker socket into the application container.
- Do not require privileged containers.
- Avoid unnecessary network access at runtime.
- Document that loading untrusted Joblib or pickle model files can execute arbitrary code.

### H. Testing the Docker Setup

Actually test the containerized workflow where the environment allows:

1. Build the image successfully.
2. Run the training service.
3. Confirm that models and reports are written to named volumes.
4. Remove and recreate the containers without removing volumes.
5. Confirm that the application still loads the trained models.
6. Test the prediction interface.
7. Verify that health checks work.
8. Inspect logs for dependency, permission, and missing-artifact errors.
9. Confirm that the dataset, model artifacts, and reports persist after `docker compose down` and a subsequent `docker compose up -d app`.

If Docker is unavailable in the development environment, create the required files and commands, then clearly document which integration tests could not be executed. Do not claim that unexecuted tests passed.

### I. README Updates

Add a dedicated Docker Deployment section covering:

- Architecture and service responsibilities.
- Docker and Docker Compose prerequisites.
- Building the image.
- Downloading or supplying the dataset.
- Running model training.
- Starting the application.
- Viewing logs and troubleshooting errors.
- Persistent volume locations and lifecycle.
- Backing up and restoring volumes.
- Rebuilding images without losing trained models.
- Removing containers and, separately, deleting volumes when intentionally resetting all data.

The final result must be a self-contained, Dockerized ML mini-project that a faculty member can run using documented Docker commands without manually configuring a Python environment on their host machine.

Do not merely create Docker configuration files. Validate the configuration and run the complete workflow wherever Docker and network access are available.