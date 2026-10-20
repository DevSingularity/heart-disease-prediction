"""Reference-paper metadata and actually published metrics.

Verified directly against the full text of the reference article
(PMC9085310).  Only metrics that the paper actually reports are stored here;
everything else must be rendered as ``N/A`` and must never be invented.
"""

from __future__ import annotations

PAPER: dict = {
    "title": (
        "Implementation of a Heart Disease Risk Prediction Model "
        "Using Machine Learning"
    ),
    "authors": "K. Karthick, S. K. Aruna, R. Samikannu, R. Kuppusamy, "
    "Y. Teekaraman, A. R. Thelkar",
    "journal": "Computational and Mathematical Methods in Medicine",
    "year": 2022,
    "doi": "10.1155/2022/6517716",
    "pmcid": "PMC9085310",
    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9085310/",
    # The article was formally retracted on 2023-07-19; this is documented
    # prominently in the README and in the application.
    "retracted": True,
    "retraction_notice": (
        "As of July 2023 this article is marked as RETRACTED by the publisher "
        "(Comput Math Methods Med. 2023;2023:9764021). Its reported numbers are "
        "used here only as an approximate external benchmark, with full "
        "disclosure, and are not treated as validated ground truth."
    ),
    "protocol": (
        "Cleveland heart-disease dataset, 303 instances; 13 attributes selected "
        "with a chi-square (SelectKBest) test out of the full attribute set; "
        "80:20 train/test split; target values 1-4 collapsed to class 1."
    ),
    # Test-set accuracy (%), Table 3 + abstract/conclusion of the paper.
    "reported_test_accuracy": {
        "svm": 80.32,
        "gaussian_nb": 78.68,
        "logistic_regression": 80.32,
        "lightgbm": 77.04,
        "xgboost": 73.77,
        "random_forest": 88.5,
    },
    # Training-set accuracy (%), Table 3 of the paper.
    "reported_train_accuracy": {
        "svm": 92.56,
        "gaussian_nb": 86.77,
        "logistic_regression": 85.95,
        "lightgbm": 98.76,
        "xgboost": 99.58,
        "random_forest": 100.0,
    },
    # ROC-AUC: only reported for random forest (Section 5).
    "reported_roc_auc": {"random_forest": 0.92},
    # Models trained locally that the paper never evaluated.
    "models_not_in_paper": ["decision_tree", "knn"],
    # Paper models not trained locally (documented, never fabricated).
    "models_not_trained_locally": ["gaussian_nb", "lightgbm", "xgboost"],
    "missing_metric_note": (
        "The paper reports only train/test accuracy and, for random forest, "
        "ROC-AUC. Precision, recall, F1 and specificity are not published "
        "and are therefore recorded as N/A."
    ),
}

NA = "N/A"
