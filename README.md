# Assignment 2 — Pokemon Legendary Classification

This README is formatted for the assignment submission and PDF export. It contains the required sections: (a) Problem statement, (b) Dataset description, (c) Models used, a comparison table with evaluation metrics for all six models, and observations.

**a. Problem statement**

- Goal: Predict whether a Pokemon is Legendary (target column `Legendary`) using the provided dataset. Train and compare six classifiers using standard evaluation metrics.

**b. Dataset description**

- Source: `Pokemon.csv` (repository root).
- Reproducible splits created by the project: `Train_Data.csv` (640 rows) and `Test_Data.csv` (160 rows).
- Target variable: `Legendary` (boolean). Training distribution: False = 585, True = 55.
- Features: numeric stats (HP, Attack, Defense, etc.) and categorical attributes (`Type 1`, `Type 2`). Identifier columns (e.g., `#`, `Name`) are dropped during preprocessing.

**c. Models used**

- Logistic Regression
- Decision Tree
- kNN (K-Nearest Neighbors)
- Gaussian Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

Comparison Table — evaluation metrics (from `results/metrics.csv`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9453125 | 0.9703389831 | 0.6666666667 | 0.6 | 0.6315789474 | 0.6031064246 |
| Decision Tree | 0.9375 | 0.8288135593 | 0.5833333333 | 0.7 | 0.6363636364 | 0.6054818092 |
| kNN | 0.9375 | 0.8288135593 | 1.0 | 0.2 | 0.3333333333 | 0.4327835340 |
| Naive Bayes | 0.390625 | 0.5322033898 | 0.0853658537 | 0.7 | 0.1521739130 | 0.0360235697 |
| Random Forest (Ensemble) | 0.953125 | 0.9805084746 | 0.8333333333 | 0.5 | 0.625 | 0.6240673323 |
| XGBoost (Ensemble) | 0.953125 | 0.9864406780 | 0.8333333333 | 0.5 | 0.625 | 0.6240673323 |

Observations on model performance (tabular)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC and balanced precision/recall — reliable and interpretable. |
| Decision Tree | Competitive accuracy and F1; AUC lower than ensembles. May benefit from pruning or tuning to reduce variance. |
| kNN | High precision but very low recall — predicts positive class conservatively; sensitive to scaling and class imbalance. |
| Naive Bayes | Poor accuracy/AUC; likely affected by correlated features violating independence assumptions. High recall but extremely low precision. |
| Random Forest (Ensemble) | Best overall trade-off (highest accuracy and high AUC). Handles interactions and imbalance better. |
| XGBoost (Ensemble) | Very similar to Random Forest with slightly higher AUC; strong performer when tuned. |

Notes and recommendations

- The dataset is imbalanced (few Legendary examples). Consider resampling (SMOTE), using `class_weight`, or threshold tuning to improve minority-class performance.
- Perform hyperparameter tuning (GridSearchCV/RandomizedSearchCV) with cross-validation for stable model selection.
- For reproducibility include `Train_Data.csv` and `Test_Data.csv` and the `results/metrics.csv` and `results/cm_*.png` images in your submission.

Files to include in submission

- `StreamLit_App.py` — interactive app for model evaluation.
- `evaluate_models.py` — trains all models and saves `results/metrics.csv` and confusion matrices to `results/`.
- `Train_Data.csv`, `Test_Data.csv` — reproducible splits.
- `results/metrics.csv`, `results/cm_*.png` — evaluation outputs.

Next steps

- I ran the automated evaluation and updated this README with the exact metrics from `results/metrics.csv`.
- Would you like me to (1) generate a formatted PDF from this README, (2) commit & push the README and results to a remote, or (3) add additional plots/analysis to the report? Reply with the number or combination you prefer.

