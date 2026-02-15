# Assignment 2: Pokemon Legendary Classification

---

## a. Problem Statement

Develop and evaluate six machine learning classifiers to predict whether a Pokémon is Legendary using the provided dataset. Calculate and compare performance metrics across all models.

---

## b. Dataset Description

| Attribute | Details |
|-----------|---------|
| **Source** | `Pokemon.csv` |
| **Train Split** | `Train_Data.csv` (640 rows) |
| **Test Split** | `Test_Data.csv` (160 rows) |
| **Target Variable** | `Legendary` (Boolean) |
| **Target Distribution (Train)** | False: 585, True: 55 |
| **Features** | Numeric stats (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed) and categorical attributes (Type 1, Type 2) |
| **Preprocessing** | Identifier columns (#, Name, ID) removed; categorical features one-hot encoded |

---

## c. Models Used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## d. Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9453 | 0.9703 | 0.6667 | 0.6000 | 0.6316 | 0.6031 |
| Decision Tree | 0.9375 | 0.8288 | 0.5833 | 0.7000 | 0.6364 | 0.6055 |
| kNN | 0.9375 | 0.8288 | 1.0000 | 0.2000 | 0.3333 | 0.4328 |
| Naive Bayes | 0.3906 | 0.5322 | 0.0854 | 0.7000 | 0.1522 | 0.0360 |
| Random Forest (Ensemble) | 0.9531 | 0.9805 | 0.8333 | 0.5000 | 0.6250 | 0.6241 |
| XGBoost (Ensemble) | 0.9531 | 0.9864 | 0.8333 | 0.5000 | 0.6250 | 0.6241 |

---

## e. Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Strong baseline with high AUC (0.97) and balanced precision/recall. Reliable and interpretable; excellent generalization. |
| **Decision Tree** | Competitive accuracy (0.9375) and F1 score; lower AUC (0.8288) indicates potential overfitting. Tuning max depth or pruning could improve performance. |
| **kNN** | High precision (1.0) but very low recall (0.2) — overly conservative in predicting the minority class. Sensitive to feature scaling and class imbalance; k-value tuning needed. |
| **Naive Bayes** | Poorest performer (Accuracy 0.3906, AUC 0.5322). Independence assumption violated by correlated features. High recall but extremely low precision; unsuitable for this dataset. |
| **Random Forest (Ensemble)** | Best trade-off: highest accuracy (0.9531), high AUC (0.9805), strong MCC. Handles feature interactions and class imbalance well. Robust ensemble approach. |
| **XGBoost (Ensemble)** | Very similar to Random Forest with slightly higher AUC (0.9864). Excellent performance when tuned; strong gradient boosting approach for this classification task. |

---

## f. Key Findings

- **Ensemble methods** (Random Forest, XGBoost) significantly outperform simpler models on accuracy and AUC.
- **Class imbalance** affects minority-class prediction; kNN and Naive Bayes struggle particularly.
- **Feature scaling** is critical for distance-based and probabilistic models.
- **Logistic Regression** provides an excellent interpretable baseline with strong performance.

---

## g. Recommendations

1. Use **ensemble methods** (Random Forest or XGBoost) for production deployment.
2. Address **class imbalance** via SMOTE, class weights, or threshold adjustment for better minority-class recall.
3. Perform **hyperparameter tuning** using GridSearchCV and k-fold cross-validation.
4. Consider **feature engineering** to improve Naive Bayes and kNN performance.

---

## h. Project Files

| File | Purpose |
|------|---------|
| `README.md` | This report |
| `StreamLit_App.py` | Interactive evaluation dashboard |
| `evaluate_models.py` | Automated training and metrics generation |
| `Train_Data.csv` | Training dataset (640 rows) |
| `Test_Data.csv` | Test dataset (160 rows) |
| `results/metrics.csv` | Consolidated evaluation metrics |
| `results/cm_*.png` | Confusion matrix images for all models |

---

**Generated:** February 15, 2026  
**Status:** Ready for submission

