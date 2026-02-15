import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


st.set_page_config(page_title="Pokemon Legendary Classifier", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "Pokemon.csv")
TARGET_COL = "Legendary"
DROP_COLS = ["#", "Name"]


def make_preprocessor(df: pd.DataFrame, dense: bool = False, scale_numeric: bool = False) -> ColumnTransformer:
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def get_model(model_name: str):
    if model_name == "Logistic_Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced"), dict(dense=False, scale_numeric=True)
    if model_name == "Decision_Tree":
        return DecisionTreeClassifier(random_state=42, class_weight="balanced"), dict(dense=False, scale_numeric=False)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=7), dict(dense=False, scale_numeric=True)
    if model_name == "naive_bayes_gaussian":
        return GaussianNB(), dict(dense=True, scale_numeric=True)
    if model_name == "random_forest_classifier":
        return RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced_subsample"), dict(dense=False, scale_numeric=False)
    if model_name == "xgboost_classifier":
        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=4,
        ), dict(dense=False, scale_numeric=False)
    raise ValueError("Unknown model")


def compute_metrics(y_true, y_pred, y_proba=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else None,
    }
    return out


@st.cache_data(show_spinner=False)
def load_train_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_pipeline(model_name: str):
    df = load_train_data()
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, prep_cfg = get_model(model_name)
    preprocessor = make_preprocessor(df, **prep_cfg)

    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    # Evaluate on validation split (for display)
    y_pred = pipe.predict(X_valid)
    y_proba = pipe.predict_proba(X_valid)[:, 1] if hasattr(pipe, "predict_proba") else None
    metrics = compute_metrics(y_valid, y_pred, y_proba)
    cm = confusion_matrix(y_valid, y_pred)

    return pipe, metrics, cm, classification_report(y_valid, y_pred, zero_division=0)


st.title("🧪 Pokemon Legendary Classification (Streamlit)")
st.caption("Train-on-load app: models are trained on Pokemon.csv and evaluated on a held-out validation split.")

model_name = st.selectbox(
    "Select model",
    ["Decision_Tree", "KNN", "Logistic_Regression", "naive_bayes_gaussian", "random_forest_classifier", "xgboost_classifier"],
)

pipe, val_metrics, val_cm, val_report = train_pipeline(model_name)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Validation metrics (held-out split)")
    st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in val_metrics.items()})

with col2:
    st.subheader("Validation confusion matrix")
    st.write("Rows = true, columns = predicted (0=Not Legendary, 1=Legendary)")
    st.dataframe(pd.DataFrame(val_cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

st.subheader("Validation classification report")
st.code(val_report)

st.divider()
st.subheader("Upload a test dataset (CSV) for scoring")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    test_df = pd.read_csv(uploaded)

    # If the uploaded file contains the target column, compute metrics. Otherwise only show predictions.
    has_target = TARGET_COL in test_df.columns

    X_test = test_df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in test_df.columns], errors="ignore")
    y_pred = pipe.predict(X_test)

    st.write("Sample predictions:")
    preview = test_df.copy()
    preview["Predicted_Legendary"] = y_pred.astype(int)
    st.dataframe(preview.head(20))

    if has_target:
        y_true = test_df[TARGET_COL].astype(int).values
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        test_metrics = compute_metrics(y_true, y_pred, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)

        st.subheader("Uploaded test-set metrics")
        st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in test_metrics.items()})

        st.subheader("Uploaded test-set confusion matrix")
        st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

        st.subheader("Uploaded test-set classification report")
        st.code(report)
    else:
        st.info(f"'{TARGET_COL}' column not found in uploaded CSV. Metrics/confusion matrix require true labels.")
