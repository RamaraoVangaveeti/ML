import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import io

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

BASE = Path(__file__).parent
DROP_CANDIDATES = ["#", "Name", "ID", "Id", "index"]

st.set_page_config(page_title="ML — Classification Models", layout="wide")
st.title("ML — Classification Models")
#st.write("Upload your test dataset to evaluate ML models trained on pokémon data.")

# Download button for Test_Data.csv
test_data_path = Path(__file__).parent / "Test_Data.csv"
if test_data_path.exists():
    with open(test_data_path, 'rb') as f:
        test_data_bytes = f.read()
    st.download_button(
        label="📥 Download Test_Data.csv",
        data=test_data_bytes,
        file_name="Test_Data.csv",
        mime="text/csv"
    )

def load_csv(path: Path):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def build_model(name: str):
    if name == "Decision_Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "Logistic_Regression":
        return LogisticRegression(max_iter=2000, solver="lbfgs")
    if name == "naive_bayes_gaussian":
        return GaussianNB()
    if name == "random_forest_classifier":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    if name == "xgboost_classifier":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not available in the environment")
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    raise ValueError(f"Unknown model {name}")

def prepare_xy(df: pd.DataFrame, target: str):
    df = df.copy()
    for c in DROP_CANDIDATES:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors='ignore')
    if target not in df.columns:
        raise ValueError("Target column not found in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def align_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame, scale: bool = False):
    # one-hot encode categorical features and align columns
    X_train_d = pd.get_dummies(X_train)
    X_test_d = pd.get_dummies(X_test)
    X_train_d, X_test_d = X_train_d.align(X_test_d, fill_value=0, axis=1)
    if scale:
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_scaled = [c for c in X_train_d.columns if any(c.startswith(n) for n in num_cols)]
        if len(num_cols_scaled) > 0:
            scaler = StandardScaler()
            scaler.fit(X_train_d[num_cols_scaled])
            X_train_d[num_cols_scaled] = scaler.transform(X_train_d[num_cols_scaled])
            X_test_d[num_cols_scaled] = scaler.transform(X_test_d[num_cols_scaled])
    return X_train_d, X_test_d

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics["AUC"] = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        metrics["AUC"] = np.nan
    return metrics

def plot_confusion(cm, labels):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# UI
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", [
        "Decision_Tree",
        "KNN",
        "Logistic_Regression",
        "naive_bayes_gaussian",
        "random_forest_classifier",
        "xgboost_classifier",
    ])
    
    uploaded_test = st.file_uploader("Upload Test CSV", type=["csv"])

st.write("## Dataset and Evaluation")

train_path = BASE / "Train_Data.csv"
train_df = load_csv(train_path)
test_df = None

if train_df is None:
    st.error("Training data file (Train_Data.csv) not found.")

# Load test data from uploaded file
if uploaded_test is not None:
    try:
        test_df = pd.read_csv(uploaded_test)
        st.subheader("Test Data — preview")
        st.dataframe(test_df.head())
    except Exception as e:
        st.error(f"Could not read uploaded test CSV: {e}")

target_col = st.text_input("Target column name", value="Legendary")

if st.button("Train & Evaluate"):
    try:
        if train_df is None or test_df is None:
            st.error("Please upload a test CSV file.")
        else:
            X_train, y_train = prepare_xy(train_df, target_col)
            X_test, y_test = prepare_xy(test_df, target_col)
            scale = model_name in ["KNN", "Logistic_Regression"]
            Xtr, Xte = align_and_scale(X_train, X_test, scale=scale)
            clf = build_model(model_name)
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            y_proba = clf.predict_proba(Xte) if hasattr(clf, "predict_proba") else None
            metrics = compute_metrics(y_test, y_pred, y_proba)
            st.subheader("Metrics")
            st.table(pd.DataFrame([metrics]))

            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred, zero_division=0))

            st.subheader("Confusion matrix")
            labels = np.unique(y_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            buf = plot_confusion(cm, labels)
            st.image(buf)
    except Exception as e:
        st.error(f"Error during train/evaluate: {e}")

