import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def load_and_prepare(path):
    df = pd.read_csv(path)
    drop_cols = ["#", "Name", "ID", "Id", "index"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if "Legendary" not in df.columns:
        raise ValueError("Expected target column 'Legendary' in dataset")
    y = df["Legendary"].astype(int)
    X = df.drop(columns=["Legendary"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def make_models(n_classes=2):
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=2000),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes_gaussian": GaussianNB(),
        "random_forest_classifier": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost_classifier"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    else:
        print("Warning: xgboost not available; skipping xgboost_classifier")
    return models


def compute_metrics(y_true, y_pred, y_prob=None):
    res = {}
    res["Accuracy"] = accuracy_score(y_true, y_pred)
    res["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    res["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    res["F1"] = f1_score(y_true, y_pred, zero_division=0)
    res["MCC"] = matthews_corrcoef(y_true, y_pred)
    try:
        if y_prob is not None:
            res["AUC"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            res["AUC"] = np.nan
    except Exception:
        res["AUC"] = np.nan
    return res


def plot_cm(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    data_path = "Train_Data.csv"
    X, y = load_and_prepare(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = make_models()
    results = []
    os.makedirs("results", exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)
        cm = confusion_matrix(y_test, y_pred)
        plot_cm(cm, labels=["0", "1"], outpath=f"results/cm_{name}.png")

    df_res = pd.DataFrame(results).set_index("Model")
    df_res = df_res[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    df_res.to_csv("results/metrics.csv")
    print("Saved metrics to results/metrics.csv")
    print(df_res)


if __name__ == "__main__":
    main()
