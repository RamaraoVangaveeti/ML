
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Pokemon Legendary Classification App")

st.write("Upload a CSV dataset (must contain 'Legendary' column as target).")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    (
        "Decision_Tree",
        "KNN",
        "Logistic_Regression",
        "naive_bayes_gaussian",
        "random_forest_classifier",
        "xgboost_classifier"
    )
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Legendary" not in df.columns:
        st.error("Dataset must contain 'Legendary' column as target variable.")
    else:
        X = df.drop("Legendary", axis=1)
        y = df["Legendary"].astype(int)

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols)
            ]
        )

        if model_choice == "Logistic_Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Decision_Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        elif model_choice == "naive_bayes_gaussian":
            model = GaussianNB()
        elif model_choice == "random_forest_classifier":
            model = RandomForestClassifier()
        elif model_choice == "xgboost_classifier":
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"AUC: {auc:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"MCC: {mcc:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
