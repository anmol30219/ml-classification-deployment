"""
=============================================================================
Course: BITS Pilani M.Tech (AIML) - Machine Learning
Assignment: Assignment 2 (Classification & Streamlit Cloud Deployment)
Author: Anmol Ahlawat
Module: train.py (End-to-End Training, Cross-Validation & Model Serialization)
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def build_and_evaluate_pipelines():
    # 1. Load Dataset
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    
    # Feature columns (30) and Target column (1)
    feature_cols = cancer.feature_names
    target_col = 'target'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("=" * 80)
    print(f"Dataset Successfully Loaded: UCI Breast Cancer Wisconsin (Diagnostic)")
    print(f"Total Instances: {X.shape[0]} | Total Morphological Features: {X.shape[1]}")
    print(f"Target Distribution: Benign (0) = {np.sum(y == 0)}, Malignant (1) = {np.sum(y == 1)}")
    print("=" * 80)

    # 2. Stratified Holdout Partition (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3. Export Holdout Test Partition for Streamlit App Testing
    os.makedirs("model", exist_ok=True)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("test_data.csv", index=False)
    print(f"Holdout test partition saved to 'test_data.csv' (Shape: {test_df.shape})")

    # 4. Define Model Pipelines
    # Note: StandardScaler is bundled inside the pipeline to prevent train-test contamination.
    classifiers = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            # Decision trees are invariant to monotonic monotonic feature scaling
            ('clf', DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42))
        ]),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
        ]),
        "Naive Bayes": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
        ])
    }

    model_filenames = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/svm.joblib"
    }

    # 5. Model Training, 5-Fold Stratified Cross-Validation & Holdout Evaluation
    results = []
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 95)
    print(f"{'Model Evaluation Matrix (Holdout Test Set : N=114)':^95}")
    print("=" * 95)

    for name, pipeline in classifiers.items():
        # Fit on training split
        pipeline.fit(X_train, y_train)

        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate assignment metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

        # Serialize model pipeline to disk
        joblib.dump(pipeline, model_filenames[name])
        print(f"[SERIALIZED] {name:<22} -> {model_filenames[name]}")

    # 6. Display Comparison Table
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    print("=" * 95)
    print("All models successfully trained, evaluated, and saved to './model/'.")

if __name__ == "__main__":
    build_and_evaluate_pipelines()