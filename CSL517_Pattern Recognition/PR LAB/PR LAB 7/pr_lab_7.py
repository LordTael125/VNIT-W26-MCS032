# -*- coding: utf-8 -*-
"""PR LAB 7
Assignment 07: K-Nearest Neighbors (KNN) Classification & Bayes Classifier Integration
Problem Statement:
Develop a Machine Learning application using Flask / Python to perform classification 
using the K-Nearest Neighbors (KNN) algorithm, analyze the relationship between K 
and classification error rate (Elbow Method), compare K=1 vs. Elbow-point K, 
integrate with Bayes Classifier, and predict class labels for new instances.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    print("=" * 80)
    print("   Pattern Recognition Lab - Assignment 07: KNN & Bayes Classifier Studio")
    print("=" * 80)

    # 1. Locate dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, 'Classifiers', 'sample_datasets')
    
    # Priority order of sample datasets
    candidates = [
        os.path.join(sample_dir, 'heart.csv'),
        os.path.join(sample_dir, 'Iris_Data.csv'),
        os.path.join(sample_dir, 'Naive-Bayes-Classification-Data.csv'),
        os.path.join(base_dir, '..', 'PR LAB 4', 'heart.csv')
    ]
    
    csv_path = None
    for cand in candidates:
        if os.path.exists(cand):
            csv_path = cand
            break

    if not csv_path:
        print("Error: No sample dataset found. Please ensure a .csv file is present.")
        return

    print(f"\n[+] Loading Dataset: {os.path.basename(csv_path)}")
    data = pd.read_csv(csv_path)

    # 2. Dataset Inspection
    print(f"\n--- Dataset Overview ---")
    print(f"Dimensions: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Missing Values: {data.isnull().sum().sum()}")
    print(f"Duplicate Rows: {data.duplicated().sum()}")
    print("\nFirst 5 Records:")
    print(data.head())

    # Pre-processing: deduplication and imputation if needed
    if data.duplicated().sum() > 0:
        dups = data.duplicated().sum()
        data.drop_duplicates(inplace=True)
        print(f"\n[!] Removed {dups} duplicate records.")

    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
            print(f"[!] Imputed missing values for column: {col}")

    # Determine target variable
    target_col = data.columns[-1]
    if 'target' in data.columns:
        target_col = 'target'
    elif 'species' in data.columns:
        target_col = 'species'
    
    print(f"\n[+] Target Variable Selected: '{target_col}'")
    target_counts = data[target_col].value_counts()
    print("Class Distribution:\n", target_counts.to_string())

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(data[target_col].astype(str))
    target_classes = [str(c) for c in le_target.classes_]

    # Prepare features
    X_df = data.drop(columns=[target_col])
    for col in X_df.select_dtypes(exclude=[np.number]).columns:
        le_feat = LabelEncoder()
        X_df[col] = le_feat.fit_transform(X_df[col].astype(str))

    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    # Train / Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n[+] Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

    # Feature Standardization (Crucial for KNN)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    print("[+] Feature standardization applied using StandardScaler (mean=0, variance=1).")

    # ============================================================
    # PHASE 1: Baseline KNN with K = 1
    # ============================================================
    print("\n" + "=" * 50)
    print("   Phase 1: Baseline KNN Classification (K = 1)")
    print("=" * 50)
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train_std, y_train)
    y_pred_k1 = knn1.predict(X_test_std)

    acc_k1 = accuracy_score(y_test, y_pred_k1)
    err_k1 = float(np.sum(y_pred_k1 != y_test) / len(y_test))
    prec_k1 = precision_score(y_test, y_pred_k1, average='weighted', zero_division=0)
    rec_k1 = recall_score(y_test, y_pred_k1, average='weighted', zero_division=0)
    f1_k1 = f1_score(y_test, y_pred_k1, average='weighted', zero_division=0)
    cm_k1 = confusion_matrix(y_test, y_pred_k1)

    print(f"Accuracy:    {acc_k1 * 100:.2f}%")
    print(f"Error Rate:  {err_k1:.4f} ({err_k1 * 100:.2f}%)")
    print(f"Precision:   {prec_k1:.4f}")
    print(f"Recall:      {rec_k1:.4f}")
    print(f"F1-Score:    {f1_k1:.4f}")
    print("\nConfusion Matrix (K = 1):\n", cm_k1)

    # ============================================================
    # PHASE 2: Finding Optimal Value of K (Elbow Method)
    # ============================================================
    print("\n" + "=" * 50)
    print("   Phase 2: Finding Optimal K (Elbow Method 1..40)")
    print("=" * 50)
    max_k = min(40, len(X_train_std) - 1)
    k_range = list(range(1, max_k + 1))
    error_rates = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_std, y_train)
        pred_k = knn.predict(X_test_std)
        error_rate = np.mean(pred_k != y_test)
        error_rates.append(error_rate)

    # Elbow selection (min error rate)
    min_err = min(error_rates)
    optimal_k = k_range[error_rates.index(min_err)]
    print(f"\n[+] Optimal Value of K identified at Elbow Point: K = {optimal_k} (Error Rate = {min_err:.4f})")

    # Plot K vs. Error Rate
    plot_dir = os.path.join(base_dir, 'Classifiers', 'static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'cli_k_vs_error_rate.png')

    plt.figure(figsize=(9, 5))
    plt.plot(k_range, error_rates, color='#4361ee', linestyle='-', linewidth=2,
             marker='o', markersize=6, markerfacecolor='#3a0ca3', markeredgecolor='white',
             label='Classification Error Rate')
    plt.scatter([optimal_k], [min_err], color='#e63946', s=160, zorder=5,
                label=f'Optimal Elbow Point (K={optimal_k})')
    plt.axvline(x=optimal_k, color='#e63946', linestyle='--', alpha=0.6)
    plt.title("Elbow Method: 'K' Value vs. Classification Error Rate", fontsize=13, fontweight='bold')
    plt.xlabel("Values of K (Number of Neighbors)", fontsize=11)
    plt.ylabel("Error Rate (Incorrect / Total Predictions)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved Elbow Plot to: {plot_path}")

    # ============================================================
    # PHASE 3: Retrain with Optimal Elbow K & Comparison
    # ============================================================
    print("\n" + "=" * 50)
    print(f"   Phase 3: Retraining KNN with Optimal Elbow K = {optimal_k}")
    print("=" * 50)
    knn_opt = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_opt.fit(X_train_std, y_train)
    y_pred_opt = knn_opt.predict(X_test_std)

    acc_opt = accuracy_score(y_test, y_pred_opt)
    err_opt = float(np.sum(y_pred_opt != y_test) / len(y_test))
    prec_opt = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    rec_opt = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    f1_opt = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    cm_opt = confusion_matrix(y_test, y_pred_opt)

    print(f"Accuracy:    {acc_opt * 100:.2f}%")
    print(f"Error Rate:  {err_opt:.4f} ({err_opt * 100:.2f}%)")
    print(f"Precision:   {prec_opt:.4f}")
    print(f"Recall:      {rec_opt:.4f}")
    print(f"F1-Score:    {f1_opt:.4f}")
    print(f"\nConfusion Matrix (Optimal K = {optimal_k}):\n", cm_opt)

    # Comparison Table
    print("\n" + "-" * 70)
    print("   Comparative Performance: Baseline K = 1  vs.  Optimal Elbow K")
    print("-" * 70)
    print(f"{'Metric':<15} | {'K = 1':<12} | {'Optimal K (' + str(optimal_k) + ')':<15} | {'Difference':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<15} | {acc_k1:<12.4f} | {acc_opt:<15.4f} | {(acc_opt - acc_k1)*100:+.2f}%")
    print(f"{'Error Rate':<15} | {err_k1:<12.4f} | {err_opt:<15.4f} | {(err_k1 - err_opt)*100:-.2f}% reduced")
    print(f"{'Precision':<15} | {prec_k1:<12.4f} | {prec_opt:<15.4f} | {(prec_opt - prec_k1):+.4f}")
    print(f"{'Recall':<15} | {rec_k1:<12.4f} | {rec_opt:<15.4f} | {(rec_opt - rec_k1):+.4f}")
    print(f"{'F1-Score':<15} | {f1_k1:<12.4f} | {f1_opt:<15.4f} | {(f1_opt - f1_k1):+.4f}")
    print("-" * 70)

    # ============================================================
    # PHASE 4: Integration with Naive Bayes Classifier
    # ============================================================
    print("\n" + "=" * 50)
    print("   Phase 4: Bayes Classifier Benchmark (GaussianNB)")
    print("=" * 50)
    nb = GaussianNB()
    nb.fit(X_train_std, y_train)
    y_pred_nb = nb.predict(X_test_std)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    err_nb = np.mean(y_pred_nb != y_test)
    f1_nb = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)

    print(f"Naive Bayes Accuracy:   {acc_nb * 100:.2f}%")
    print(f"Naive Bayes Error Rate: {err_nb:.4f}")
    print(f"Naive Bayes F1-Score:   {f1_nb:.4f}")

    print("\n--- Classifier Benchmark (KNN vs. Naive Bayes) ---")
    print(f"KNN (Optimal K={optimal_k}) Accuracy: {acc_opt * 100:.2f}% | F1: {f1_opt:.4f}")
    print(f"Naive Bayes Accuracy:            {acc_nb * 100:.2f}% | F1: {f1_nb:.4f}")

    # ============================================================
    # PHASE 5: Prediction on New Instance
    # ============================================================
    print("\n" + "=" * 50)
    print("   Phase 5: Classifying Sample New Instance")
    print("=" * 50)
    sample_vec = X_test[0:1]
    sample_std = scaler.transform(sample_vec)
    pred_code = knn_opt.predict(sample_std)[0]
    pred_label = le_target.inverse_transform([pred_code])[0]

    print(f"Sample Input Vector (first test row): {sample_vec.flatten()[:5]} ...")
    print(f"Predicted Class Code:  {pred_code}")
    print(f"Predicted Class Label: {pred_label}")
    print(f"Ground Truth Label:    {le_target.inverse_transform([y_test[0]])[0]}")
    print("\n[+] Verification and automated benchmark complete!")

if __name__ == '__main__':
    main()
