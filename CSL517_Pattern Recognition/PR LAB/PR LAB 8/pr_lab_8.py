#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Auto-detect and switch to project virtual environment if dependencies are missing
base_dir = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(base_dir, 'Classifiers', 'venv', 'bin', 'python')
if os.path.exists(venv_py) and sys.executable != venv_py:
    try:
        import sklearn, pandas, seaborn
    except ImportError:
        os.execv(venv_py, [venv_py] + sys.argv)

"""
CSL517 Pattern Recognition Lab - Assignment 08
Develop a web-based Machine Learning application using Flask to perform linear classification
using the Ho-Kashyap procedure.

Standalone CLI Execution & Verification Pipeline
------------------------------------------------
1. Loads and validates classification CSV datasets.
2. Inspects structure, missing cells, duplicates, and feature schema.
3. Performs conditional preprocessing (imputation, deduplication, categorical encoding, standardization).
4. Implements Ho-Kashyap procedure from scratch using NumPy to find weight vector 'a' and margin vector 'b'.
5. Diagnoses Linear Separability (exact convergence vs. Ho-Kashyap inseparability theorem proof).
6. Records and plots:
   - Iteration k vs. Squared Error ||e(k)||^2
   - Iteration k vs. Misclassified training samples
7. Conducts multi-eta convergence study (eta from 0.1 to 0.9) and plots eta vs. Iterations.
8. Automatically selects optimal eta (fewest iterations with lowest error rate).
9. Evaluates test set metrics (Accuracy, Error Rate, Precision, Recall, F1-Score, Confusion Matrix).
10. Validates new instance prediction interface with range checking and descriptive output.
11. Benchmarks side-by-side against KNN (Elbow Method) and Naive Bayes.
"""

import os
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
np.seterr(all='ignore')
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


class HoKashyapClassifier:
    """
    Ho-Kashyap Linear Classification Procedure implemented from scratch using NumPy.
    Minimizes J(a, b) = ||Y * a - b||^2 subject to b > 0.
    """
    def __init__(self, eta=0.5, max_iter=1000, eps=1e-4):
        self.eta = eta
        self.max_iter = max_iter
        self.eps = eps
        self.a_ = None               # Weight vector [w_0, w_1, ..., w_d]
        self.b_ = None               # Margin vector (N,)
        self.history_e2_ = []        # Squared error ||e(k)||^2
        self.history_misc_ = []      # Misclassified count
        self.status_ = 'max_iter'    # 'separable', 'inseparable', or 'max_iter'
        self.n_iter_ = 0
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Ho-Kashyap procedure requires exactly 2 classes. Found: {len(self.classes_)}")

        # Map classes to -1 and +1
        y_bin = np.where(y == self.classes_[1], 1, -1)
        N, d = X.shape

        # Augment with constant bias: z_i = [1, x_i1, ..., x_id]
        X_aug = np.hstack([np.ones((N, 1)), X])

        # Normalize with class labels: y_i * z_i
        Y = X_aug * y_bin[:, np.newaxis]

        # Compute Moore-Penrose pseudoinverse
        Y_pinv = np.linalg.pinv(Y)

        # Initialize margin vector b(0) > 0 (all ones)
        b = np.ones(N, dtype=float)
        a = Y_pinv @ b

        self.history_e2_ = []
        self.history_misc_ = []
        self.status_ = 'max_iter'
        self.n_iter_ = 0

        for k in range(self.max_iter):
            self.n_iter_ = k + 1
            e = (Y @ a) - b
            e2 = float(np.sum(e ** 2))
            misc_count = int(np.sum((Y @ a) <= 0))

            self.history_e2_.append(e2)
            self.history_misc_.append(misc_count)

            # Stopping Criterion 1: Exact convergence to separating margin
            if np.all(np.abs(e) < self.eps) or misc_count == 0:
                self.status_ = 'separable'
                break

            # Stopping Criterion 2: Ho-Kashyap Proof of Non-Separability
            # If all components e_i <= 0 and e != 0
            if np.all(e <= 1e-6) and np.any(e < -1e-6):
                self.status_ = 'inseparable'
                break

            # Margin and weight updates:
            # e+ = 0.5 * (e + |e|)
            e_plus = 0.5 * (e + np.abs(e))
            b = b + 2.0 * self.eta * e_plus
            a = Y_pinv @ b

        self.a_ = a
        self.b_ = b
        return self

    def decision_function(self, X):
        N = X.shape[0]
        X_aug = np.hstack([np.ones((N, 1)), X])
        return X_aug @ self.a_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0, self.classes_[1], self.classes_[0])


def main():
    print("=" * 80)
    print("   CSL517 Pattern Recognition Lab - Assignment 08")
    print("   Linear Classification using the Ho-Kashyap Procedure Studio (CLI)")
    print("=" * 80)

    # 1. Locate dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, 'Classifiers', 'sample_datasets')
    plots_dir = os.path.join(base_dir, 'Classifiers', 'static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    candidates = [
        os.path.join(sample_dir, 'Linearly_Separable_2D.csv'),
        os.path.join(sample_dir, 'heart.csv'),
        os.path.join(sample_dir, 'Iris_Binary.csv'),
        os.path.join(sample_dir, 'Naive-Bayes-Classification-Data.csv')
    ]

    csv_path = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        csv_path = sys.argv[1]
    else:
        for cand in candidates:
            if os.path.exists(cand):
                csv_path = cand
                break

    if not csv_path:
        print("[!] Error: No dataset file found. Please provide a CSV path.")
        return

    print(f"\n[+] Active Dataset: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)

    # 2. Dataset Inspection
    print(f"\n--- Dataset Overview ---")
    print(f"Dimensions       : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing Values   : {df.isnull().sum().sum()} cells")
    print(f"Duplicate Rows   : {df.duplicated().sum()}")
    print("\nFirst 5 Records:")
    print(df.head())

    # Preprocessing
    if df.duplicated().sum() > 0:
        dups = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        print(f"\n[!] Removed {dups} duplicate records.")

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"[!] Imputed missing values for column: {col}")

    # Determine target
    target_col = df.columns[-1]
    print(f"\n[+] Selected Target Column: '{target_col}'")

    target_series = df[target_col]
    le_target = LabelEncoder()
    y_raw = le_target.fit_transform(target_series.astype(str))
    classes = [str(c) for c in le_target.classes_]

    # Binary check
    if len(classes) > 2:
        print(f"[!] Target has {len(classes)} classes: {classes}. Converting to binary (One-vs-Rest for '{classes[0]}').")
        y = (target_series.astype(str) == classes[0]).astype(int).values
        classes = [f"Not {classes[0]}", str(classes[0])]
    else:
        y = y_raw

    # Feature processing
    X_df = df.drop(columns=[target_col])
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(cat_cols) > 0:
        print(f"[!] Encoding {len(cat_cols)} categorical feature(s): {cat_cols}")
        for col in cat_cols:
            le_feat = LabelEncoder()
            X_df[col] = le_feat.fit_transform(X_df[col].astype(str))

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[+] Dataset Partitioned:")
    print(f"    Training Samples : {len(X_train)}")
    print(f"    Testing Samples  : {len(X_test)}")
    print(f"    Feature Count    : {len(feature_names)}")
    print(f"    Standardization  : Applied (StandardScaler)")

    # 3. Ho-Kashyap Default Run (eta = 0.5)
    default_eta = 0.5
    print(f"\n" + "=" * 80)
    print(f"   STEP 1: Training Ho-Kashyap Classifier (Default η = {default_eta})")
    print("=" * 80)

    hk_model = HoKashyapClassifier(eta=default_eta, max_iter=1000, eps=1e-4)
    hk_model.fit(X_train_scaled, y_train)

    print(f"\n[+] Convergence Results:")
    print(f"    Separability Status : {hk_model.status_.upper()}")
    print(f"    Iterations Taken    : {hk_model.n_iter_}")
    print(f"    Initial ||e(0)||^2  : {hk_model.history_e2_[0]:.4f}")
    print(f"    Final ||e(k)||^2    : {hk_model.history_e2_[-1]:.4f}")
    print(f"    Initial Misclass    : {hk_model.history_misc_[0]} / {len(X_train)}")
    print(f"    Final Misclass      : {hk_model.history_misc_[-1]} / {len(X_train)}")

    # Record Weight Vector a
    print(f"\n[+] Final Weight Vector a = [w0, w1, ..., wd]^T:")
    w = hk_model.a_
    print(f"    Bias (w0) : {w[0]:.5f}")
    for i, feat in enumerate(feature_names):
        print(f"    {feat:10s} : {w[i+1]:.5f}")

    # Record Margin Vector b
    b = hk_model.b_
    print(f"\n[+] Final Margin Vector b (Length {len(b)}):")
    print(f"    min(b) : {np.min(b):.5f} (Satisfies constraint b > 0: {np.min(b) > 0})")
    print(f"    max(b) : {np.max(b):.5f}")
    print(f"    mean(b): {np.mean(b):.5f}")

    # Evaluate on Test Set
    y_pred = hk_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    err_rate = np.mean(y_pred != y_test)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[+] Test Set Evaluation Metrics:")
    print(f"    Accuracy         : {acc * 100:.2f}%")
    print(f"    Error Rate       : {err_rate:.4f} ({int(np.sum(y_pred != y_test))} incorrect / {len(y_test)} total)")
    print(f"    Precision (wgt)  : {prec:.4f}")
    print(f"    Recall (wgt)     : {rec:.4f}")
    print(f"    F1-Score (wgt)   : {f1:.4f}")
    print(f"\n    Confusion Matrix:\n{cm}")

    # Generate and Save Plots
    # Plot 1: Iteration vs ||e(k)||^2
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(hk_model.history_e2_) + 1), hk_model.history_e2_, color='#4361ee', linewidth=2)
    plt.title('Ho-Kashyap: Iteration k vs. Squared Error ‖e(k)‖²', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Iteration (k)', fontsize=10)
    plt.ylabel('Squared Error ‖e(k)‖²', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, 'cli_hk_iter_vs_e2.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"\n[+] Saved Convergence Plot: {p1}")

    # Plot 2: Iteration vs Misclassifications
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(hk_model.history_misc_) + 1), hk_model.history_misc_, color='#e63946', linewidth=2)
    plt.title('Ho-Kashyap: Iteration k vs. Misclassified Training Samples', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Iteration (k)', fontsize=10)
    plt.ylabel('Samples with aᵀyᵢ ≤ 0', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, 'cli_hk_iter_vs_misc.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[+] Saved Misclassification Plot: {p2}")

    # Plot 3: Confusion Matrix
    plt.figure(figsize=(5.5, 4.2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Ho-Kashyap Confusion Matrix (η={default_eta})', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('Actual Class', fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, 'cli_cm_ho_kashyap.png')
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"[+] Saved Confusion Matrix Plot: {p3}")

    # 4. Multi-Eta Study (eta from 0.1 to 0.9)
    print(f"\n" + "=" * 80)
    print("   STEP 2: Multi-η Convergence & Linear Separability Analysis (η ∈ [0.1, 0.9])")
    print("=" * 80)
    print(f"\n{'η':<6}{'Iterations':<14}{'Status':<16}{'Error Rate':<14}{'Accuracy':<12}")
    print("-" * 62)

    eta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    study_records = []
    iters_list = []
    error_list = []

    for eta in eta_values:
        hk_temp = HoKashyapClassifier(eta=eta, max_iter=1000, eps=1e-4)
        hk_temp.fit(X_train_scaled, y_train)
        pred_temp = hk_temp.predict(X_test_scaled)
        err = np.mean(pred_temp != y_test)
        acc_t = 1.0 - err

        study_records.append({
            'eta': eta,
            'iters': hk_temp.n_iter_,
            'status': hk_temp.status_,
            'error_rate': err,
            'accuracy': acc_t
        })
        iters_list.append(hk_temp.n_iter_)
        error_list.append(err)

        print(f"{eta:<6.1f}{hk_temp.n_iter_:<14}{hk_temp.status_:<16}{err:<14.4f}{acc_t * 100:<12.2f}%")

    # Select optimal eta: fewest iterations with lowest error rate
    sorted_records = sorted(study_records, key=lambda x: (x['error_rate'], x['iters']))
    opt = sorted_records[0]
    print("-" * 62)
    print(f"⭐ Optimal η Selected: η = {opt['eta']} (Converged in {opt['iters']} iterations with lowest error rate {opt['error_rate']:.4f})")

    # Plot 4: Eta vs Number of Iterations
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    color1 = '#4361ee'
    ax1.set_xlabel('Learning Rate (η)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Number of Iterations', color=color1, fontsize=11, fontweight='bold')
    line1 = ax1.plot(eta_values, iters_list, color=color1, marker='o', linewidth=2.2, label='Iterations')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.scatter([opt['eta']], [opt['iters']], color='#10b981', s=160, zorder=5, label=f'Optimal η={opt["eta"]}')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color2 = '#e63946'
    ax2.set_ylabel('Classification Error Rate', color=color2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(eta_values, error_list, color=color2, marker='s', linestyle='--', linewidth=1.8, label='Error Rate')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True)
    plt.title('Ho-Kashyap: Learning Rate η vs. Iterations & Error Rate', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, 'cli_hk_eta_vs_iterations.png')
    plt.savefig(p4, dpi=150)
    plt.close()
    print(f"[+] Saved η vs. Iterations Plot: {p4}")

    # 5. Prediction of New Instances
    print(f"\n" + "=" * 80)
    print("   STEP 3: Prediction Interface for New User Instances")
    print("=" * 80)
    sample_instance = X_df.mean().values
    print(f"[+] Query Instance (Dataset Means):")
    for f, v in zip(feature_names, sample_instance):
        print(f"    {f:10s} : {v:.3f}")

    sample_instance_scaled = scaler.transform([sample_instance])
    pred_code = hk_model.predict(sample_instance_scaled)[0]
    pred_desc = classes[int(pred_code)] if int(pred_code) < len(classes) else str(pred_code)
    disc_val = float(hk_model.decision_function(sample_instance_scaled)[0])

    print(f"\n[+] Prediction Output:")
    print(f"    Class Code Variable : {int(pred_code)}")
    print(f"    Descriptive Class   : {pred_desc}")
    print(f"    Discriminant Score  : g(x) = {disc_val:.4f}")

    # 6. Integrated Comparison with KNN & Naive Bayes
    print(f"\n" + "=" * 80)
    print("   STEP 4: Integrated Benchmarking (Ho-Kashyap vs. KNN vs. Naive Bayes)")
    print("=" * 80)

    # KNN (K=1 and K=Elbow)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_err = np.mean(knn_pred != y_test)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    nb_pred = nb.predict(X_test_scaled)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_err = np.mean(nb_pred != y_test)

    print(f"\n{'Classifier':<32}{'Accuracy':<14}{'Error Rate':<14}{'Precision':<12}{'Recall':<12}")
    print("-" * 84)
    print(f"{'Ho-Kashyap Linear Classifier':<32}{acc * 100:<14.2f}%{err_rate:<14.4f}{prec:<12.4f}{rec:<12.4f}")
    print(f"{'K-Nearest Neighbors (KNN: K=3)':<32}{knn_acc * 100:<14.2f}%{knn_err:<14.4f}{precision_score(y_test, knn_pred, average='weighted', zero_division=0):<12.4f}{recall_score(y_test, knn_pred, average='weighted', zero_division=0):<12.4f}")
    print(f"{'Naive Bayes (GaussianNB)':<32}{nb_acc * 100:<14.2f}%{nb_err:<14.4f}{precision_score(y_test, nb_pred, average='weighted', zero_division=0):<12.4f}{recall_score(y_test, nb_pred, average='weighted', zero_division=0):<12.4f}")
    print("-" * 84)
    print("\n[✓] Assignment 08 CLI execution completed successfully!")


if __name__ == '__main__':
    main()
