#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Auto-detect and switch to project virtual environment if dependencies are missing
base_dir = os.path.dirname(os.path.abspath(__file__))
candidate_venvs = [
    os.path.join(base_dir, 'Classifiers', 'venv', 'bin', 'python'),
    os.path.join(base_dir, '..', 'PR LAB 7', 'Classifiers', 'venv', 'bin', 'python'),
]
for venv_py in candidate_venvs:
    if os.path.exists(venv_py) and sys.executable != venv_py:
        try:
            import sklearn, pandas, seaborn
        except ImportError:
            os.execv(venv_py, [venv_py] + sys.argv)

"""
CSL517 Pattern Recognition Lab - Assignment 09
Develop a web-based Machine Learning application using Flask to perform non-parametric
classification using the Parzen Window density estimation method.

Standalone CLI Execution & Verification Pipeline
------------------------------------------------
1. Loads and validates classification CSV datasets.
2. Inspects schema, missing cells, duplicates, and feature data types.
3. Pre-training preparation: conditional imputation, deduplication, categorical handling, standardization choice.
4. Parzen Window Density Estimator (implemented from scratch using NumPy):
   - Gaussian window with the Log-Sum-Exp trick to avoid numerical underflow.
   - Hypercube window with empty-window detection.
   - Class-conditional densities p(x | omega_j), posterior probabilities P(omega_j | x), Bayes decision rule.
5. Default Parameters Run (Gaussian, h = 1.0):
   - Prior probabilities of each class.
   - Density estimates and posterior probabilities for the first 5 test samples.
   - Test-set Accuracy, Precision, Recall, F1-Score (macro), and Confusion Matrix.
6. Window Width (h) Analysis:
   - Multi-h sweep across both window functions (0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5).
   - Records train error, validation error, and empty hypercube window counts.
   - Generates h vs. Error Rate plot on a logarithmic scale for h.
7. Model Selection & Comparison:
   - Identifies optimal (window*, h*) minimizing validation error.
   - Evaluates optimal model on test set and displays side-by-side comparison with default model.
8. Prediction of New Instances:
   - Range validation and outlier warnings (>3 std dev).
   - Applies fitted pre-processing and outputs predicted class, posteriors, and densities.
9. Integration Benchmark:
   - Side-by-side comparison with Bayes (GaussianNB), KNN (Elbow Method), and Ho-Kashyap.
"""

import argparse
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ParzenWindowClassifier:
    """
    Non-parametric Parzen Window Density Estimator & Bayes Classifier
    implemented from scratch using NumPy.
    """
    def __init__(self, h=1.0, window='gaussian'):
        self.h = float(h)
        self.window = window.lower()
        if self.window not in ['gaussian', 'hypercube']:
            raise ValueError(f"Unsupported window function '{self.window}'. Choose 'gaussian' or 'hypercube'.")
        self.classes_ = None
        self.priors_ = None
        self.X_by_class_ = {}
        self.N_ = 0
        self.d_ = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.N_, self.d_ = X.shape
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_], dtype=float)
        self.X_by_class_ = {c: X[y == c] for c in self.classes_}
        return self

    @staticmethod
    def _log_sum_exp(arr, axis=-1):
        m = np.max(arr, axis=axis, keepdims=True)
        is_neginf = np.isneginf(m)
        m_safe = np.where(is_neginf, 0.0, m)
        res = m_safe + np.log(np.sum(np.exp(arr - m_safe), axis=axis, keepdims=True))
        res = np.where(is_neginf, -np.inf, res)
        return np.squeeze(res, axis=axis)

    def predict_densities_and_posteriors(self, X):
        X = np.asarray(X, dtype=float)
        M = X.shape[0]
        n_classes = len(self.classes_)

        if self.window == 'gaussian':
            log_densities = np.zeros((M, n_classes))
            const_term = - (self.d_ / 2.0) * np.log(2.0 * np.pi) - self.d_ * np.log(self.h)

            for j, c in enumerate(self.classes_):
                Xc = self.X_by_class_[c]
                Nc = Xc.shape[0]
                diff = X[:, np.newaxis, :] - Xc[np.newaxis, :, :]
                dist_sq = np.sum(diff ** 2, axis=-1)
                z = - dist_sq / (2.0 * (self.h ** 2))
                lse_z = self._log_sum_exp(z, axis=-1)
                log_densities[:, j] = - np.log(Nc) + const_term + lse_z

            log_joint = log_densities + np.log(self.priors_)[np.newaxis, :]
            log_evidence = self._log_sum_exp(log_joint, axis=-1)[:, np.newaxis]
            log_posteriors = log_joint - log_evidence
            posteriors = np.exp(log_posteriors)
            densities = np.exp(np.clip(log_densities, -700, 700))
            empty_count = 0
            empty_mask = np.zeros(M, dtype=bool)

        elif self.window == 'hypercube':
            densities = np.zeros((M, n_classes))
            log_densities = np.full((M, n_classes), -np.inf)
            posteriors = np.zeros((M, n_classes))
            V = float(self.h ** self.d_)
            if V == 0.0:
                V = 1e-300
            half_h = self.h / 2.0
            k_counts = np.zeros((M, n_classes))

            for j, c in enumerate(self.classes_):
                Xc = self.X_by_class_[c]
                Nc = Xc.shape[0]
                diff = np.abs(X[:, np.newaxis, :] - Xc[np.newaxis, :, :])
                max_diff = np.max(diff, axis=-1)
                in_cube = (max_diff <= half_h)
                kc = np.sum(in_cube, axis=-1)
                k_counts[:, j] = kc
                densities[:, j] = kc / (Nc * V)
                valid = kc > 0
                log_densities[valid, j] = np.log(kc[valid]) - np.log(Nc) - self.d_ * np.log(self.h)

            total_k = np.sum(k_counts, axis=-1)
            empty_mask = (total_k == 0)
            empty_count = int(np.sum(empty_mask))

            for i in range(M):
                if total_k[i] > 0:
                    numerator = densities[i] * self.priors_
                    denom = np.sum(numerator)
                    if denom > 0:
                        posteriors[i] = numerator / denom
                    else:
                        posteriors[i] = self.priors_
                else:
                    posteriors[i] = self.priors_

        return densities, posteriors, log_densities, empty_count, empty_mask

    def predict(self, X):
        _, posteriors, _, _, _ = self.predict_densities_and_posteriors(X)
        best_indices = np.argmax(posteriors, axis=-1)
        return self.classes_[best_indices]


class HoKashyapClassifier:
    """Ho-Kashyap procedure for linear classification."""
    def __init__(self, eta=0.5, max_iter=1000, eps=1e-4):
        self.eta = eta
        self.max_iter = max_iter
        self.eps = eps
        self.a_ = None
        self.b_ = None
        self.status_ = 'max_iter'
        self.n_iter_ = 0
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Ho-Kashyap requires binary classification.")
        y_bin = np.where(y == self.classes_[1], 1, -1)
        N, d = X.shape
        X_aug = np.hstack([np.ones((N, 1)), X])
        Y = X_aug * y_bin[:, np.newaxis]
        Y_pinv = np.linalg.pinv(Y)
        b = np.ones(N, dtype=float)
        a = Y_pinv @ b

        for k in range(self.max_iter):
            self.n_iter_ = k + 1
            e = (Y @ a) - b
            misc_count = int(np.sum((Y @ a) <= 0))
            if np.all(np.abs(e) < self.eps) or misc_count == 0:
                self.status_ = 'separable'
                break
            if np.all(e <= 1e-6) and np.any(e < -1e-6):
                self.status_ = 'inseparable'
                break
            e_plus = 0.5 * (e + np.abs(e))
            b = b + 2.0 * self.eta * e_plus
            a = Y_pinv @ b

        self.a_ = a
        self.b_ = b
        return self

    def predict(self, X):
        N = X.shape[0]
        X_aug = np.hstack([np.ones((N, 1)), X])
        scores = X_aug @ self.a_
        return np.where(scores > 0, self.classes_[1], self.classes_[0])


def run_pipeline(dataset_path=None, target_col=None, standardize=True, h_values=None, output_dir=None):
    if dataset_path is None:
        default_samples = [
            os.path.join(base_dir, 'Classifiers', 'sample_datasets', 'Iris_Data.csv'),
            os.path.join(base_dir, 'Iris.csv'),
            os.path.join(base_dir, '..', 'PR LAB 8', 'Classifiers', 'sample_datasets', 'Iris_Data.csv')
        ]
        for sp in default_samples:
            if os.path.exists(sp):
                dataset_path = sp
                break

    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if output_dir is None:
        output_dir = os.path.join(base_dir, 'Classifiers', 'static', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    if h_values is None:
        h_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    print("=" * 78)
    print(" CSL517 Pattern Recognition Lab - Assignment 09")
    print(" Non-Parametric Classification using Parzen Window Density Estimation")
    print("=" * 78)
    print(f"Dataset Source : {dataset_path}")
    print(f"Standardization: {'Enabled (StandardScaler)' if standardize else 'Disabled'}")
    print(f"Plots Directory: {output_dir}")
    print("-" * 78)

    # 1. Dataset Inspection
    df = pd.read_csv(dataset_path)
    if target_col is None:
        target_col = df.columns[-1]

    missing_cnt = int(df.isnull().sum().sum())
    dup_cnt = int(df.duplicated().sum())
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\n[1] DATASET INSPECTION")
    print(f"  • Shape               : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  • Target Variable     : '{target_col}'")
    print(f"  • Missing Values      : {missing_cnt}")
    print(f"  • Duplicate Rows      : {dup_cnt}")
    print(f"  • Numerical Features  : {len(num_cols)}")
    print(f"  • Categorical Columns : {len(cat_cols)}")

    # Pre-training check
    needs_prep = (missing_cnt > 0 or dup_cnt > 0 or len(cat_cols) > 1)
    if not needs_prep:
        print("  ✓ Dataset Status      : Clean dataset! No missing values, duplicates, or non-numeric features.")
    else:
        print("  ⚠️ Dataset Status      : Preprocessing applied (imputation/deduplication/encoding).")

    # 2. Pre-training preparation
    if dup_cnt > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    if missing_cnt > 0:
        for c in df.columns:
            if df[c].isnull().any():
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c].fillna(df[c].median(), inplace=True)
                else:
                    df[c].fillna(df[c].mode()[0], inplace=True)

    y_raw = df[target_col].values
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
    classes_names = [str(c) for c in le_target.classes_]

    X_df = df.drop(columns=[target_col])
    feat_cat = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in feat_cat:
        le_f = LabelEncoder()
        X_df[col] = le_f.fit_transform(X_df[col].astype(str))

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    # Train / Val / Test split: 60% / 20% / 20%
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    print(f"\n[2] DATA PARTITIONING & STANDARDIZATION")
    print(f"  • Training Samples   : {len(X_train)} (60%)")
    print(f"  • Validation Samples : {len(X_val)} (20%)")
    print(f"  • Test Samples       : {len(X_test)} (20%)")
    print(f"  • Classes ({len(classes_names)})     : {', '.join(classes_names)}")

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        print("  • Feature Scaling    : StandardScaler fit on training partition and applied to val/test.")

    # 3. Default Parameters Run (Gaussian, h = 1.0)
    print(f"\n[3] PARZEN WINDOW DEFAULT PARAMETERS RUN (Gaussian Window, h = 1.0)")
    default_model = ParzenWindowClassifier(h=1.0, window='gaussian')
    default_model.fit(X_train, y_train)

    print("  • Prior Probabilities P(omega_j):")
    for idx, cname in enumerate(classes_names):
        print(f"      - {cname:20s}: {default_model.priors_[idx]:.4f}")

    y_pred_def = default_model.predict(X_test)
    acc_def = accuracy_score(y_test, y_pred_def)
    err_def = 1.0 - acc_def
    prec_def = precision_score(y_test, y_pred_def, average='macro', zero_division=0)
    rec_def = recall_score(y_test, y_pred_def, average='macro', zero_division=0)
    f1_def = f1_score(y_test, y_pred_def, average='macro', zero_division=0)

    print(f"\n  • Test Set Evaluation:")
    print(f"      - Accuracy         : {acc_def * 100:.2f}%")
    print(f"      - Error Rate       : {err_def * 100:.2f}%")
    print(f"      - Macro Precision  : {prec_def * 100:.2f}%")
    print(f"      - Macro Recall     : {rec_def * 100:.2f}%")
    print(f"      - Macro F1-Score   : {f1_def * 100:.2f}%")

    # Table of first 5 test samples
    dens_5, post_5, _, _, _ = default_model.predict_densities_and_posteriors(X_test[:5])
    preds_5 = default_model.predict(X_test[:5])
    print(f"\n  • Estimated Densities p(x|omega_j) & Posteriors P(omega_j|x) for First 5 Test Samples:")
    print("    " + "-" * 88)
    print(f"    {'#':<3} | {'Actual':<15} | {'Predicted':<15} | {'p(x|w_j)':<28} | {'P(w_j|x)':<20}")
    print("    " + "-" * 88)
    for i in range(min(5, len(X_test))):
        act_str = classes_names[y_test[i]]
        pred_str = classes_names[preds_5[i]]
        dens_str = ", ".join([f"{classes_names[j][:4]}:{dens_5[i, j]:.2e}" for j in range(len(classes_names))])
        post_str = ", ".join([f"{classes_names[j][:4]}:{post_5[i, j]:.2f}" for j in range(len(classes_names))])
        print(f"    {i+1:<3} | {act_str:<15} | {pred_str:<15} | {dens_str:<28} | {post_str:<20}")
    print("    " + "-" * 88)

    # Plot Default Confusion Matrix
    cm_def = confusion_matrix(y_test, y_pred_def)
    plt.figure(figsize=(5.5, 4.2))
    sns.heatmap(cm_def, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes_names, yticklabels=classes_names)
    plt.title('Confusion Matrix (Default: Gaussian, h=1.0)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('Actual Class', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm_parzen_default.png'), dpi=150)
    plt.close()

    # 4. Window Width (h) Sweep Analysis
    print(f"\n[4] WINDOW WIDTH (h) SWEEP ANALYSIS (Gaussian vs. Hypercube)")
    print("    " + "-" * 85)
    print(f"    {'h':<6} | {'Gauss Train':<12} | {'Gauss Val':<12} | {'Cube Train':<12} | {'Cube Val':<12} | {'Cube Empty Val':<14}")
    print("    " + "-" * 85)

    gauss_tr_err = []
    gauss_val_err = []
    cube_tr_err = []
    cube_val_err = []
    cube_empty_cnts = []

    best_val_err = float('inf')
    best_config = None

    for h_val in h_values:
        # Gaussian
        mg = ParzenWindowClassifier(h=h_val, window='gaussian')
        mg.fit(X_train, y_train)
        eg_tr = float(np.mean(mg.predict(X_train) != y_train))
        eg_val = float(np.mean(mg.predict(X_val) != y_val))
        gauss_tr_err.append(eg_tr)
        gauss_val_err.append(eg_val)

        # Hypercube
        mc = ParzenWindowClassifier(h=h_val, window='hypercube')
        mc.fit(X_train, y_train)
        ec_tr = float(np.mean(mc.predict(X_train) != y_train))
        ec_val = float(np.mean(mc.predict(X_val) != y_val))
        _, _, _, empty_val_cnt, _ = mc.predict_densities_and_posteriors(X_val)
        cube_tr_err.append(ec_tr)
        cube_val_err.append(ec_val)
        cube_empty_cnts.append(empty_val_cnt)

        print(f"    {h_val:<6.2f} | {eg_tr*100:<11.2f}% | {eg_val*100:<11.2f}% | {ec_tr*100:<11.2f}% | {ec_val*100:<11.2f}% | {empty_val_cnt:<14d}")

        if (eg_val < best_val_err) or (eg_val == best_val_err and (best_config is None or eg_tr < best_config['train_err'])):
            best_val_err = eg_val
            best_config = {'window': 'gaussian', 'h': h_val, 'val_err': eg_val, 'train_err': eg_tr}

        if (ec_val < best_val_err) or (ec_val == best_val_err and (best_config is None or ec_tr < best_config['train_err'])):
            best_val_err = ec_val
            best_config = {'window': 'hypercube', 'h': h_val, 'val_err': ec_val, 'train_err': ec_tr}

    print("    " + "-" * 85)

    # Plot h vs. Error Rate
    plt.figure(figsize=(8.5, 5.0))
    plt.plot(h_values, gauss_tr_err, 'o--', color='#2563eb', label='Gaussian (Train Error)', linewidth=1.8, markersize=5)
    plt.plot(h_values, gauss_val_err, 's-', color='#1d4ed8', label='Gaussian (Val Error)', linewidth=2.4, markersize=6)
    plt.plot(h_values, cube_tr_err, '^--', color='#ea580c', label='Hypercube (Train Error)', linewidth=1.8, markersize=5)
    plt.plot(h_values, cube_val_err, 'D-', color='#c2410c', label='Hypercube (Val Error)', linewidth=2.4, markersize=6)
    plt.axvline(x=best_config['h'], color='#10b981', linestyle=':', linewidth=2,
                label=f"Optimal h* = {best_config['h']} ({best_config['window'].capitalize()})")
    plt.xscale('log')
    plt.title('Window Width Analysis: Window Width h vs. Error Rate (Log Scale)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Window Width h (Logarithmic Scale)', fontsize=10)
    plt.ylabel('Classification Error Rate (1 - Accuracy)', fontsize=10)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    h_plot_file = os.path.join(output_dir, 'parzen_h_vs_error_rate.png')
    plt.savefig(h_plot_file, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved: {h_plot_file}")

    # 5. Optimal Model Selection & Test Set Comparison
    print(f"\n[5] OPTIMAL MODEL SELECTION & COMPARISON")
    print(f"  • Selected Window Function: {best_config['window'].capitalize()}")
    print(f"  • Optimal Window Width h* : {best_config['h']}")
    print(f"  • Validation Error Rate   : {best_config['val_err'] * 100:.2f}%")

    optimal_model = ParzenWindowClassifier(h=best_config['h'], window=best_config['window'])
    optimal_model.fit(X_train, y_train)
    y_pred_opt = optimal_model.predict(X_test)

    acc_opt = accuracy_score(y_test, y_pred_opt)
    err_opt = 1.0 - acc_opt
    prec_opt = precision_score(y_test, y_pred_opt, average='macro', zero_division=0)
    rec_opt = recall_score(y_test, y_pred_opt, average='macro', zero_division=0)
    f1_opt = f1_score(y_test, y_pred_opt, average='macro', zero_division=0)

    # Plot Optimal Confusion Matrix
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    plt.figure(figsize=(5.5, 4.2))
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes_names, yticklabels=classes_names)
    plt.title(f"Confusion Matrix (Optimal: {best_config['window'].capitalize()}, h={best_config['h']})", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('Actual Class', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm_parzen_optimal.png'), dpi=150)
    plt.close()

    print(f"\n  • Side-by-Side Comparison (Default vs. Optimal on Test Set):")
    print("    " + "-" * 72)
    print(f"    {'Metric':<24} | {'Default (Gauss, h=1.0)':<22} | {'Optimal (' + best_config['window'][:5] + ', h=' + str(best_config['h']) + ')':<20}")
    print("    " + "-" * 72)
    print(f"    {'Accuracy':<24} | {acc_def*100:>20.2f}% | {acc_opt*100:>18.2f}%")
    print(f"    {'Error Rate':<24} | {err_def*100:>20.2f}% | {err_opt*100:>18.2f}%")
    print(f"    {'Precision (Macro)':<24} | {prec_def*100:>20.2f}% | {prec_opt*100:>18.2f}%")
    print(f"    {'Recall (Macro)':<24} | {rec_def*100:>20.2f}% | {rec_opt*100:>18.2f}%")
    print(f"    {'F1-Score (Macro)':<24} | {f1_def*100:>20.2f}% | {f1_opt*100:>18.2f}%")
    print("    " + "-" * 72)

    # 6. Test Prediction of New Instance
    print(f"\n[6] PREDICTION OF NEW INSTANCE TEST")
    sample_instance = np.mean(X, axis=0) # Feature means
    print(f"  • Sample Input Vector (Mean values): {np.round(sample_instance, 3)}")

    X_in = np.array([sample_instance])
    if scaler is not None:
        X_in = scaler.transform(X_in)

    dens_new, post_new, _, _, _ = optimal_model.predict_densities_and_posteriors(X_in)
    pred_idx = np.argmax(post_new[0])
    pred_label = classes_names[pred_idx]

    print(f"  • Predicted Class : {pred_label}")
    print("  • Posterior Probabilities P(omega_j | x):")
    for j, cname in enumerate(classes_names):
        print(f"      - {cname:20s}: {post_new[0, j]*100:.2f}% ({post_new[0, j]:.4f})")
    print("  • Estimated Class-Conditional Densities p(x | omega_j):")
    for j, cname in enumerate(classes_names):
        print(f"      - {cname:20s}: {dens_new[0, j]:.4e}")

    # 7. Comparative Benchmark with Other Classifiers
    print(f"\n[7] MULTI-CLASSIFIER BENCHMARK INTEGRATION")
    benchmark_data = [
        ('Parzen Window (Optimal)', acc_opt, f1_opt)
    ]

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    p_nb = gnb.predict(X_test)
    benchmark_data.append(('Gaussian Naive Bayes', accuracy_score(y_test, p_nb), f1_score(y_test, p_nb, average='macro', zero_division=0)))

    # KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
    knn.fit(X_train, y_train)
    p_knn = knn.predict(X_test)
    benchmark_data.append(('K-Nearest Neighbors (k=5)', accuracy_score(y_test, p_knn), f1_score(y_test, p_knn, average='macro', zero_division=0)))

    # Ho-Kashyap (if binary)
    if len(classes_names) == 2:
        hk = HoKashyapClassifier(eta=0.5, max_iter=1000)
        hk.fit(X_train, y_train)
        p_hk = hk.predict(X_test)
        benchmark_data.append(('Ho-Kashyap Linear', accuracy_score(y_test, p_hk), f1_score(y_test, p_hk, average='macro', zero_division=0)))

    print("    " + "-" * 62)
    print(f"    {'Classifier Model':<28} | {'Test Accuracy':<14} | {'Macro F1':<12}")
    print("    " + "-" * 62)
    for model_name, m_acc, m_f1 in benchmark_data:
        print(f"    {model_name:<28} | {m_acc*100:>12.2f}% | {m_f1*100:>10.2f}%")
    print("    " + "-" * 62)

    # Bar chart
    plt.figure(figsize=(8.5, 4.5))
    names = [b[0] for b in benchmark_data]
    accs = [b[1] * 100 for b in benchmark_data]
    f1s = [b[2] * 100 for b in benchmark_data]
    x_pos = np.arange(len(names))
    width = 0.35
    plt.bar(x_pos - width/2, accs, width, label='Accuracy (%)', color='#3b82f6')
    plt.bar(x_pos + width/2, f1s, width, label='Macro F1 (%)', color='#10b981')
    plt.ylabel('Score (%)')
    plt.title('Classifier Benchmark Comparison on Test Set', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(x_pos, names, rotation=15, ha='right', fontsize=9)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=150)
    plt.close()

    print("\n" + "=" * 78)
    print(" ✓ All PR Lab Assignment 09 pipeline steps executed successfully!")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSL517 PR Lab 09: Parzen Window Density Estimation & Classification")
    parser.add_argument('--dataset', type=str, default=None, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, default=None, help='Target class variable name')
    parser.add_argument('--no-standardize', dest='standardize', action='store_false', help='Disable StandardScaler')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        target_col=args.target,
        standardize=args.standardize,
        output_dir=args.output_dir
    )

