import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import glob
import json
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import joblib
import pandas as pd
import numpy as np
np.seterr(all='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)
app.secret_key = 'csl517_pr_lab_09_parzen_window_secret'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PLOTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'plots')
SAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_datasets')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)


def clean_previous_state():
    """Ensure data from previous uploads does not affect newly uploaded datasets."""
    for folder in [UPLOAD_FOLDER, PLOTS_FOLDER]:
        for filepath in glob.glob(os.path.join(folder, '*')):
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception:
                pass


# ============================================================
# PARZEN WINDOW CLASSIFIER (IMPLEMENTED FROM SCRATCH USING NUMPY)
# ============================================================

class ParzenWindowClassifier:
    """
    Non-parametric Parzen Window Density Estimator & Bayes Classifier
    implemented from scratch using NumPy.
    
    Supports:
      1. Gaussian window function with the log-sum-exp trick to avoid numerical underflow.
      2. Hypercube window function with detection of empty-window samples.
      3. Class-conditional probability density estimation p(x | omega_j).
      4. Posterior probability computation P(omega_j | x) and Bayes decision classification.
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
        # Prior probabilities P(omega_j) = N_j / N
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_], dtype=float)
        self.X_by_class_ = {c: X[y == c] for c in self.classes_}
        return self

    @staticmethod
    def _log_sum_exp(arr, axis=-1):
        """Numerically stable Log-Sum-Exp trick across an axis."""
        m = np.max(arr, axis=axis, keepdims=True)
        is_neginf = np.isneginf(m)
        m_safe = np.where(is_neginf, 0.0, m)
        res = m_safe + np.log(np.sum(np.exp(arr - m_safe), axis=axis, keepdims=True))
        res = np.where(is_neginf, -np.inf, res)
        return np.squeeze(res, axis=axis)

    def predict_densities_and_posteriors(self, X):
        """
        Calculates:
          1. Class-conditional densities p(x | omega_j)
          2. Posterior probabilities P(omega_j | x)
          3. Log densities ln p(x | omega_j)
          4. Count and boolean mask of samples whose hypercube window contains no training sample.
        """
        X = np.asarray(X, dtype=float)
        M = X.shape[0]
        n_classes = len(self.classes_)

        if self.window == 'gaussian':
            # Isotropic Gaussian Parzen Window with Log-Sum-Exp
            # ln p(x | omega_j) = -ln(N_j) - (d/2)*ln(2*pi) - d*ln(h) + LSE(- ||x - x_i||^2 / (2*h^2))
            log_densities = np.zeros((M, n_classes))
            const_term = - (self.d_ / 2.0) * np.log(2.0 * np.pi) - self.d_ * np.log(self.h)

            for j, c in enumerate(self.classes_):
                Xc = self.X_by_class_[c]
                Nc = Xc.shape[0]
                # Vectorized pairwise squared Euclidean distances: shape (M, Nc)
                diff = X[:, np.newaxis, :] - Xc[np.newaxis, :, :]
                dist_sq = np.sum(diff ** 2, axis=-1)
                z = - dist_sq / (2.0 * (self.h ** 2))
                lse_z = self._log_sum_exp(z, axis=-1)
                log_densities[:, j] = - np.log(Nc) + const_term + lse_z

            # Posterior computation in log-space:
            # ln joint = ln p(x | omega_j) + ln P(omega_j)
            log_joint = log_densities + np.log(self.priors_)[np.newaxis, :]
            log_evidence = self._log_sum_exp(log_joint, axis=-1)[:, np.newaxis]
            log_posteriors = log_joint - log_evidence
            posteriors = np.exp(log_posteriors)

            # Prevent exp overflow/underflow for raw densities
            densities = np.exp(np.clip(log_densities, -700, 700))
            empty_count = 0
            empty_mask = np.zeros(M, dtype=bool)

        elif self.window == 'hypercube':
            # Hypercube Parzen Window:
            # phi(u) = 1 if max_i |u_i| <= 1/2, else 0
            # V = h^d
            densities = np.zeros((M, n_classes))
            log_densities = np.full((M, n_classes), -np.inf)
            posteriors = np.zeros((M, n_classes))
            V = float(self.h ** self.d_)
            # In high dimensions, V might be 0.0 or Inf if h is extreme
            if V == 0.0:
                V = 1e-300
            half_h = self.h / 2.0
            k_counts = np.zeros((M, n_classes))

            for j, c in enumerate(self.classes_):
                Xc = self.X_by_class_[c]
                Nc = Xc.shape[0]
                diff = np.abs(X[:, np.newaxis, :] - Xc[np.newaxis, :, :])
                max_diff = np.max(diff, axis=-1)  # shape (M, Nc)
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
                    # Window contains no training sample: fallback to class priors
                    posteriors[i] = self.priors_

        return densities, posteriors, log_densities, empty_count, empty_mask

    def predict(self, X):
        _, posteriors, _, _, _ = self.predict_densities_and_posteriors(X)
        best_indices = np.argmax(posteriors, axis=-1)
        return self.classes_[best_indices]


# ============================================================
# HO-KASHYAP LINEAR CLASSIFIER (FROM SCRATCH NUMPY)
# ============================================================

class HoKashyapClassifier:
    """
    Ho-Kashyap procedure for linear classification implemented from scratch using NumPy.
    Finds the weight vector 'a' and margin vector 'b' such that Y * a = b > 0.
    Provides rigorous detection of linear separability and non-separability.
    """
    def __init__(self, eta=0.5, max_iter=1000, eps=1e-4):
        self.eta = eta
        self.max_iter = max_iter
        self.eps = eps
        self.a_ = None               # Weight vector: [w_0, w_1, ..., w_d] (d+1,)
        self.b_ = None               # Margin vector (N,)
        self.history_e2_ = []        # Squared error ||e(k)||^2 at each iteration
        self.history_misc_ = []      # Misclassified training samples at each iteration
        self.status_ = 'max_iter'    # 'separable', 'inseparable', or 'max_iter'
        self.n_iter_ = 0
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Ho-Kashyap procedure requires exactly 2 classes (binary classification).")

        y_bin = np.where(y == self.classes_[1], 1, -1)
        N, d = X.shape

        X_aug = np.hstack([np.ones((N, 1)), X])
        Y = X_aug * y_bin[:, np.newaxis]
        Y_pinv = np.linalg.pinv(Y)

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

    def decision_function(self, X):
        N = X.shape[0]
        X_aug = np.hstack([np.ones((N, 1)), X])
        return X_aug @ self.a_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0, self.classes_[1], self.classes_[0])


def find_elbow_point(k_values, error_rates):
    """Elbow Point finder for KNN."""
    if not k_values or not error_rates:
        return 1
    min_err = min(error_rates)
    min_indices = [i for i, err in enumerate(error_rates) if err == min_err]
    p1 = np.array([k_values[0], error_rates[0]])
    p2 = np.array([k_values[-1], error_rates[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len > 1e-9:
        line_unit = line_vec / line_len
        distances = []
        for i in range(len(k_values)):
            p = np.array([k_values[i], error_rates[i]])
            vec_to_p = p - p1
            proj = np.dot(vec_to_p, line_unit)
            normal_vec = vec_to_p - proj * line_unit
            distances.append(np.linalg.norm(normal_vec))
        knee_idx = int(np.argmax(distances))
        if error_rates[knee_idx] <= min_err + 0.03:
            return k_values[knee_idx]
    return k_values[min_indices[0]]


# ============================================================
# FLASK WEB APPLICATION ROUTES
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    metadata_path = os.path.join(UPLOAD_FOLDER, 'metadata.json')

    sample_files = []
    if os.path.exists(SAMPLE_FOLDER):
        sample_files = sorted([f for f in os.listdir(SAMPLE_FOLDER) if f.endswith('.csv')])
    context['sample_files'] = sample_files

    if request.method == 'POST':
        action = request.form.get('action')

        # ------------------------------------------------------------
        # STAGE 1: Dataset Upload & Inspection
        # ------------------------------------------------------------
        if action == 'upload_dataset':
            clean_previous_state()
            df = None

            selected_sample = request.form.get('sample_dataset')
            if selected_sample and selected_sample in sample_files:
                src_path = os.path.join(SAMPLE_FOLDER, selected_sample)
                df = pd.read_csv(src_path)
                df.to_csv(filepath, index=False)
                context['dataset_source'] = f"Benchmark Dataset: {selected_sample}"
            elif 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                if not file.filename.lower().endswith('.csv'):
                    flash("Invalid file format. Please upload a standard tabular CSV file.", "danger")
                    return redirect(url_for('index'))
                file.save(filepath)
                try:
                    df = pd.read_csv(filepath)
                except Exception as e:
                    flash(f"Error parsing CSV file: {str(e)}", "danger")
                    return redirect(url_for('index'))
                context['dataset_source'] = f"Uploaded File: {file.filename}"
            else:
                flash("Please choose a CSV file or select a benchmark dataset.", "warning")
                return redirect(url_for('index'))

            if df.empty or df.shape[0] < 5 or df.shape[1] < 2:
                flash("Dataset is too small or empty. Please provide a valid classification dataset.", "danger")
                return redirect(url_for('index'))

            missing_total = int(df.isnull().sum().sum())
            duplicates_count = int(df.duplicated().sum())
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            col_details = []
            for col in df.columns:
                col_details.append({
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'is_numeric': col in num_cols,
                    'missing': int(df[col].isnull().sum()),
                    'unique_values': int(df[col].nunique())
                })

            needs_preprocessing = (missing_total > 0 or duplicates_count > 0 or len(cat_cols) > 1)

            context['summary'] = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_total': missing_total,
                'duplicates_count': duplicates_count,
                'num_cols_count': len(num_cols),
                'cat_cols_count': len(cat_cols),
                'needs_preprocessing': needs_preprocessing
            }
            context['columns'] = df.columns.tolist()
            context['col_details'] = col_details
            context['head'] = df.head(5).to_html(classes='table table-hover', index=False)
            context['stage'] = 'preprocessing_and_config'

            with open(metadata_path, 'w') as f:
                json.dump({
                    'rows': df.shape[0],
                    'cols': df.shape[1],
                    'missing_total': missing_total,
                    'duplicates_count': duplicates_count,
                    'num_cols': num_cols,
                    'cat_cols': cat_cols,
                    'needs_preprocessing': needs_preprocessing
                }, f)

        # ------------------------------------------------------------
        # STAGE 2: Pre-Training Preparation & Classifier Training
        # ------------------------------------------------------------
        elif action == 'train_model':
            if not os.path.exists(filepath):
                flash("No active dataset found. Please upload a dataset first.", "warning")
                return redirect(url_for('index'))

            df = pd.read_csv(filepath)
            target = request.form.get('target_var')
            classifier_type = request.form.get('classifier_type', 'parzen_window')
            use_standardization = request.form.get('use_standardization') == '1'
            impute_strategy = request.form.get('impute_strategy', 'median_mode')
            remove_duplicates = request.form.get('remove_duplicates') == '1'
            cat_handling = request.form.get('cat_handling', 'encode')

            # Parzen Window Hyperparameters & h sweep values
            h_input_raw = request.form.get('h_values', '0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0')
            try:
                selected_h_values = sorted([float(x.strip()) for x in h_input_raw.split(',') if x.strip() and float(x.strip()) > 0])
                if not selected_h_values:
                    selected_h_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            except Exception:
                selected_h_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

            if not target or target not in df.columns:
                flash("Please select a valid target variable.", "danger")
                return redirect(url_for('index'))

            # Handle duplicate rows
            dups_removed = 0
            if remove_duplicates and df.duplicated().sum() > 0:
                dups_removed = int(df.duplicated().sum())
                df = df.drop_duplicates().reset_index(drop=True)

            # Handle missing values
            missing_handled = int(df.isnull().sum().sum())
            if missing_handled > 0:
                if impute_strategy == 'drop':
                    df = df.dropna().reset_index(drop=True)
                else:
                    for col in df.columns:
                        if df[col].isnull().any():
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col].fillna(df[col].median(), inplace=True)
                            else:
                                df[col].fillna(df[col].mode()[0], inplace=True)

            # Target encoding
            target_series = df[target]
            le_target = LabelEncoder()
            y_raw = le_target.fit_transform(target_series.astype(str))
            target_classes = [str(c) for c in le_target.classes_]

            # Feature preparation
            X_df = df.drop(columns=[target])
            cat_features = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

            if cat_handling == 'exclude' and len(cat_features) > 0:
                X_df = X_df.drop(columns=cat_features)
                feature_encoders = {}
            else:
                feature_encoders = {}
                for col in cat_features:
                    le_feat = LabelEncoder()
                    X_df[col] = le_feat.fit_transform(X_df[col].astype(str))
                    feature_encoders[col] = le_feat

            feature_names = X_df.columns.tolist()
            if len(feature_names) == 0:
                flash("No numeric features remain after excluding categorical features. Please choose 'Encode'.", "danger")
                return redirect(url_for('index'))

            X = X_df.values.astype(float)
            y = y_raw

            if len(X) < 10 or len(np.unique(y)) < 2:
                flash("Dataset does not have enough samples or classes for training.", "danger")
                return redirect(url_for('index'))

            # Split dataset into:
            # 60% Train, 20% Validation, 20% Test
            class_counts = pd.Series(y).value_counts()
            can_stratify = (class_counts >= 3).all()

            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42,
                stratify=y if can_stratify else None
            )

            class_counts_sub = pd.Series(y_train_full).value_counts()
            can_stratify_sub = (class_counts_sub >= 2).all()

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.25, random_state=42,
                stratify=y_train_full if can_stratify_sub else None
            )

            # Feature Standardization
            scaler = None
            if use_standardization:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

            # Save artifacts
            joblib.dump(scaler, os.path.join(UPLOAD_FOLDER, 'scaler.pkl'))
            joblib.dump(feature_names, os.path.join(UPLOAD_FOLDER, 'features.pkl'))
            joblib.dump(le_target, os.path.join(UPLOAD_FOLDER, 'target_encoder.pkl'))
            joblib.dump(feature_encoders, os.path.join(UPLOAD_FOLDER, 'feature_encoders.pkl'))

            feature_stats = {}
            for col in feature_names:
                feature_stats[col] = {
                    'min': round(float(X_df[col].min()), 2),
                    'max': round(float(X_df[col].max()), 2),
                    'mean': round(float(X_df[col].mean()), 2),
                    'std': round(float(X_df[col].std()) if X_df[col].std() > 0 else 1.0, 2)
                }
            joblib.dump(feature_stats, os.path.join(UPLOAD_FOLDER, 'feature_stats.pkl'))

            context['prep_summary'] = {
                'target': target,
                'target_classes': target_classes,
                'class_distribution': {str(k): int(v) for k, v in target_series.value_counts().items()},
                'total_samples': len(df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'num_features': len(feature_names),
                'standardization_used': use_standardization,
                'duplicates_removed': dups_removed,
                'missing_handled': missing_handled,
                'cat_handling': cat_handling
            }

            results = {}

            # ========================================================
            # 1. PARZEN WINDOW DENSITY ESTIMATION & CLASSIFICATION
            # ========================================================
            if classifier_type in ['parzen_window', 'all']:
                # ----------------------------------------------------
                # A. Default Parameters Run (Gaussian Window, h = 1.0)
                # ----------------------------------------------------
                default_parzen = ParzenWindowClassifier(h=1.0, window='gaussian')
                default_parzen.fit(X_train, y_train)

                y_pred_def_test = default_parzen.predict(X_test)
                acc_def = float(accuracy_score(y_test, y_pred_def_test))
                err_def = 1.0 - acc_def
                prec_def = float(precision_score(y_test, y_pred_def_test, average='macro', zero_division=0))
                rec_def = float(recall_score(y_test, y_pred_def_test, average='macro', zero_division=0))
                f1_def = float(f1_score(y_test, y_pred_def_test, average='macro', zero_division=0))

                # Estimated densities p(x | omega_j) and posteriors P(omega_j | x) for first 5 test samples
                first_five_X = X_test[:5]
                first_five_y = y_test[:5]
                densities_5, posteriors_5, log_dens_5, _, _ = default_parzen.predict_densities_and_posteriors(first_five_X)
                preds_5 = default_parzen.predict(first_five_X)

                first_five_samples_info = []
                for idx in range(len(first_five_X)):
                    density_dict = {}
                    posterior_dict = {}
                    for c_idx, c_name in enumerate(target_classes):
                        d_val = densities_5[idx, c_idx]
                        p_val = posteriors_5[idx, c_idx]
                        density_dict[c_name] = f"{d_val:.4e}" if (d_val < 0.001 or d_val > 1000) else f"{d_val:.4f}"
                        posterior_dict[c_name] = f"{p_val:.4f} ({p_val*100:.1f}%)"

                    first_five_samples_info.append({
                        'sample_id': idx + 1,
                        'actual_class': target_classes[first_five_y[idx]],
                        'predicted_class': target_classes[preds_5[idx]],
                        'is_correct': bool(first_five_y[idx] == preds_5[idx]),
                        'densities': density_dict,
                        'posteriors': posterior_dict
                    })

                # Confusion Matrix Heatmap for Default Run
                cm_def = confusion_matrix(y_test, y_pred_def_test)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_def, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title('Confusion Matrix (Default: Gaussian, h=1.0)', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_def_path = 'plots/cm_parzen_default.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_parzen_default.png'), dpi=150)
                plt.close()

                # Class priors info
                class_priors_info = [
                    {'class_name': target_classes[i], 'prior': round(float(default_parzen.priors_[i]), 4)}
                    for i in range(len(target_classes))
                ]

                # ----------------------------------------------------
                # B. Window Width Analysis (Sweep across h for both kernels)
                # ----------------------------------------------------
                sweep_results = []
                gauss_tr_err = []
                gauss_val_err = []
                cube_tr_err = []
                cube_val_err = []
                cube_empty_cnts = []

                best_val_err = float('inf')
                best_config = None

                for h_val in selected_h_values:
                    # 1. Gaussian Window
                    m_gauss = ParzenWindowClassifier(h=h_val, window='gaussian')
                    m_gauss.fit(X_train, y_train)
                    p_tr_g = m_gauss.predict(X_train)
                    p_val_g = m_gauss.predict(X_val)
                    e_tr_g = float(np.mean(p_tr_g != y_train))
                    e_val_g = float(np.mean(p_val_g != y_val))
                    gauss_tr_err.append(e_tr_g)
                    gauss_val_err.append(e_val_g)

                    # 2. Hypercube Window
                    m_cube = ParzenWindowClassifier(h=h_val, window='hypercube')
                    m_cube.fit(X_train, y_train)
                    p_tr_c = m_cube.predict(X_train)
                    p_val_c = m_cube.predict(X_val)
                    e_tr_c = float(np.mean(p_tr_c != y_train))
                    e_val_c = float(np.mean(p_val_c != y_val))
                    _, _, _, empty_val_c, _ = m_cube.predict_densities_and_posteriors(X_val)
                    cube_tr_err.append(e_tr_c)
                    cube_val_err.append(e_val_c)
                    cube_empty_cnts.append(empty_val_c)

                    sweep_results.append({
                        'h': h_val,
                        'gauss_train_err': round(e_tr_g, 4),
                        'gauss_val_err': round(e_val_g, 4),
                        'cube_train_err': round(e_tr_c, 4),
                        'cube_val_err': round(e_val_c, 4),
                        'cube_empty_count': empty_val_c
                    })

                    # Check for optimal selection (lowest validation error rate)
                    # Gaussian candidate
                    if (e_val_g < best_val_err) or (e_val_g == best_val_err and (best_config is None or e_tr_g < best_config['train_err'])):
                        best_val_err = e_val_g
                        best_config = {'window': 'gaussian', 'h': h_val, 'val_err': e_val_g, 'train_err': e_tr_g}

                    # Hypercube candidate
                    if (e_val_c < best_val_err) or (e_val_c == best_val_err and (best_config is None or e_tr_c < best_config['train_err'])):
                        best_val_err = e_val_c
                        best_config = {'window': 'hypercube', 'h': h_val, 'val_err': e_val_c, 'train_err': e_tr_c}

                # Plot: h vs. Error Rate on logarithmic scale
                plt.figure(figsize=(8.5, 5.0))
                plt.plot(selected_h_values, gauss_tr_err, 'o--', color='#2563eb', label='Gaussian (Train Error)', linewidth=1.8, markersize=5)
                plt.plot(selected_h_values, gauss_val_err, 's-', color='#1d4ed8', label='Gaussian (Validation Error)', linewidth=2.4, markersize=6)
                plt.plot(selected_h_values, cube_tr_err, '^--', color='#ea580c', label='Hypercube (Train Error)', linewidth=1.8, markersize=5)
                plt.plot(selected_h_values, cube_val_err, 'D-', color='#c2410c', label='Hypercube (Validation Error)', linewidth=2.4, markersize=6)

                # Highlight optimal point
                plt.axvline(x=best_config['h'], color='#10b981', linestyle=':', linewidth=2, label=f"Optimal h* = {best_config['h']} ({best_config['window'].capitalize()})")
                plt.xscale('log')
                plt.title('Window Width Analysis: Window Width h vs. Error Rate (Log Scale)', fontsize=12, fontweight='bold', pad=12)
                plt.xlabel('Window Width h (Logarithmic Scale)', fontsize=10)
                plt.ylabel('Classification Error Rate (1 - Accuracy)', fontsize=10)
                plt.grid(True, which="both", ls=":", alpha=0.6)
                plt.legend(frameon=True, loc='best')
                plt.tight_layout()
                h_curve_path = 'plots/parzen_h_vs_error_rate.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'parzen_h_vs_error_rate.png'), dpi=150)
                plt.close()

                # ----------------------------------------------------
                # C. Evaluate Optimal Model on Test Set
                # ----------------------------------------------------
                optimal_parzen = ParzenWindowClassifier(h=best_config['h'], window=best_config['window'])
                optimal_parzen.fit(X_train, y_train)
                y_pred_opt_test = optimal_parzen.predict(X_test)

                acc_opt = float(accuracy_score(y_test, y_pred_opt_test))
                err_opt = 1.0 - acc_opt
                prec_opt = float(precision_score(y_test, y_pred_opt_test, average='macro', zero_division=0))
                rec_opt = float(recall_score(y_test, y_pred_opt_test, average='macro', zero_division=0))
                f1_opt = float(f1_score(y_test, y_pred_opt_test, average='macro', zero_division=0))

                # Optimal Confusion Matrix
                cm_opt = confusion_matrix(y_test, y_pred_opt_test)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title(f"Confusion Matrix (Optimal: {best_config['window'].capitalize()}, h={best_config['h']})", fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_opt_path = 'plots/cm_parzen_optimal.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_parzen_optimal.png'), dpi=150)
                plt.close()

                # Comparison Table: Default vs Optimal
                comparison_rows = [
                    {'metric': 'Window Function', 'default': 'Gaussian', 'optimal': best_config['window'].capitalize(), 'diff': '-'},
                    {'metric': 'Window Width (h)', 'default': '1.0', 'optimal': str(best_config['h']), 'diff': '-'},
                    {'metric': 'Test Accuracy', 'default': f"{acc_def*100:.2f}%", 'optimal': f"{acc_opt*100:.2f}%", 'diff': f"{(acc_opt - acc_def)*100:+.2f}%"},
                    {'metric': 'Test Error Rate', 'default': f"{err_def*100:.2f}%", 'optimal': f"{err_opt*100:.2f}%", 'diff': f"{(err_opt - err_def)*100:+.2f}%"},
                    {'metric': 'Precision (Macro)', 'default': f"{prec_def*100:.2f}%", 'optimal': f"{prec_opt*100:.2f}%", 'diff': f"{(prec_opt - prec_def)*100:+.2f}%"},
                    {'metric': 'Recall (Macro)', 'default': f"{rec_def*100:.2f}%", 'optimal': f"{rec_opt*100:.2f}%", 'diff': f"{(rec_opt - rec_def)*100:+.2f}%"},
                    {'metric': 'F1-Score (Macro)', 'default': f"{f1_def*100:.2f}%", 'optimal': f"{f1_opt*100:.2f}%", 'diff': f"{(f1_opt - f1_def)*100:+.2f}%"},
                ]

                # Save optimal model for prediction
                joblib.dump(optimal_parzen, os.path.join(UPLOAD_FOLDER, 'optimal_model.pkl'))
                joblib.dump('parzen_window', os.path.join(UPLOAD_FOLDER, 'active_classifier.pkl'))

                results['parzen'] = {
                    'default': {
                        'window': 'Gaussian',
                        'h': 1.0,
                        'accuracy': round(acc_def, 4),
                        'error_rate': round(err_def, 4),
                        'precision': round(prec_def, 4),
                        'recall': round(rec_def, 4),
                        'f1': round(f1_def, 4),
                        'cm_plot': cm_def_path,
                        'class_priors': class_priors_info,
                        'first_five_samples': first_five_samples_info
                    },
                    'sweep': sweep_results,
                    'h_plot': h_curve_path,
                    'optimal': {
                        'window': best_config['window'].capitalize(),
                        'h': best_config['h'],
                        'val_error_rate': round(best_config['val_err'], 4),
                        'accuracy': round(acc_opt, 4),
                        'error_rate': round(err_opt, 4),
                        'precision': round(prec_opt, 4),
                        'recall': round(rec_opt, 4),
                        'f1': round(f1_opt, 4),
                        'cm_plot': cm_opt_path
                    },
                    'comparison': comparison_rows
                }

            # ========================================================
            # 2. BAYES CLASSIFIER (GAUSSIAN NAIVE BAYES)
            # ========================================================
            if classifier_type in ['bayes', 'all']:
                gnb = GaussianNB()
                gnb.fit(X_train, y_train)
                y_pred_nb = gnb.predict(X_test)

                acc_nb = float(accuracy_score(y_test, y_pred_nb))
                err_nb = 1.0 - acc_nb
                prec_nb = float(precision_score(y_test, y_pred_nb, average='macro', zero_division=0))
                rec_nb = float(recall_score(y_test, y_pred_nb, average='macro', zero_division=0))
                f1_nb = float(f1_score(y_test, y_pred_nb, average='macro', zero_division=0))

                cm_nb = confusion_matrix(y_test, y_pred_nb)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title('Confusion Matrix (Gaussian Naive Bayes)', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_nb_path = 'plots/cm_bayes.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_bayes.png'), dpi=150)
                plt.close()

                if classifier_type == 'bayes':
                    joblib.dump(gnb, os.path.join(UPLOAD_FOLDER, 'optimal_model.pkl'))
                    joblib.dump('bayes', os.path.join(UPLOAD_FOLDER, 'active_classifier.pkl'))

                results['bayes'] = {
                    'accuracy': round(acc_nb, 4),
                    'error_rate': round(err_nb, 4),
                    'precision': round(prec_nb, 4),
                    'recall': round(rec_nb, 4),
                    'f1': round(f1_nb, 4),
                    'cm_plot': cm_nb_path,
                    'class_priors': [{'class_name': target_classes[i], 'prior': round(float(gnb.class_prior_[i]), 4)} for i in range(len(target_classes))]
                }

            # ========================================================
            # 3. K-NEAREST NEIGHBORS (WITH ELBOW METHOD)
            # ========================================================
            if classifier_type in ['knn', 'all']:
                max_k = min(15, len(X_train) - 1)
                k_range = list(range(1, max_k + 1))
                knn_error_rates = []

                for k_val in k_range:
                    knn_temp = KNeighborsClassifier(n_neighbors=k_val)
                    knn_temp.fit(X_train, y_train)
                    err_k = float(1.0 - knn_temp.score(X_val, y_val))
                    knn_error_rates.append(err_k)

                best_k = find_elbow_point(k_range, knn_error_rates)

                plt.figure(figsize=(7, 4.5))
                plt.plot(k_range, knn_error_rates, marker='o', linestyle='-', color='#8b5cf6', linewidth=2, label='Val Error Rate')
                plt.axvline(x=best_k, color='#ef4444', linestyle='--', label=f'Optimal k={best_k}')
                plt.title('KNN Optimization: k vs. Validation Error Rate', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Number of Neighbors (k)', fontsize=10)
                plt.ylabel('Validation Error Rate', fontsize=10)
                plt.xticks(k_range)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(frameon=True)
                plt.tight_layout()
                knn_plot_path = 'plots/knn_elbow.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'knn_elbow.png'), dpi=150)
                plt.close()

                knn_opt = KNeighborsClassifier(n_neighbors=best_k)
                knn_opt.fit(X_train, y_train)
                y_pred_knn = knn_opt.predict(X_test)

                acc_knn = float(accuracy_score(y_test, y_pred_knn))
                err_knn = 1.0 - acc_knn
                prec_knn = float(precision_score(y_test, y_pred_knn, average='macro', zero_division=0))
                rec_knn = float(recall_score(y_test, y_pred_knn, average='macro', zero_division=0))
                f1_knn = float(f1_score(y_test, y_pred_knn, average='macro', zero_division=0))

                cm_knn = confusion_matrix(y_test, y_pred_knn)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Purples',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title(f'Confusion Matrix (KNN, k={best_k})', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_knn_path = 'plots/cm_knn.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_knn.png'), dpi=150)
                plt.close()

                if classifier_type == 'knn':
                    joblib.dump(knn_opt, os.path.join(UPLOAD_FOLDER, 'optimal_model.pkl'))
                    joblib.dump('knn', os.path.join(UPLOAD_FOLDER, 'active_classifier.pkl'))

                results['knn'] = {
                    'optimal_k': best_k,
                    'accuracy': round(acc_knn, 4),
                    'error_rate': round(err_knn, 4),
                    'precision': round(prec_knn, 4),
                    'recall': round(rec_knn, 4),
                    'f1': round(f1_knn, 4),
                    'cm_plot': cm_knn_path,
                    'elbow_plot': knn_plot_path
                }

            # ========================================================
            # 4. HO-KASHYAP LINEAR CLASSIFIER
            # ========================================================
            if classifier_type in ['ho_kashyap', 'all']:
                # Binary restriction check for Ho-Kashyap
                if len(target_classes) == 2:
                    hk_model = HoKashyapClassifier(eta=0.5, max_iter=1000, eps=1e-4)
                    hk_model.fit(X_train, y_train)
                    y_pred_hk = hk_model.predict(X_test)

                    acc_hk = float(accuracy_score(y_test, y_pred_hk))
                    err_hk = 1.0 - acc_hk
                    prec_hk = float(precision_score(y_test, y_pred_hk, average='macro', zero_division=0))
                    rec_hk = float(recall_score(y_test, y_pred_hk, average='macro', zero_division=0))
                    f1_hk = float(f1_score(y_test, y_pred_hk, average='macro', zero_division=0))

                    cm_hk = confusion_matrix(y_test, y_pred_hk)
                    plt.figure(figsize=(5.5, 4.2))
                    sns.heatmap(cm_hk, annot=True, fmt='d', cmap='Blues',
                                xticklabels=target_classes, yticklabels=target_classes)
                    plt.title('Confusion Matrix (Ho-Kashyap, η=0.5)', fontsize=12, fontweight='bold', pad=10)
                    plt.xlabel('Predicted Class', fontsize=10)
                    plt.ylabel('Actual Class', fontsize=10)
                    plt.tight_layout()
                    cm_hk_path = 'plots/cm_ho_kashyap.png'
                    plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_ho_kashyap.png'), dpi=150)
                    plt.close()

                    if classifier_type == 'ho_kashyap':
                        joblib.dump(hk_model, os.path.join(UPLOAD_FOLDER, 'optimal_model.pkl'))
                        joblib.dump('ho_kashyap', os.path.join(UPLOAD_FOLDER, 'active_classifier.pkl'))

                    results['ho_kashyap'] = {
                        'accuracy': round(acc_hk, 4),
                        'error_rate': round(err_hk, 4),
                        'precision': round(prec_hk, 4),
                        'recall': round(rec_hk, 4),
                        'f1': round(f1_hk, 4),
                        'status': hk_model.status_,
                        'iterations': hk_model.n_iter_,
                        'cm_plot': cm_hk_path,
                        'is_applicable': True
                    }
                else:
                    results['ho_kashyap'] = {
                        'is_applicable': False,
                        'message': 'Ho-Kashyap requires exactly 2 classes (binary classification). Dataset has >2 classes.'
                    }

            # ========================================================
            # 5. ALL CLASSIFIERS COMPARISON BENCHMARK
            # ========================================================
            if classifier_type == 'all':
                comp_models = []
                comp_accs = []
                comp_f1s = []

                if 'parzen' in results:
                    comp_models.append(f"Parzen ({results['parzen']['optimal']['window']}, h={results['parzen']['optimal']['h']})")
                    comp_accs.append(results['parzen']['optimal']['accuracy'] * 100)
                    comp_f1s.append(results['parzen']['optimal']['f1'] * 100)

                if 'bayes' in results:
                    comp_models.append("Gaussian Naive Bayes")
                    comp_accs.append(results['bayes']['accuracy'] * 100)
                    comp_f1s.append(results['bayes']['f1'] * 100)

                if 'knn' in results:
                    comp_models.append(f"KNN (k={results['knn']['optimal_k']})")
                    comp_accs.append(results['knn']['accuracy'] * 100)
                    comp_f1s.append(results['knn']['f1'] * 100)

                if 'ho_kashyap' in results and results['ho_kashyap'].get('is_applicable', False):
                    comp_models.append("Ho-Kashyap Linear")
                    comp_accs.append(results['ho_kashyap']['accuracy'] * 100)
                    comp_f1s.append(results['ho_kashyap']['f1'] * 100)

                plt.figure(figsize=(9, 4.5))
                x_pos = np.arange(len(comp_models))
                width = 0.35
                plt.bar(x_pos - width/2, comp_accs, width, label='Accuracy (%)', color='#3b82f6')
                plt.bar(x_pos + width/2, comp_f1s, width, label='F1-Score (%)', color='#10b981')
                plt.ylabel('Score (%)', fontsize=10)
                plt.title('Classifier Benchmark Comparison on Test Set', fontsize=12, fontweight='bold', pad=12)
                plt.xticks(x_pos, comp_models, rotation=15, ha='right', fontsize=9)
                plt.ylim(0, 105)
                plt.grid(axis='y', linestyle=':', alpha=0.6)
                plt.legend(frameon=True)
                plt.tight_layout()
                bench_plot_path = 'plots/benchmark_comparison.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'benchmark_comparison.png'), dpi=150)
                plt.close()

                results['benchmark_plot'] = bench_plot_path

            context['results'] = results
            context['classifier_type'] = classifier_type
            context['feature_names'] = feature_names
            context['feature_stats'] = feature_stats
            context['stage'] = 'results_and_prediction'

            joblib.dump(results, os.path.join(UPLOAD_FOLDER, 'results.pkl'))
            joblib.dump(context['prep_summary'], os.path.join(UPLOAD_FOLDER, 'prep_summary.pkl'))

        # ------------------------------------------------------------
        # STAGE 3: Prediction of New Instances
        # ------------------------------------------------------------
        elif action == 'predict_instance':
            model_path = os.path.join(UPLOAD_FOLDER, 'optimal_model.pkl')
            scaler_path = os.path.join(UPLOAD_FOLDER, 'scaler.pkl')
            features_path = os.path.join(UPLOAD_FOLDER, 'features.pkl')
            target_enc_path = os.path.join(UPLOAD_FOLDER, 'target_encoder.pkl')
            feat_enc_path = os.path.join(UPLOAD_FOLDER, 'feature_encoders.pkl')
            stats_path = os.path.join(UPLOAD_FOLDER, 'feature_stats.pkl')
            active_cls_path = os.path.join(UPLOAD_FOLDER, 'active_classifier.pkl')
            results_path = os.path.join(UPLOAD_FOLDER, 'results.pkl')
            prep_path = os.path.join(UPLOAD_FOLDER, 'prep_summary.pkl')

            if not os.path.exists(model_path) or not os.path.exists(features_path):
                flash("Model artifacts missing. Please train a classifier first.", "warning")
                return redirect(url_for('index'))

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(features_path)
            target_enc = joblib.load(target_enc_path)
            feat_encoders = joblib.load(feat_enc_path)
            feature_stats = joblib.load(stats_path)
            active_cls = joblib.load(active_cls_path) if os.path.exists(active_cls_path) else 'parzen_window'

            if os.path.exists(results_path):
                context['results'] = joblib.load(results_path)
            if os.path.exists(prep_path):
                context['prep_summary'] = joblib.load(prep_path)

            input_vector = []
            input_warnings = []
            entered_values = {}

            for col in feature_names:
                raw_val = request.form.get(col, '').strip()
                entered_values[col] = raw_val

                if raw_val == '':
                    input_warnings.append(f"Feature '{col}' was left empty. Defaulting to training mean.")
                    num_val = feature_stats[col]['mean']
                else:
                    try:
                        num_val = float(raw_val)
                    except ValueError:
                        if col in feat_encoders and raw_val in feat_encoders[col].classes_:
                            num_val = float(feat_encoders[col].transform([raw_val])[0])
                        else:
                            input_warnings.append(f"Invalid non-numeric value for '{col}'. Defaulting to training mean.")
                            num_val = feature_stats[col]['mean']

                # Distribution Range Validation Check
                col_min = feature_stats[col]['min']
                col_max = feature_stats[col]['max']
                col_mean = feature_stats[col]['mean']
                col_std = feature_stats[col]['std']

                if num_val < (col_mean - 3.0 * col_std) or num_val > (col_mean + 3.0 * col_std):
                    input_warnings.append(
                        f"Warning for '{col}': Value {num_val} is outside standard training bounds [{col_min}, {col_max}]."
                    )

                input_vector.append(num_val)

            X_in = np.array([input_vector], dtype=float)
            if scaler is not None:
                X_in = scaler.transform(X_in)

            # Perform prediction based on active model type
            pred_class_label = "Unknown"
            posteriors_list = []
            densities_list = []

            if isinstance(model, ParzenWindowClassifier):
                dens, post, _, empty_cnt, _ = model.predict_densities_and_posteriors(X_in)
                pred_idx = np.argmax(post[0])
                pred_class_label = str(target_enc.inverse_transform([model.classes_[pred_idx]])[0])

                for j, c in enumerate(model.classes_):
                    c_name = str(target_enc.inverse_transform([c])[0])
                    p_val = float(post[0, j])
                    d_val = float(dens[0, j])
                    posteriors_list.append({
                        'class_name': c_name,
                        'prob': round(p_val * 100, 2),
                        'raw_prob': f"{p_val:.4f}"
                    })
                    densities_list.append({
                        'class_name': c_name,
                        'density': f"{d_val:.4e}" if (d_val < 0.001 or d_val > 1000) else f"{d_val:.4f}"
                    })
            elif hasattr(model, 'predict_proba'):
                post = model.predict_proba(X_in)[0]
                pred_idx = np.argmax(post)
                pred_class_label = str(target_enc.inverse_transform([model.classes_[pred_idx]])[0])
                for j, c in enumerate(model.classes_):
                    c_name = str(target_enc.inverse_transform([c])[0])
                    p_val = float(post[j])
                    posteriors_list.append({
                        'class_name': c_name,
                        'prob': round(p_val * 100, 2),
                        'raw_prob': f"{p_val:.4f}"
                    })
            else:
                pred = model.predict(X_in)[0]
                pred_class_label = str(target_enc.inverse_transform([pred])[0])
                posteriors_list.append({'class_name': pred_class_label, 'prob': 100.0, 'raw_prob': '1.0000'})

            context['prediction'] = {
                'predicted_class': pred_class_label,
                'posteriors': posteriors_list,
                'densities': densities_list,
                'warnings': input_warnings,
                'entered_values': entered_values,
                'model_name': model.__class__.__name__
            }
            context['feature_names'] = feature_names
            context['feature_stats'] = feature_stats
            context['stage'] = 'results_and_prediction'

    return render_template('index.html', **context)


@app.route('/reset')
def reset():
    clean_previous_state()
    flash("Session reset successfully. Upload a new dataset to begin.", "info")
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
