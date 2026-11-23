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
app.secret_key = 'csl517_pr_lab_08_ho_kashyap_secret'

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
# HO-KASHYAP LINEAR CLASSIFIER (IMPLEMENTED FROM SCRATCH)
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

        # Class 0 -> -1, Class 1 -> +1
        y_bin = np.where(y == self.classes_[1], 1, -1)
        N, d = X.shape

        # Augment with constant bias term: z_i = [1, x_i1, ..., x_id]
        X_aug = np.hstack([np.ones((N, 1)), X])

        # Normalize with class labels: y_i * z_i
        Y = X_aug * y_bin[:, np.newaxis]

        # Compute Moore-Penrose pseudoinverse using NumPy
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

            # Stopping Criterion 1: Convergence to separating hyperplane
            # Either error norm is within epsilon, or all samples are correctly classified
            if np.all(np.abs(e) < self.eps) or misc_count == 0:
                self.status_ = 'separable'
                break

            # Stopping Criterion 2: Ho-Kashyap Proof of Non-Separability
            # If all components e_i <= 0 and e != 0, no separating vector exists
            if np.all(e <= 1e-6) and np.any(e < -1e-6):
                self.status_ = 'inseparable'
                break

            # Ho-Kashyap margin and weight vector updates
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

        # ============================================================
        # STAGE 1: Dataset Upload & Inspection
        # ============================================================
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

            context['summary'] = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_total': missing_total,
                'duplicates_count': duplicates_count,
                'num_cols_count': len(num_cols),
                'cat_cols_count': len(cat_cols)
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
                    'cat_cols': cat_cols
                }, f)

        # ============================================================
        # STAGE 2: Pre-Training Preparation & Classifier Training
        # ============================================================
        elif action == 'train_model':
            if not os.path.exists(filepath):
                flash("No active dataset found. Please upload a dataset first.", "warning")
                return redirect(url_for('index'))

            df = pd.read_csv(filepath)
            target = request.form.get('target_var')
            classifier_type = request.form.get('classifier_type', 'ho_kashyap')
            use_standardization = request.form.get('use_standardization') == '1'
            impute_strategy = request.form.get('impute_strategy', 'median_mode')
            remove_duplicates = request.form.get('remove_duplicates') == '1'
            cat_handling = request.form.get('cat_handling', 'encode') # 'encode' or 'exclude'
            
            # Ho-Kashyap specific hyper-parameters
            default_eta = float(request.form.get('default_eta', 0.5))
            max_iter = int(request.form.get('max_iter', 1000))
            eps_val = float(request.form.get('eps_val', 1e-4))
            
            # User selected eta values for convergence analysis
            selected_etas = request.form.getlist('eta_values')
            if not selected_etas:
                selected_etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            else:
                selected_etas = sorted([float(e) for e in selected_etas])

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

            # Binary restriction check for Ho-Kashyap
            if classifier_type in ['ho_kashyap', 'all'] and len(target_classes) > 2:
                # If user provided a multi-class dataset, reduce to binary (top 2 classes or class 0 vs rest)
                chosen_pair = request.form.get('binary_class_pair')
                if chosen_pair and '_' in chosen_pair:
                    c1_str, c2_str = chosen_pair.split('_', 1)
                    mask = target_series.astype(str).isin([c1_str, c2_str])
                    df = df[mask].reset_index(drop=True)
                    target_series = df[target]
                    le_target = LabelEncoder()
                    y_raw = le_target.fit_transform(target_series.astype(str))
                    target_classes = [str(c) for c in le_target.classes_]
                else:
                    # Default: first class vs all other classes (One-vs-Rest)
                    first_class = target_classes[0]
                    binary_series = (target_series.astype(str) == first_class).astype(int)
                    df[target] = binary_series
                    target_series = df[target]
                    le_target = LabelEncoder()
                    y_raw = le_target.fit_transform(target_series.astype(str))
                    target_classes = [f"Not {first_class}", str(first_class)]

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

            # Train / Test split
            class_counts = pd.Series(y).value_counts()
            can_stratify = (class_counts >= 2).all()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42,
                stratify=y if can_stratify else None
            )

            # Feature Standardization
            scaler = None
            if use_standardization:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
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
                    'mean': round(float(X_df[col].mean()), 2)
                }
            joblib.dump(feature_stats, os.path.join(UPLOAD_FOLDER, 'feature_stats.pkl'))

            context['prep_summary'] = {
                'target': target,
                'target_classes': target_classes,
                'class_distribution': {str(k): int(v) for k, v in target_series.value_counts().items()},
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': len(feature_names),
                'standardization_used': use_standardization,
                'duplicates_removed': dups_removed,
                'missing_handled': missing_handled,
                'cat_handling': cat_handling
            }

            results = {}

            # ========================================================
            # 1. HO-KASHYAP LINEAR CLASSIFIER WORKFLOW
            # ========================================================
            if classifier_type in ['ho_kashyap', 'all']:
                # Step A: Default Parameters Run
                hk_model = HoKashyapClassifier(eta=default_eta, max_iter=max_iter, eps=eps_val)
                hk_model.fit(X_train, y_train)
                y_pred_hk = hk_model.predict(X_test)

                acc_hk = float(accuracy_score(y_test, y_pred_hk))
                err_hk = float(np.sum(y_pred_hk != y_test) / len(y_test))
                prec_hk = float(precision_score(y_test, y_pred_hk, average='weighted', zero_division=0))
                rec_hk = float(recall_score(y_test, y_pred_hk, average='weighted', zero_division=0))
                f1_hk = float(f1_score(y_test, y_pred_hk, average='weighted', zero_division=0))

                # Weight vector a components
                weight_a = hk_model.a_.tolist()
                weight_details = []
                weight_details.append({'name': 'Bias (w0)', 'value': round(weight_a[0], 5), 'is_bias': True})
                for idx, feat in enumerate(feature_names):
                    weight_details.append({'name': feat, 'value': round(weight_a[idx + 1], 5), 'is_bias': False})

                # Margin vector b summary
                b_min = float(np.min(hk_model.b_))
                b_max = float(np.max(hk_model.b_))
                b_mean = float(np.mean(hk_model.b_))

                # Confusion Matrix Plot
                cm_hk = confusion_matrix(y_test, y_pred_hk)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_hk, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title(f'Confusion Matrix (Ho-Kashyap, η={default_eta})', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_hk_path = 'plots/cm_ho_kashyap.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_ho_kashyap.png'), dpi=150)
                plt.close()

                # Step B: Convergence Plots (Iter vs ||e(k)||^2 and Iter vs Misclassified)
                k_iters = list(range(1, len(hk_model.history_e2_) + 1))

                # Plot 1: Iteration vs ||e(k)||^2
                plt.figure(figsize=(7, 4.5))
                plt.plot(k_iters, hk_model.history_e2_, color='#4361ee', linewidth=2, label='Squared Error ‖e(k)‖²')
                plt.title('Convergence Analysis: Iteration k vs. Squared Error ‖e(k)‖²', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Iteration (k)', fontsize=10)
                plt.ylabel('Squared Error ‖e(k)‖²', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(frameon=True, loc='upper right')
                plt.tight_layout()
                plot_e2_path = 'plots/hk_iter_vs_e2.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'hk_iter_vs_e2.png'), dpi=150)
                plt.close()

                # Plot 2: Iteration vs Number of Misclassified Samples
                plt.figure(figsize=(7, 4.5))
                plt.plot(k_iters, hk_model.history_misc_, color='#e63946', linewidth=2, label='Misclassified Samples')
                plt.title('Convergence Analysis: Iteration k vs. Misclassified Samples', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Iteration (k)', fontsize=10)
                plt.ylabel('Samples with aᵀyᵢ ≤ 0', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(frameon=True, loc='upper right')
                plt.tight_layout()
                plot_misc_path = 'plots/hk_iter_vs_misc.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'hk_iter_vs_misc.png'), dpi=150)
                plt.close()

                # Step C: Multi-Eta Convergence & Linear Separability Analysis
                eta_study_results = []
                eta_list = []
                iter_list = []
                error_rate_list = []

                for eta_val in selected_etas:
                    hk_temp = HoKashyapClassifier(eta=eta_val, max_iter=max_iter, eps=eps_val)
                    hk_temp.fit(X_train, y_train)
                    y_test_pred = hk_temp.predict(X_test)
                    err_rate = float(np.sum(y_test_pred != y_test) / len(y_test))
                    acc = 1.0 - err_rate

                    eta_study_results.append({
                        'eta': round(eta_val, 2),
                        'iterations': hk_temp.n_iter_,
                        'status': hk_temp.status_,
                        'error_rate': round(err_rate, 4),
                        'accuracy': round(acc, 4),
                        'final_e2': round(hk_temp.history_e2_[-1], 4),
                        'final_misc': hk_temp.history_misc_[-1]
                    })
                    eta_list.append(eta_val)
                    iter_list.append(hk_temp.n_iter_)
                    error_rate_list.append(err_rate)

                # Find optimal eta: lowest error rate, then fewest iterations
                sorted_by_criteria = sorted(eta_study_results, key=lambda x: (x['error_rate'], x['iterations']))
                optimal_eta_record = sorted_by_criteria[0]

                # Plot 3: Eta vs Number of Iterations (and Error Rate)
                fig, ax1 = plt.subplots(figsize=(8, 4.8))
                color = '#4361ee'
                ax1.set_xlabel('Learning Rate (η)', fontsize=11, fontweight='bold', labelpad=8)
                ax1.set_ylabel('Number of Iterations to Converge', color=color, fontsize=11, fontweight='bold')
                line1 = ax1.plot(eta_list, iter_list, color=color, marker='o', linewidth=2.2, markersize=7, label='Iterations')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(True, linestyle=':', alpha=0.6)

                # Highlight optimal eta
                opt_eta = optimal_eta_record['eta']
                opt_iters = optimal_eta_record['iterations']
                ax1.scatter([opt_eta], [opt_iters], color='#10b981', s=160, zorder=5, label=f'Optimal η={opt_eta}')

                # Secondary axis for Error Rate
                ax2 = ax1.twinx()
                color2 = '#e63946'
                ax2.set_ylabel('Classification Error Rate', color=color2, fontsize=11, fontweight='bold')
                line2 = ax2.plot(eta_list, error_rate_list, color=color2, marker='s', linestyle='--', linewidth=1.8, markersize=6, label='Error Rate')
                ax2.tick_params(axis='y', labelcolor=color2)

                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right', frameon=True)
                plt.title('Ho-Kashyap Analysis: Learning Rate η vs. Number of Iterations & Error Rate', fontsize=12, fontweight='bold', pad=12)
                plt.tight_layout()
                plot_eta_path = 'plots/hk_eta_vs_iterations.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'hk_eta_vs_iterations.png'), dpi=150)
                plt.close()

                # Linear Separability Diagnosis Text
                if hk_model.status_ == 'separable':
                    separability_status = "Linearly Separable"
                    separability_desc = "The dataset is linearly separable. The Ho-Kashyap procedure converged to a hyperplane separating all training instances with margin b > 0."
                    badge_class = "badge-separable"
                elif hk_model.status_ == 'inseparable':
                    separability_status = "Linearly Inseparable"
                    separability_desc = "Proven linearly inseparable by the Ho-Kashyap theorem (all components of error vector e(k) become non-positive while e(k) ≠ 0). No hyperplane can separate the classes."
                    badge_class = "badge-inseparable"
                else:
                    separability_status = "Inconclusive / Maximum Iterations"
                    separability_desc = f"Maximum iterations ({max_iter}) reached without exact margin convergence. The dataset is likely linearly inseparable."
                    badge_class = "badge-maxiter"

                # Save active model
                joblib.dump(hk_model, os.path.join(UPLOAD_FOLDER, 'active_model.pkl'))
                with open(os.path.join(UPLOAD_FOLDER, 'model_info.json'), 'w') as f:
                    json.dump({'type': 'Ho-Kashyap', 'eta': default_eta, 'status': hk_model.status_}, f)

                results['ho_kashyap'] = {
                    'accuracy': round(acc_hk, 3),
                    'precision': round(prec_hk, 3),
                    'recall': round(rec_hk, 3),
                    'f1': round(f1_hk, 3),
                    'error_rate': round(err_hk, 3),
                    'weight_vector': weight_details,
                    'b_min': round(b_min, 4),
                    'b_max': round(b_max, 4),
                    'b_mean': round(b_mean, 4),
                    'cm_path': cm_hk_path,
                    'plot_e2_path': plot_e2_path,
                    'plot_misc_path': plot_misc_path,
                    'plot_eta_path': plot_eta_path,
                    'eta_study_results': eta_study_results,
                    'optimal_eta': optimal_eta_record,
                    'separability_status': separability_status,
                    'separability_desc': separability_desc,
                    'badge_class': badge_class,
                    'total_iterations': hk_model.n_iter_
                }

            # ========================================================
            # 2. K-NEAREST NEIGHBORS (KNN) WORKFLOW
            # ========================================================
            if classifier_type in ['knn', 'all']:
                # Step A: Baseline K=1
                knn1 = KNeighborsClassifier(n_neighbors=1)
                knn1.fit(X_train, y_train)
                y_pred_k1 = knn1.predict(X_test)

                acc_k1 = float(accuracy_score(y_test, y_pred_k1))
                err_k1 = float(np.sum(y_pred_k1 != y_test) / len(y_test))
                prec_k1 = float(precision_score(y_test, y_pred_k1, average='weighted', zero_division=0))
                rec_k1 = float(recall_score(y_test, y_pred_k1, average='weighted', zero_division=0))
                f1_k1 = float(f1_score(y_test, y_pred_k1, average='weighted', zero_division=0))

                cm_k1 = confusion_matrix(y_test, y_pred_k1)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_k1, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title('Confusion Matrix (KNN: K=1)', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_k1_path = 'plots/cm_knn_k1.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_knn_k1.png'), dpi=150)
                plt.close()

                # Step B: Elbow Curve Analysis
                max_allowable_k = min(40, len(X_train) - 1)
                if max_allowable_k < 1:
                    max_allowable_k = 1
                k_range = list(range(1, max_allowable_k + 1))
                knn_error_rates = []

                for k in k_range:
                    clf = KNeighborsClassifier(n_neighbors=k)
                    clf.fit(X_train, y_train)
                    p = clf.predict(X_test)
                    knn_error_rates.append(round(float(np.mean(p != y_test)), 4))

                opt_k = find_elbow_point(k_range, knn_error_rates)

                # Plot Elbow Curve
                plt.figure(figsize=(8, 4.5))
                plt.plot(k_range, knn_error_rates, color='#4361ee', marker='o', linewidth=2, label='Error Rate')
                plt.scatter([opt_k], [knn_error_rates[k_range.index(opt_k)]], color='#e63946', s=160, zorder=5,
                            label=f'Optimal Elbow K={opt_k}')
                plt.axvline(x=opt_k, color='#e63946', linestyle='--', alpha=0.5)
                plt.title("Elbow Method: 'K' vs. Classification Error Rate", fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Values of K', fontsize=10)
                plt.ylabel('Error Rate', fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(frameon=True)
                plt.tight_layout()
                elbow_plot_path = 'plots/knn_elbow.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'knn_elbow.png'), dpi=150)
                plt.close()

                # Retrain with optimal K
                knn_opt = KNeighborsClassifier(n_neighbors=opt_k)
                knn_opt.fit(X_train, y_train)
                y_pred_opt = knn_opt.predict(X_test)

                acc_opt = float(accuracy_score(y_test, y_pred_opt))
                err_opt = float(np.sum(y_pred_opt != y_test) / len(y_test))
                prec_opt = float(precision_score(y_test, y_pred_opt, average='weighted', zero_division=0))
                rec_opt = float(recall_score(y_test, y_pred_opt, average='weighted', zero_division=0))
                f1_opt = float(f1_score(y_test, y_pred_opt, average='weighted', zero_division=0))

                cm_opt = confusion_matrix(y_test, y_pred_opt)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title(f'Confusion Matrix (KNN: Optimal K={opt_k})', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_knn_opt_path = 'plots/cm_knn_opt.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_knn_opt.png'), dpi=150)
                plt.close()

                if classifier_type == 'knn':
                    joblib.dump(knn_opt, os.path.join(UPLOAD_FOLDER, 'active_model.pkl'))
                    with open(os.path.join(UPLOAD_FOLDER, 'model_info.json'), 'w') as f:
                        json.dump({'type': 'KNN', 'k': opt_k}, f)

                results['knn'] = {
                    'k1': {
                        'accuracy': round(acc_k1, 3),
                        'precision': round(prec_k1, 3),
                        'recall': round(rec_k1, 3),
                        'f1': round(f1_k1, 3),
                        'error_rate': round(err_k1, 3),
                        'cm_path': cm_k1_path
                    },
                    'optimal': {
                        'k': opt_k,
                        'accuracy': round(acc_opt, 3),
                        'precision': round(prec_opt, 3),
                        'recall': round(rec_opt, 3),
                        'f1': round(f1_opt, 3),
                        'error_rate': round(err_opt, 3),
                        'cm_path': cm_knn_opt_path
                    },
                    'elbow_plot_path': elbow_plot_path
                }

            # ========================================================
            # 3. NAIVE BAYES (GAUSSIAN NB) WORKFLOW
            # ========================================================
            if classifier_type in ['bayes', 'all']:
                nb_model = GaussianNB()
                nb_model.fit(X_train, y_train)
                y_pred_nb = nb_model.predict(X_test)

                acc_nb = float(accuracy_score(y_test, y_pred_nb))
                err_nb = float(np.sum(y_pred_nb != y_test) / len(y_test))
                prec_nb = float(precision_score(y_test, y_pred_nb, average='weighted', zero_division=0))
                rec_nb = float(recall_score(y_test, y_pred_nb, average='weighted', zero_division=0))
                f1_nb = float(f1_score(y_test, y_pred_nb, average='weighted', zero_division=0))

                cm_nb = confusion_matrix(y_test, y_pred_nb)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title('Confusion Matrix (Naive Bayes)', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_nb_path = 'plots/cm_nb.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_nb.png'), dpi=150)
                plt.close()

                if classifier_type == 'bayes':
                    joblib.dump(nb_model, os.path.join(UPLOAD_FOLDER, 'active_model.pkl'))
                    with open(os.path.join(UPLOAD_FOLDER, 'model_info.json'), 'w') as f:
                        json.dump({'type': 'Naive Bayes'}, f)

                results['bayes'] = {
                    'accuracy': round(acc_nb, 3),
                    'precision': round(prec_nb, 3),
                    'recall': round(rec_nb, 3),
                    'f1': round(f1_nb, 3),
                    'error_rate': round(err_nb, 3),
                    'cm_path': cm_nb_path
                }

            context['results'] = results
            context['classifier_type'] = classifier_type
            context['feature_names'] = feature_names
            context['feature_stats'] = feature_stats
            context['stage'] = 'evaluation_and_prediction'

        # ============================================================
        # STAGE 3: Prediction of New Instances
        # ============================================================
        elif action == 'predict_instance':
            model_path = os.path.join(UPLOAD_FOLDER, 'active_model.pkl')
            features_path = os.path.join(UPLOAD_FOLDER, 'features.pkl')
            scaler_path = os.path.join(UPLOAD_FOLDER, 'scaler.pkl')
            encoder_path = os.path.join(UPLOAD_FOLDER, 'target_encoder.pkl')
            cat_encoders_path = os.path.join(UPLOAD_FOLDER, 'feature_encoders.pkl')
            stats_path = os.path.join(UPLOAD_FOLDER, 'feature_stats.pkl')
            info_path = os.path.join(UPLOAD_FOLDER, 'model_info.json')

            if not os.path.exists(model_path) or not os.path.exists(features_path):
                flash("Model not found. Please train a classifier first.", "danger")
                return redirect(url_for('index'))

            model = joblib.load(model_path)
            feature_names = joblib.load(features_path)
            scaler = joblib.load(scaler_path)
            le_target = joblib.load(encoder_path)
            feature_encoders = joblib.load(cat_encoders_path)
            feature_stats = joblib.load(stats_path)
            with open(info_path, 'r') as f:
                model_info = json.load(f)

            input_values = {}
            row_vector = []
            warnings_list = []

            try:
                for feat in feature_names:
                    val_str = request.form.get(feat, '').strip()
                    if not val_str:
                        flash(f"Please provide an input value for '{feat}'.", "warning")
                        return redirect(url_for('index'))

                    # Encode if categorical, else numeric float
                    if feat in feature_encoders:
                        try:
                            val_num = float(feature_encoders[feat].transform([val_str])[0])
                        except Exception:
                            flash(f"Unrecognized category '{val_str}' for feature '{feat}'.", "danger")
                            return redirect(url_for('index'))
                    else:
                        try:
                            val_num = float(val_str)
                        except ValueError:
                            flash(f"Feature '{feat}' requires a numeric value.", "danger")
                            return redirect(url_for('index'))

                    # Range validation: check if input is far outside training range
                    stat = feature_stats.get(feat, {})
                    f_min = stat.get('min')
                    f_max = stat.get('max')
                    if f_min is not None and f_max is not None:
                        margin_range = (f_max - f_min) * 0.5
                        if val_num < (f_min - margin_range) or val_num > (f_max + margin_range):
                            warnings_list.append(f"Value {val_num} for '{feat}' is far outside the training range [{f_min}, {f_max}].")

                    input_values[feat] = val_str
                    row_vector.append(val_num)

                X_new = np.array([row_vector], dtype=float)
                if scaler is not None:
                    X_new = scaler.transform(X_new)

                pred_raw = model.predict(X_new)[0]
                pred_label = le_target.inverse_transform([int(pred_raw)])[0]

                # Discriminant score and interpretation for Ho-Kashyap
                discriminant_value = None
                discriminant_formula = None
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(X_new)[0])
                    discriminant_value = round(score, 4)
                    w = model.a_
                    formula_parts = [f"{round(w[0], 3)} (bias)"]
                    for i, feat in enumerate(feature_names):
                        formula_parts.append(f"{round(w[i+1], 3)}*{feat}")
                    discriminant_formula = " + ".join(formula_parts)

                context['prediction_result'] = {
                    'model_type': model_info.get('type', 'Trained Classifier'),
                    'predicted_class_code': int(pred_raw),
                    'predicted_label': str(pred_label),
                    'input_values': input_values,
                    'discriminant_value': discriminant_value,
                    'discriminant_formula': discriminant_formula,
                    'warnings': warnings_list
                }

            except Exception as e:
                flash(f"Prediction failed: {str(e)}", "danger")

            context['stage'] = 'evaluation_and_prediction'
            context['feature_names'] = feature_names
            context['feature_stats'] = feature_stats

    return render_template('index.html', **context)


@app.route('/reset')
def reset():
    clean_previous_state()
    flash("Session reset successfully. Upload a new dataset to begin.", "info")
    return redirect(url_for('index'))


if __name__ == '__main__':
    import socket
    port = int(os.environ.get('PORT', 5001))
    # If chosen port is busy, automatically find next free port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', port))
        s.close()
    except OSError:
        for p in range(5001, 5020):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('0.0.0.0', p))
                s.close()
                port = p
                break
            except OSError:
                continue
    print(f"\n==============================================================")
    print(f" Classifiers Studio Web Application Running!")
    print(f" Open in your browser: http://127.0.0.1:{port}/")
    print(f"==============================================================\n")
    app.run(debug=True, host='0.0.0.0', port=port)
