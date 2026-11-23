import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import glob
import json
import joblib
import pandas as pd
import numpy as np
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
app.secret_key = 'csl517_pr_lab_07_secret_key'

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


def find_elbow_point(k_values, error_rates):
    """
    Find optimal K using minimum error rate with bias toward smaller K to avoid tie-breaks,
    or using distance to the secant line between end points (Knee method).
    """
    if not k_values or not error_rates:
        return 1
    
    # Primary heuristic: Minimum error rate
    min_err = min(error_rates)
    min_indices = [i for i, err in enumerate(error_rates) if err == min_err]
    
    # Secondary check: if there is a flat valley, pick the knee where error drops significantly
    # Distance from curve to chord connecting first and last point
    p1 = np.array([k_values[0], error_rates[0]])
    p2 = np.array([k_values[-1], error_rates[-1]])
    
    distances = []
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len > 1e-9:
        line_unit = line_vec / line_len
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

    # Available sample datasets for quick 1-click loading
    sample_files = []
    if os.path.exists(SAMPLE_FOLDER):
        sample_files = [f for f in os.listdir(SAMPLE_FOLDER) if f.endswith('.csv')]
    context['sample_files'] = sample_files

    if request.method == 'POST':
        action = request.form.get('action')

        # ============================================================
        # STAGE 1: Dataset Upload & Initial Inspection
        # ============================================================
        if action == 'upload_dataset':
            clean_previous_state()
            df = None

            # Handle sample dataset loading or file upload
            selected_sample = request.form.get('sample_dataset')
            if selected_sample and selected_sample in sample_files:
                src_path = os.path.join(SAMPLE_FOLDER, selected_sample)
                df = pd.read_csv(src_path)
                df.to_csv(filepath, index=False)
                context['dataset_source'] = f"Sample Dataset: {selected_sample}"
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
                flash("Please choose a CSV file or select a sample dataset.", "warning")
                return redirect(url_for('index'))

            # Check if dataframe is empty
            if df.empty or df.shape[0] < 5 or df.shape[1] < 2:
                flash("Dataset is too small or empty. Please provide a valid classification dataset.", "danger")
                return redirect(url_for('index'))

            # Inspection metrics
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

            # Persist dataset state metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'rows': df.shape[0],
                    'cols': df.shape[1],
                    'missing_total': missing_total,
                    'duplicates_count': duplicates_count
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
            classifier_type = request.form.get('classifier_type', 'knn')
            use_standardization = request.form.get('use_standardization') == '1'
            impute_strategy = request.form.get('impute_strategy', 'median_mode')
            remove_duplicates = request.form.get('remove_duplicates') == '1'
            custom_max_k = int(request.form.get('max_k', 40))

            if not target or target not in df.columns:
                flash("Please select a valid target variable.", "danger")
                return redirect(url_for('index'))

            # Handle duplicate rows if requested
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
            y = le_target.fit_transform(target_series.astype(str))
            target_classes = [str(c) for c in le_target.classes_]

            # Feature preparation
            X_df = df.drop(columns=[target])
            feature_names = X_df.columns.tolist()

            # Encode categorical features in X
            cat_features = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
            feature_encoders = {}
            for col in cat_features:
                le_feat = LabelEncoder()
                X_df[col] = le_feat.fit_transform(X_df[col].astype(str))
                feature_encoders[col] = le_feat

            X = X_df.values.astype(float)

            # Minimum sample checks
            if len(X) < 10 or len(np.unique(y)) < 2:
                flash("Dataset does not have enough distinct classes or samples for classification.", "danger")
                return redirect(url_for('index'))

            # Train / Test split (Stratified if every class has >= 2 samples)
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

            # Save training artifacts for prediction phase
            joblib.dump(scaler, os.path.join(UPLOAD_FOLDER, 'scaler.pkl'))
            joblib.dump(feature_names, os.path.join(UPLOAD_FOLDER, 'features.pkl'))
            joblib.dump(le_target, os.path.join(UPLOAD_FOLDER, 'target_encoder.pkl'))
            joblib.dump(feature_encoders, os.path.join(UPLOAD_FOLDER, 'feature_encoders.pkl'))

            # Save feature summary stats for input field placeholders
            feature_stats = {}
            for col in feature_names:
                feature_stats[col] = {
                    'min': round(float(X_df[col].min()), 2),
                    'max': round(float(X_df[col].max()), 2),
                    'mean': round(float(X_df[col].mean()), 2)
                }
            joblib.dump(feature_stats, os.path.join(UPLOAD_FOLDER, 'feature_stats.pkl'))

            # Pre-training report context
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
                'missing_handled': missing_handled
            }

            results = {}

            # ========================================================
            # 1. K-NEAREST NEIGHBORS (KNN) WORKFLOW
            # ========================================================
            if classifier_type in ['knn', 'both']:
                # ----------------- Step A: KNN with K=1 -----------------
                knn1 = KNeighborsClassifier(n_neighbors=1)
                knn1.fit(X_train, y_train)
                y_pred_k1 = knn1.predict(X_test)

                acc_k1 = accuracy_score(y_test, y_pred_k1)
                err_k1 = float(np.sum(y_pred_k1 != y_test) / len(y_test))
                prec_k1 = precision_score(y_test, y_pred_k1, average='weighted', zero_division=0)
                rec_k1 = recall_score(y_test, y_pred_k1, average='weighted', zero_division=0)
                f1_k1 = f1_score(y_test, y_pred_k1, average='weighted', zero_division=0)

                # Confusion Matrix for K=1
                cm_k1 = confusion_matrix(y_test, y_pred_k1)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_k1, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title('Confusion Matrix (KNN: K=1)', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_k1_path = 'plots/cm_k1.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_k1.png'), dpi=150)
                plt.close()

                # --------- Step B: Error Rate Analysis (K = 1 to max_k) ---------
                max_allowable_k = min(custom_max_k, len(X_train) - 1)
                if max_allowable_k < 1:
                    max_allowable_k = 1

                k_range = list(range(1, max_allowable_k + 1))
                error_rates = []
                accuracies = []

                for k in k_range:
                    knn_k = KNeighborsClassifier(n_neighbors=k)
                    knn_k.fit(X_train, y_train)
                    pred_k = knn_k.predict(X_test)
                    incorrect = int(np.sum(pred_k != y_test))
                    total = len(y_test)
                    err = incorrect / total
                    error_rates.append(round(err, 4))
                    accuracies.append(round(accuracy_score(y_test, pred_k), 4))

                # Identify Elbow Point
                optimal_k = find_elbow_point(k_range, error_rates)

                # Plot K vs Error Rate Graph
                plt.figure(figsize=(9, 4.8))
                plt.plot(k_range, error_rates, color='#4361ee', linestyle='-', linewidth=2,
                         marker='o', markersize=6, markerfacecolor='#3a0ca3', markeredgecolor='white',
                         label='Classification Error Rate')
                
                # Highlight Elbow Point
                opt_err = error_rates[k_range.index(optimal_k)]
                plt.scatter([optimal_k], [opt_err], color='#e63946', s=160, zorder=5,
                            label=f'Optimal Elbow Point (K={optimal_k}, Error={opt_err:.3f})')
                plt.axvline(x=optimal_k, color='#e63946', linestyle='--', alpha=0.6)

                plt.title("Elbow Method: 'K' Value vs. Classification Error Rate", fontsize=14, fontweight='bold', pad=12)
                plt.xlabel("Values of K (Number of Neighbors)", fontsize=11, labelpad=8)
                plt.ylabel("Error Rate (Incorrect / Total Predictions)", fontsize=11, labelpad=8)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(frameon=True, facecolor='white', loc='upper right')
                plt.tight_layout()
                elbow_plot_path = 'plots/k_vs_error_rate.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'k_vs_error_rate.png'), dpi=150)
                plt.close()

                # --------- Step C: Retrain KNN with Elbow-Point K ---------
                knn_opt = KNeighborsClassifier(n_neighbors=optimal_k)
                knn_opt.fit(X_train, y_train)
                y_pred_opt = knn_opt.predict(X_test)

                acc_opt = accuracy_score(y_test, y_pred_opt)
                err_opt = float(np.sum(y_pred_opt != y_test) / len(y_test))
                prec_opt = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)
                rec_opt = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)
                f1_opt = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)

                # Confusion Matrix for Optimal K
                cm_opt = confusion_matrix(y_test, y_pred_opt)
                plt.figure(figsize=(5.5, 4.2))
                sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens',
                            xticklabels=target_classes, yticklabels=target_classes)
                plt.title(f'Confusion Matrix (KNN: Optimal K={optimal_k})', fontsize=12, fontweight='bold', pad=10)
                plt.xlabel('Predicted Class', fontsize=10)
                plt.ylabel('Actual Class', fontsize=10)
                plt.tight_layout()
                cm_opt_path = 'plots/cm_elbow.png'
                plt.savefig(os.path.join(PLOTS_FOLDER, 'cm_elbow.png'), dpi=150)
                plt.close()

                # Save the optimal KNN model as the primary model
                joblib.dump(knn_opt, os.path.join(UPLOAD_FOLDER, 'active_model.pkl'))
                with open(os.path.join(UPLOAD_FOLDER, 'model_info.json'), 'w') as f:
                    json.dump({'type': 'KNN', 'k': optimal_k}, f)

                # Record comparative performance
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
                        'k': optimal_k,
                        'accuracy': round(acc_opt, 3),
                        'precision': round(prec_opt, 3),
                        'recall': round(rec_opt, 3),
                        'f1': round(f1_opt, 3),
                        'error_rate': round(err_opt, 3),
                        'cm_path': cm_opt_path
                    },
                    'improvement': {
                        'acc_diff': round((acc_opt - acc_k1) * 100, 2),
                        'error_diff': round((err_k1 - err_opt) * 100, 2)
                    },
                    'k_range': k_range,
                    'error_rates': error_rates,
                    'elbow_plot_path': elbow_plot_path
                }

            # ========================================================
            # 2. NAIVE BAYES CLASSIFIER WORKFLOW
            # ========================================================
            if classifier_type in ['bayes', 'both']:
                nb_model = GaussianNB()
                nb_model.fit(X_train, y_train)
                y_pred_nb = nb_model.predict(X_test)

                acc_nb = accuracy_score(y_test, y_pred_nb)
                err_nb = float(np.sum(y_pred_nb != y_test) / len(y_test))
                prec_nb = precision_score(y_test, y_pred_nb, average='weighted', zero_division=0)
                rec_nb = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)
                f1_nb = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)

                # Confusion Matrix for Naive Bayes
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
            try:
                for feat in feature_names:
                    val_str = request.form.get(feat, '').strip()
                    if not val_str:
                        flash(f"Please provide a valid input value for '{feat}'.", "warning")
                        return redirect(url_for('index'))

                    # Encode if categorical, otherwise parse float
                    if feat in feature_encoders:
                        val_num = float(feature_encoders[feat].transform([val_str])[0])
                    else:
                        val_num = float(val_str)

                    input_values[feat] = val_str
                    row_vector.append(val_num)

                X_new = np.array([row_vector])
                if scaler is not None:
                    X_new = scaler.transform(X_new)

                pred_idx = model.predict(X_new)[0]
                pred_label = le_target.inverse_transform([int(pred_idx)])[0]

                # Probabilities / voting distribution if supported
                confidence = None
                if hasattr(model, 'predict_proba'):
                    prob_array = model.predict_proba(X_new)[0]
                    confidence = round(float(np.max(prob_array)) * 100, 1)

                context['prediction_result'] = {
                    'model_type': model_info.get('type', 'Trained Classifier'),
                    'k_val': model_info.get('k', None),
                    'predicted_class_code': int(pred_idx),
                    'predicted_label': str(pred_label),
                    'confidence': confidence,
                    'input_values': input_values
                }

            except Exception as e:
                flash(f"Prediction failed: {str(e)}", "danger")

            context['stage'] = 'evaluation_and_prediction'
            context['feature_names'] = feature_names
            context['feature_stats'] = feature_stats
            # Restore previous plots if existing
            if os.path.exists(os.path.join(PLOTS_FOLDER, 'k_vs_error_rate.png')):
                context['cached_elbow_plot'] = 'plots/k_vs_error_rate.png'

    return render_template('index.html', **context)


@app.route('/reset')
def reset():
    """Completely resets state and files."""
    clean_previous_state()
    flash("Session reset successfully. Upload a new dataset to begin.", "info")
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
