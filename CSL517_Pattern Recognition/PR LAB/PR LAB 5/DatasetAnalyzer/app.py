import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents GUI errors in Flask when generating plots
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')

    if request.method == 'POST':
        # --- PHASE 1: Handle CSV Upload & Preprocessing ---
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file.save(filepath)
            df = pd.read_csv(filepath)

            # 1. Initial Dataset Analysis
            context['summary'] = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }

            context['info'] = []
            for col in df.columns:
                context['info'].append({
                    'name': col,
                    'type': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'non_null': int(df[col].notnull().sum())
                })

            # 2. Data Preprocessing
            rows_before = df.shape[0]
            duplicates_found = df.duplicated().sum()
            df.drop_duplicates(inplace=True)
            
            # Simple missing value handling: median for numbers, mode for text
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)

            context['preprocessing'] = {
                'rows_before': rows_before,
                'duplicates_found': duplicates_found,
                'duplicates_removed': duplicates_found,
                'rows_after': df.shape[0],
                'missing_after': int(df.isnull().sum().sum())
            }

            # Save cleaned data to maintain state
            df.to_csv(filepath, index=False)
            
            # Pass first 5 records and numeric features to frontend
            context['head'] = df.head().to_html(classes='table', index=False)
            context['num_features'] = df.select_dtypes(include=[np.number]).columns.tolist()

        # --- PHASE 2: Handle Feature Selection & Graph Generation ---
        elif 'features' in request.form:
            df = pd.read_csv(filepath)
            selected_features = request.form.getlist('features')
            context['selected_features'] = selected_features
            
            # Keep previous UI elements visible
            context['num_features'] = df.select_dtypes(include=[np.number]).columns.tolist()
            context['head'] = df.head().to_html(classes='table', index=False)

            # 3. Statistical Analysis
            stats = []
            for col in selected_features:
                stats.append({
                    'feature': col,
                    'mean': round(df[col].mean(), 3),
                    'median': round(df[col].median(), 3),
                    'mode': round(df[col].mode()[0], 3) if not df[col].mode().empty else 'N/A',
                    'var': round(df[col].var(), 3),
                    'std': round(df[col].std(), 3),
                    'min': round(df[col].min(), 3),
                    'max': round(df[col].max(), 3),
                    'range': round(df[col].max() - df[col].min(), 3),
                    'q1': round(df[col].quantile(0.25), 3),
                    'q3': round(df[col].quantile(0.75), 3),
                    'iqr': round(df[col].quantile(0.75) - df[col].quantile(0.25), 3)
                })
            context['stats'] = stats

            # 4. Generate Graphs
            # Clear old plots
            for old_file in os.listdir('static/plots'):
                os.remove(os.path.join('static/plots', old_file))

            hist_paths, box_paths = [], []
            for col in selected_features:
                # Histogram
                plt.figure(figsize=(6, 4))
                sns.histplot(df[col], kde=True)
                plt.title(f'Histogram - {col}')
                hist_path = f'plots/hist_{col}.png'
                plt.savefig(f'static/{hist_path}', bbox_inches='tight')
                plt.close()
                hist_paths.append(hist_path)

                # Box Plot
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=df[col])
                plt.title(f'Box Plot - {col}')
                box_path = f'plots/box_{col}.png'
                plt.savefig(f'static/{box_path}', bbox_inches='tight')
                plt.close()
                box_paths.append(box_path)

            context['hist_paths'] = hist_paths
            context['box_paths'] = box_paths

            # Correlation Heatmap
            if len(selected_features) > 1:
                plt.figure(figsize=(8, 6))
                sns.heatmap(df[selected_features].corr(), annot=True, cmap='viridis', fmt=".2f")
                plt.title('Correlation Matrix Heatmap')
                plt.savefig('static/plots/correlation.png', bbox_inches='tight')
                plt.close()
                context['corr_path'] = 'plots/correlation.png'

    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True)