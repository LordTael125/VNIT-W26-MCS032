import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents GUI errors in Flask
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')

    if request.method == 'POST':
        
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file.save(filepath)
            df = pd.read_csv(filepath)

            context['summary'] = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_values': df.isnull().sum().sum(),
            }
            context['columns'] = df.columns.tolist()
            context['head'] = df.head().to_html(classes='table', index=False)
            context['target_selection'] = True

        
        elif 'target_var' in request.form:
            df = pd.read_csv(filepath)
            target = request.form['target_var']
            
           
            df = df.dropna()
            
          
            num_features = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if target in num_features: num_features.remove(target)
            if target in cat_features: cat_features.remove(target)
            
           
            for col in cat_features:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            
            X = df.drop(columns=[target])
            y = df[target]

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Bayes Classifier
            model = GaussianNB()
            model.fit(X_train, y_train)
            
            # Save model and feature structure for prediction phase
            joblib.dump(model, os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl'))
            joblib.dump(X.columns.tolist(), os.path.join(app.config['UPLOAD_FOLDER'], 'features.pkl'))

            # Model Evaluation
            y_pred = model.predict(X_test)
            context['eval'] = {
                'target': target,
                'usable_records': len(df),
                'num_features_count': len(num_features),
                'cat_features_count': len(cat_features),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': round(accuracy_score(y_test, y_pred), 3),
                'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 3),
                'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 3),
                'f1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 3)
            }
            
            # Generate Confusion Matrix Graph
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Class')
            plt.ylabel('Actual Class')
            cm_path = 'plots/cm.png'
            plt.savefig(f'static/{cm_path}', bbox_inches='tight')
            plt.close()
            
            context['cm_path'] = cm_path
            context['prediction_form'] = X.columns.tolist()

        # --- PHASE 3: Prediction on New Instances ---
        elif 'predict_btn' in request.form:
            model = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl'))
            features = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'], 'features.pkl'))
            
            input_data = []
            for f in features:
                val = request.form.get(f)
                input_data.append(float(val))
            
            prediction = model.predict([input_data])[0]
            context['prediction_result'] = prediction
            context['prediction_form'] = features

    return render_template('index.html', **context)

if __name__ == '__main__':
 app.run(debug=True, host='0.0.0.0', port=5000)