# Pattern Recognition Lab - Assignment 07
## Web-based Machine Learning Application: K-Nearest Neighbors (KNN) Classifier Studio & Bayes Classifier Integration

---

## 📌 Executive Summary
This application fulfills all objectives and functional requirements defined in **Pattern Recognition Lab: Assignment 07** (Department of Computer Science & Engineering).

The application is built using **Flask**, **Pandas**, **NumPy**, **Scikit-Learn**, **Matplotlib**, and **Seaborn**. It enables users to:
1. Upload and validate any standard tabular classification CSV dataset (or choose from built-in benchmarks like Heart Disease, Iris, and Synthetic Bayes).
2. Inspect dataset dimensions, feature schema, missing values, duplicates, and sample records.
3. Perform pre-processing operations (conditional missing value imputation, deduplication, categorical label encoding, and numerical standardization via `StandardScaler`).
4. Train a baseline **K-Nearest Neighbors (KNN)** model at $K = 1$ and record evaluation metrics (Accuracy, Error Rate, Precision, Recall, F1-Score, Confusion Matrix).
5. Analyze the relationship between **$K$** and **Classification Error Rate** across $K = 1 \dots 40$ using the **Elbow Method**:
   $$\text{Error Rate} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Predictions}} = 1 - \text{Accuracy}$$
6. Retrain KNN using the detected optimal **Elbow-Point $K$**, generating comparative metrics and improvement analysis against $K = 1$.
7. Select and evaluate the **Naive Bayes (GaussianNB)** classifier from Lab 06, or execute a comprehensive side-by-side benchmark (**KNN vs. Naive Bayes**).
8. Predict class outcomes for new user-provided instances via a dynamic prediction form with human-readable class descriptions.

---

## 📂 Project Directory Structure

```text
PR LAB 7/
├── Lab Assignment 07.pdf             # Official lab assignment sheet
├── pr_lab_7.py                       # Standalone Python CLI script for automated execution & verification
├── DatasetAnalyzer -> Classifiers    # Backward-compatible symlink for evaluation scripts
└── Classifiers/                      # Integrated Flask Web Application
    ├── app.py                        # Main Flask server and ML pipelines
    ├── requirements.txt              # Python library dependencies
    ├── README.md                     # Complete documentation & Viva preparation guide
    ├── sample_datasets/              # Standard preloaded CSV benchmarks
    │   ├── heart.csv                 # Cleveland Heart Disease dataset
    │   ├── Iris_Data.csv             # Fisher's Iris Flower dataset
    │   └── Naive-Bayes-Classification-Data.csv
    ├── static/
    │   ├── css/
    │   │   └── style.css             # Modern responsive styling & design system
    │   └── plots/                    # Generated error curves & confusion matrices
    ├── templates/
    │   └── index.html                # Multi-stage web interface
    └── uploads/                      # Isolated session uploads & cached models
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Make sure Python 3.9+ is installed. Activate your project virtual environment or install dependencies:

```bash
cd "PR LAB 7/Classifiers"
pip install -r requirements.txt
```

*Required packages:* `Flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

### 2. Launching the Web Application
Run the Flask server:
```bash
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

### 3. Running the Standalone Python Script (CLI Verification)
To execute the automated pipeline from the terminal:
```bash
cd "PR LAB 7"
python pr_lab_7.py
```
This runs the full workflow on `heart.csv`, generates the $K$ vs. Error Rate plot in `Classifiers/static/plots/cli_k_vs_error_rate.png`, prints all confusion matrices, comparison tables, and test predictions.

---

## 🔍 Detailed Functional Architecture

### Step 1: Dataset Upload & Prevention of State Bleed
- Allows custom CSV file uploads or 1-click selection from bundled benchmark datasets.
- Implements strict validation against empty files, non-CSV formats, and unreadable files.
- Automatically purges prior models, uploads, and cached plots upon a new upload (`clean_previous_state()`), guaranteeing data isolation between experiments.

### Step 2: Dataset Inspection & Pre-Training Diagnostics
- Displays row/column counts, categorical vs. numeric feature distribution, duplicate row counts, and total missing cells.
- Shows an interactive 5-row sample preview and a schema table with data types and cardinality.

### Step 3: Pre-Processing & Feature Standardization
- **Conditional Missing Value Handling:** Imputation options (Median for numerical, Mode for categorical) are displayed only if missing values exist. Otherwise, a confirmation message confirms the dataset is clean.
- **Deduplication:** Allows instant removal of duplicate rows if detected.
- **Feature Standardization (`StandardScaler`):**
  $$z = \frac{x - \mu}{\sigma}$$
  Because KNN uses distance metrics (such as Euclidean distance $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$), features with larger absolute scales would dominate the metric. Standardization guarantees equal weighting.
- **Classifier Selection Dropdown:**
  - `K-Nearest Neighbors (KNN)`
  - `Naive Bayes (GaussianNB)`
  - `Both (Comparative Benchmark)`

### Step 4: KNN Execution & The Elbow Method
1. **Baseline Evaluation ($K=1$):**
   - Fits `KNeighborsClassifier(n_neighbors=1)`.
   - Records Accuracy, Precision, Recall, F1-Score, and plots the Confusion Matrix.
2. **Error Rate Analysis across $K \in [1, 40]$:**
   - Computes:
     $$\text{Error Rate} = \frac{\text{Incorrect Predictions}}{\text{Total Predictions}} = 1 - \text{Accuracy}$$
   - Generates the **$K$ vs. Error Rate** curve with markers and gridlines.
3. **Elbow-Point Identification:**
   - Identifies the sharp bend (or global minimum error rate point) that balances bias and variance.
4. **Retraining with Elbow $K$ & Comparative Study:**
   - Retrains the classifier at optimal $K$.
   - Displays a side-by-side comparative table showing the percentage improvement in Accuracy and reduction in Error Rate over $K=1$.

### Step 5: Prediction on New User Instances
- Dynamically generates input fields for each feature, pre-populated with dataset mean values for immediate testing.
- Passes new inputs through the fitted `StandardScaler` (if enabled).
- Outputs the predicted numerical class code (e.g., `0` or `1`), the descriptive class label (e.g., `Iris-virginica` or `Heart Disease Present`), and model prediction confidence.

---

## 🎓 Viva Voce & Theoretical Concepts (4 Marks Preparation)

### Q1: What is the K-Nearest Neighbors (KNN) algorithm, and why is it called a "lazy learner"?
**Answer:** KNN is a non-parametric, instance-based supervised learning algorithm. It is termed a "lazy learner" because it does not learn an explicit discriminative mathematical boundary during training. Instead, it simply stores the training instances. All computation—distance calculation, nearest neighbor identification, and majority voting—occurs at test/query time.

### Q2: Why is feature standardization so critical before training a KNN classifier?
**Answer:** KNN computes distances (typically Euclidean distance) between feature vectors. If features have vastly different numeric scales (e.g., Age $[20-80]$ vs. Annual Income $[\$20,000 - \$200,000]$), the feature with the larger scale dominates the distance calculation, making the smaller features essentially irrelevant. Standardization (`StandardScaler`) centers the features at zero mean and scales them to unit variance, ensuring every feature contributes proportionately.

### Q3: What is the Elbow Method, and how does it determine optimal $K$?
**Answer:** The Elbow Method plots $K$ (on the x-axis) against the Classification Error Rate (on the y-axis).
- At **$K=1$**, the model is prone to **high variance / overfitting** because it is sensitive to noise and outliers in the training set.
- As **$K$ increases**, the error rate typically decreases until a point of diminishing returns—the "elbow point"—where the curve stabilizes or starts increasing.
- For **very large $K$**, the model suffers from **high bias / underfitting** because the neighborhood spans multiple classes and over-smooths decision boundaries.
- The elbow point represents the optimal sweet spot minimizing generalization error.

### Q4: How is Classification Error Rate calculated?
**Answer:**
$$\text{Error Rate} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Predictions}} = \frac{\text{False Positives} + \text{False Negatives}}{\text{Total Samples}} = 1 - \text{Accuracy}$$

### Q5: How does KNN differ fundamentally from the Naive Bayes classifier?
**Answer:**
| Feature | K-Nearest Neighbors (KNN) | Naive Bayes (GaussianNB) |
|---|---|---|
| **Learning Paradigm** | Instance-based (Lazy) | Eager Probabilistic Model |
| **Model Assumptions** | Geometry/distance in metric space | Conditional feature independence given class |
| **Training Speed** | $\mathcal{O}(1)$ (just storing data) | Fast ($\mathcal{O}(N \cdot D)$ to compute mean/variance) |
| **Inference Speed** | Slow for large $N$ ($\mathcal{O}(N \cdot D)$) | Very fast ($\mathcal{O}(C \cdot D)$) |
| **Sensitivity to Scaling** | Highly sensitive (requires standardization) | Insensitive (independent likelihood computation) |
