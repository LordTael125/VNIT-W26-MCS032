# Pattern Recognition Lab - Assignment 08
## Web-based Machine Learning Application: Linear Classification using the Ho-Kashyap Procedure Studio, KNN & Bayes Classifier Integration

**Course:** CSL517 - Pattern Recognition  
**Lab Session Date:** 3rd November 2026  
**Evaluation Scheme (Total 10 Marks):**
- **Attendance:** 02 Marks
- **Assignment Completion & Correctness:** 04 Marks
- **Viva Voce:** 04 Marks

---

## 📌 Executive Summary
This application fulfills all objectives and functional specifications defined in **Pattern Recognition Lab: Assignment 08** (Department of Computer Science & Engineering).

Built using **Flask**, **NumPy** (from-scratch algorithm implementation), **Pandas**, **Scikit-Learn**, **Matplotlib**, and **Seaborn**, the application provides an interactive web-based ML environment and standalone CLI pipeline to:
1. **Upload & Validate Datasets:** Support any standard tabular classification CSV dataset, with built-in benchmarks (Linearly Separable 2D, Non-Linearly Separable 2D, Cleveland Heart Disease, Iris Binary, and Naive Bayes benchmark).
2. **Pre-Training Preparation & Inspection:** Conditionally inspect and preprocess missing values (median/mode imputation or row removal), remove duplicates, handle categorical features (Label Encoding or Exclusion), and provide user choice for numerical feature standardization via `StandardScaler`.
3. **Ho-Kashyap Procedure (From Scratch using NumPy):**
   - Augment feature vectors with bias $z_i = [1, x_i^T]^T$ and normalize with class labels $y_i \in \{+1, -1\}$.
   - Minimize $J(a, b) = \|Y a - b\|^2$ subject to $b > 0$ via the Moore-Penrose pseudoinverse $Y^\dagger$.
   - Iteratively update margin $b(k+1) = b(k) + 2\eta e^+(k)$ and weights $a(k+1) = Y^\dagger b(k+1)$.
   - Record the final weight vector $a = [w_0, w_1, \dots, w_d]^T$ and margin vector $b$ ($\min(b)$, $\max(b)$, $\text{mean}(b)$).
   - Evaluate test partition: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
4. **Convergence & Linear Separability Diagnostics:**
   - Record squared error $\|e(k)\|^2$ and misclassified training samples ($a^T y_i \le 0$) at each iteration $k$.
   - Plot **Iteration $k$ vs. $\|e(k)\|^2$** and **Iteration $k$ vs. Number of misclassified training samples**.
   - Rigorously test and diagnose linear separability using the Ho-Kashyap theorem: detects linear separability when $e(k) \to 0$ ($misc = 0$), and detects linear inseparability when all components $e_i(k) \le 0$ ($e(k) \ne 0$).
5. **Multi-$\eta$ Parameter Analysis:**
   - Sweep learning rate $\eta \in [0.1, 0.9]$ across user-selectable values.
   - Compute classification error rate: $\text{Error Rate} = \frac{\text{Incorrect Predictions}}{\text{Total Predictions}} = 1 - \text{Accuracy}$.
   - Plot **$\eta$ vs. Number of Iterations** (with dual-axis Error Rate curve).
   - Automatically identify and highlight the optimal $\eta$ that converges in the fewest iterations with the lowest error rate.
6. **Prediction of New Instances:**
   - Dynamically generated input form matching the feature schema, pre-filled with dataset training means.
   - Comprehensive input validation: non-empty check, numeric validation, and range warnings for values far outside the training distribution.
   - Descriptive class predictions ($0$ or $1$) and linear discriminant scores $g(x) = a^T [1, x]^T$.
7. **Integration with Previous Classifiers:**
   - Drop-down classifier selector: **Ho-Kashyap Procedure**, **K-Nearest Neighbors (KNN)** (with Elbow Method), **Naive Bayes (GaussianNB)**, and **Comparative Benchmark (All 3 Models Side-by-Side)**.

---

## 📂 Project Directory Structure

```text
PR LAB 8/
├── Pattern Recognition Lab Assignment 08.pdf   # Official lab assignment sheet
├── pr_lab_8.py                                 # Standalone CLI execution & verification script
├── README.md                                   # Complete documentation & 4-Mark Viva Guide
├── DatasetAnalyzer -> Classifiers              # Backward-compatibility symlink
└── Classifiers/                                # Full Flask Web Application
    ├── app.py                                  # Flask backend with Ho-Kashyap, KNN, and Bayes
    ├── requirements.txt                        # Python dependencies
    ├── run.sh                                  # Executable launch script
    ├── test_app.py                             # Automated test suite (unit & integration tests)
    ├── sample_datasets/                        # Standard preloaded CSV benchmarks
    │   ├── Linearly_Separable_2D.csv           # Synthetic separable (converges to e=0)
    │   ├── Non_Linearly_Separable_2D.csv       # Synthetic non-separable (proves inseparability)
    │   ├── heart.csv                           # Cleveland Heart Disease (Binary)
    │   ├── Iris_Binary.csv                     # Fisher Iris (Setosa vs. Versicolor)
    │   ├── Iris_Data.csv                       # Full Iris (3 classes)
    │   └── Naive-Bayes-Classification-Data.csv # Age & Blood Pressure benchmark
    ├── static/
    │   ├── css/
    │   │   └── style.css                       # Responsive UI design system
    │   └── plots/                              # Generated convergence curves & confusion matrices
    ├── templates/
    │   └── index.html                          # Multi-stage web interface
    └── uploads/                                # Isolated session cache & artifacts
```

---

## 🚀 Quick Start Guide

### 1. Environment & Dependencies
Python 3.9+ is supported. Activate your virtual environment or install dependencies:

```bash
cd "PR LAB 8/Classifiers"
pip install -r requirements.txt
```

*Required packages:* `Flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

### 2. Launching the Web Application
Run via the provided `run.sh` script or directly with Python:

```bash
# Option A: Using the launch script
./run.sh

# Option B: Directly via Python
python app.py
```

Open your browser at:
```text
http://127.0.0.1:5001/
```
*(Port `5001` is used by default because port `5000` is reserved by macOS AirPlay Receiver).*

### 3. Automated CLI Verification (`pr_lab_8.py`)
To execute the complete automated verification pipeline from the command line:

```bash
cd "PR LAB 8"
python pr_lab_8.py
```

You can also pass any custom dataset directly:
```bash
python pr_lab_8.py Classifiers/sample_datasets/heart.csv
```

### 4. Running the Automated Test Suite
To verify all unit and integration tests (tests Ho-Kashyap separability, inseparability, endpoints, and predictions):

```bash
cd "PR LAB 8/Classifiers"
python -m unittest test_app.py
```

---

## 📐 Mathematical Formulation of the Ho-Kashyap Procedure

### 1. Data Normalization & Matrix Setup
Given training samples $x_1, \dots, x_N \in \mathbb{R}^d$ and binary class labels $\omega_1$ (Class 1) and $\omega_2$ (Class 0):
1. Augment each sample with a constant bias component:
   $$z_i = [1, x_{i1}, x_{i2}, \dots, x_{id}]^T \in \mathbb{R}^{d+1}$$
2. Normalize sample vectors with respect to classes:
   $$y_i = \begin{cases} +z_i & \text{if } x_i \in \omega_1 \\ -z_i & \text{if } x_i \in \omega_2 \end{cases}$$
3. Stack into the design matrix $Y \in \mathbb{R}^{N \times (d+1)}$:
   $$Y = \begin{bmatrix} y_1^T \\ y_2^T \\ \vdots \\ y_N^T \end{bmatrix}$$

Linear separability requires finding a weight vector $a = [w_0, w_1, \dots, w_d]^T$ such that:
$$Y a > 0 \iff Y a = b \quad \text{for some } b > 0$$

### 2. Criterion Function & Pseudoinverse
Ho and Kashyap minimize the squared error criterion:
$$J(a, b) = \|Y a - b\|^2 = \sum_{i=1}^N (y_i^T a - b_i)^2 \quad \text{subject to } b > 0$$

For any positive margin vector $b$, the optimal least-squares weight vector is:
$$a = Y^\dagger b = (Y^T Y)^{-1} Y^T b$$
where $Y^\dagger$ is the Moore-Penrose pseudoinverse of $Y$.

### 3. Iterative Algorithm & Update Rules
1. **Initialization ($k=0$):**
   - Initialize margin vector $b(0) = \mathbf{1} = [1, 1, \dots, 1]^T > 0$.
   - Compute initial weight vector $a(0) = Y^\dagger b(0)$.
   - Choose learning rate $\eta \in (0, 1]$ (default $\eta = 0.5$).
   - Set convergence tolerance $\epsilon = 10^{-4}$ and maximum iterations $k_{\max} = 1000$.

2. **Iteration Step ($k$):**
   - Compute error vector:
     $$e(k) = Y a(k) - b(k)$$
   - Calculate squared error:
     $$\|e(k)\|^2 = \sum_{i=1}^N e_i(k)^2$$
   - Count misclassified training samples:
     $$\text{Misclassified Count} = \sum_{i=1}^N \mathbb{I}(y_i^T a(k) \le 0) = \sum_{i=1}^N \mathbb{I}((Y a(k))_i \le 0)$$
   - Extract the positive part of the error vector:
     $$e^+(k) = \frac{1}{2}(e(k) + |e(k)|) = \max(e(k), 0)$$
   - Update margin vector $b$ and weight vector $a$:
     $$b(k+1) = b(k) + 2\eta e^+(k) = b(k) + \eta (e(k) + |e(k)|)$$
     $$a(k+1) = Y^\dagger b(k+1)$$
     *Note:* Because $e^+(k) \ge 0$, every component of $b$ is non-decreasing ($b_i(k+1) \ge b_i(k) \ge 1 > 0$), strictly preserving the positivity constraint $b > 0$.

### 4. Convergence & Inseparability Termination Criteria
- **Separable Solution Found:** If $\|e(k)\|_\infty \le \epsilon$ or the misclassification count drops to $0$, a valid separating hyperplane has been discovered.
- **Inseparability Theorem Proof:** If all components of $e(k)$ are non-positive ($e_i(k) \le 0 \quad \forall i$) and $e(k) \ne 0$:
  $$e^+(k) = 0 \implies b(k+1) = b(k)$$
  Ho and Kashyap mathematically proved that this condition is a necessary and sufficient indicator that **no separating vector $a$ exists**. The data is provably **linearly inseparable**.
- **Iteration Limit:** If $k \ge k_{\max}$, the algorithm stops with "Maximum Iterations Reached (Likely Inseparable)".

### 5. Prediction / Decision Rule
For any query instance $x = [x_1, \dots, x_d]^T$:
$$g(x) = a^T [1, x^T]^T = w_0 + \sum_{j=1}^d w_j x_j$$
$$\hat{y} = \begin{cases} \text{Class 1} & \text{if } g(x) > 0 \\ \text{Class 0} & \text{if } g(x) \le 0 \end{cases}$$

---

## 📊 Experimental Results & Benchmark Summary

### 1. Synthetic Linearly Separable Benchmark (`Linearly_Separable_2D.csv`)
- **Status:** **Linearly Separable** (Converged in 1 iteration)
- **Final Weight Vector $a$:** $[w_0 = 0.0133, w_1 = 0.5568, w_2 = 0.4409]^T$
- **Margin Vector $b$:** $\min(b) = 1.0000, \max(b) = 1.0000$ ($b > 0$ satisfied)
- **Test Set Metrics:** Accuracy = 100.00%, Error Rate = 0.0000, F1-Score = 1.0000
- **Optimal Learning Rate:** $\eta = 0.1$ (Fastest convergence, 0 error)

### 2. Real-World Medical Benchmark (`heart.csv`)
- **Status:** **Linearly Inseparable** (Ho-Kashyap non-separability condition $e_i \le 0$ satisfied at iteration 81 for default $\eta=0.5$)
- **Final Weight Vector $a$:** Bias $w_0 = 0.05481$, CP = $+0.37767$, Thalach = $+0.27712$, Sex = $-0.29311$, CA = $-0.26726$, Thal = $-0.19270$, Trestbps = $-0.16319$, Oldpeak = $-0.16348$, Exang = $-0.17261$.
- **Margin Vector $b$:** $\min(b) = 1.0000, \max(b) = 2.52006, \text{mean}(b) = 1.20169$
- **Test Set Metrics:** Accuracy = 77.63%, Error Rate = 0.2237, Precision = 0.7789, Recall = 0.7763, F1-Score = 0.7741
- **Multi-$\eta$ Study:**
  - $\eta = 0.1 \to 424$ iterations (Error Rate = 0.2237)
  - $\eta = 0.5 \to 81$ iterations (Error Rate = 0.2237)
  - $\eta = 0.9 \to 44$ iterations (Error Rate = 0.2237)
  - **Optimal $\eta$ Selected:** $\eta = 0.9$ (Converged in fewest iterations: 44, with lowest error rate: 0.2237)

### 3. Integrated Comparative Benchmark (Heart Disease Dataset)
| Classifier | Learning Paradigm | Accuracy | Error Rate | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **Ho-Kashyap Linear Classifier** | Margin-based Linear Discriminant | **77.63%** | 0.2237 | 0.7789 | 0.7763 | 0.7741 |
| **K-Nearest Neighbors (KNN: K=3)** | Non-parametric Instance-Based | 73.68% | 0.2632 | 0.7368 | 0.7368 | 0.7368 |
| **Naive Bayes (GaussianNB)** | Generative Probabilistic | **80.26%** | 0.1974 | 0.8057 | 0.8026 | 0.8026 |

---

## 🎓 Viva Voce & Theoretical Concepts (4 Marks Preparation)

### Q1: What is the Ho-Kashyap procedure, and how does it fundamentally differ from the Perceptron Learning Algorithm?
**Answer:**
- **Perceptron:** The Perceptron algorithm minimizes the Perceptron criterion function $J_p(a) = \sum_{y \in \mathcal{Y}} (-a^T y)$.
  - **Limitation:** If the training data is linearly separable, the Perceptron converges in finite steps (Perceptron Convergence Theorem). However, if the data is **linearly inseparable**, the Perceptron cycles indefinitely and **cannot detect or prove** non-separability.
- **Ho-Kashyap:** The Ho-Kashyap procedure minimizes the squared error criterion $J(a, b) = \|Y a - b\|^2$ with respect to **both** weight vector $a$ and margin vector $b$, under the strict constraint that $b > 0$.
  - **Key Advantage:** If the classes are linearly separable, Ho-Kashyap converges in finite steps to a separating solution with $b > 0$. If the classes are **linearly inseparable**, all components of the error vector $e(k)$ will eventually become non-positive ($e_i(k) \le 0 \quad \forall i, e \ne 0$), which provides a mathematically rigorous **proof of inseparability**.

---

### Q2: Why does the Ho-Kashyap update rule use $e^+ = \frac{1}{2}(e + |e|)$ instead of updating with the entire error vector $e$?
**Answer:**
The objective is to find $a$ and $b$ such that $Y a = b > 0$.
The error is defined as $e(k) = Y a(k) - b(k)$.
- If $e_i(k) > 0$, it means $(Y a)_i > b_i$. Increasing $b_i$ helps satisfy the equality $Y a = b$ without violating $b_i > 0$.
- If $e_i(k) \le 0$, $(Y a)_i \le b_i$. If we were to decrease $b_i$, $b_i$ could become zero or negative, violating the fundamental constraint $b > 0$.
Therefore, we only adjust components where $e_i > 0$.
The term $e^+ = \frac{1}{2}(e + |e|) = \max(e, 0)$ extracts only the positive components of $e$.
Thus, $b(k+1) = b(k) + 2\eta e^+(k)$ ensures that $b$ is non-decreasing ($b_i(k+1) \ge b_i(k) \ge b_i(0) > 0$), strictly maintaining $b > 0$.

---

### Q3: How does the Ho-Kashyap procedure prove whether a dataset is linearly separable or inseparable?
**Answer:**
At any iteration $k$:
1. **Linear Separability:** If $\|e(k)\| \le \epsilon$ or all training samples satisfy $y_i^T a(k) > 0$ (zero misclassifications), then $a(k)$ is a valid linear separating hyperplane.
2. **Linear Inseparability:** If $e_i(k) \le 0$ for all $i = 1, \dots, N$ (with $e(k) \ne 0$):
   Then $e^+(k) = 0$, so $b(k+1) = b(k)$, which means the algorithm can make no further progress.
   Mathematically, since $e = (Y Y^\dagger - I)b$, this implies $(I - Y Y^\dagger)b \ge 0$. Multiplying by $b^T$ reveals that no positive vector $b > 0$ can lie in the range space of $Y$.
   Hence, by the Ho-Kashyap theorem, **no separating hyperplane exists**, proving linear inseparability.

---

### Q4: Why is the learning rate $\eta$ restricted to $0 < \eta \le 1$ (or $0 < \eta < 2$)? How does $\eta$ affect convergence?
**Answer:**
- Mathematically, the sequence $\|e(k)\|^2$ is guaranteed to converge monotonically to zero for separable data if $0 < \eta \le 1$ (and generally for $0 < \eta < 2$ from descent properties).
- If $\eta$ is too small (e.g., $\eta = 0.01$), the margin adjustments are very tiny, requiring hundreds or thousands of iterations to converge.
- If $\eta$ is within $[0.5, 0.9]$, convergence occurs in significantly fewer iterations.
- In our multi-$\eta$ experiments on `heart.csv`, increasing $\eta$ from 0.1 to 0.9 reduced the required iterations from 424 down to 44, selecting $\eta = 0.9$ as the optimal learning rate.

---

### Q5: Why is numerical feature standardization ($z = \frac{x - \mu}{\sigma}$) so important before training Ho-Kashyap?
**Answer:**
1. **Condition Number of $Y^T Y$:** The Ho-Kashyap procedure computes the Moore-Penrose pseudoinverse $Y^\dagger = (Y^T Y)^{-1} Y^T$. If features have widely differing scales (e.g., Age $[20-80]$ vs. Cholesterol $[150-400]$), the matrix $Y^T Y$ becomes severely ill-conditioned (large condition number), leading to numerical instability, precision loss, and overflow in matrix multiplication.
2. **Equal Margin Contribution:** Standardization ensures that the distance metric and margin adjustments treat all feature dimensions symmetrically rather than allowing high-magnitude features to dominate.

---

### Q6: How does Ho-Kashyap compare fundamentally with KNN and Naive Bayes?
**Answer:**
| Dimension | Ho-Kashyap Classifier | K-Nearest Neighbors (KNN) | Naive Bayes (GaussianNB) |
|---|---|---|---|
| **Decision Boundary** | Linear Hyperplane ($w^T x + w_0 = 0$) | Highly non-linear, locally adaptive | Quadratic/Linear depending on covariance |
| **Model Type** | Discriminative, Margin-based | Non-parametric, Instance-based (Lazy) | Generative, Probabilistic |
| **Separability Testing**| Explicit proof of inseparability ($e \le 0$) | Cannot test linear separability | Cannot test linear separability |
| **Training Complexity** | $\mathcal{O}(k_{\max} \cdot N \cdot d)$ | $\mathcal{O}(1)$ (stores training points) | $\mathcal{O}(N \cdot d)$ (computes mean/variance) |
| **Inference Speed** | Very fast ($\mathcal{O}(d)$ dot product) | Slow ($\mathcal{O}(N \cdot d)$ distance search) | Very fast ($\mathcal{O}(C \cdot d)$ likelihoods) |
| **Feature Scaling** | Critical for pseudoinverse stability | Critical for distance calculations | Insensitive (independent marginals) |

---

### Q7: What is the Classification Error Rate formula used in Assignment 08?
**Answer:**
$$\text{Error Rate} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Predictions}} = \frac{\text{False Positives} + \text{False Negatives}}{\text{Total Test Samples}} = 1 - \text{Accuracy}$$
In the multi-$\eta$ study, the optimal $\eta$ is chosen as the value that achieves the **lowest error rate** in the **fewest iterations**.

