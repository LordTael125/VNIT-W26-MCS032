# Pattern Recognition Lab - Assignment 09
## Web-based Machine Learning Application: Non-Parametric Classification using the Parzen Window Density Estimation Method & Multi-Classifier Studio Integration

**Course:** CSL517 - Pattern Recognition  
**Lab Session Date:** 17th November 2026  
**Evaluation Scheme (Total 10 Marks):**
- **Attendance:** 02 Marks
- **Assignment Completion & Correctness:** 04 Marks
- **Viva Voce:** 04 Marks

---

## 📌 Executive Summary
This project provides a complete, mathematically rigorous, and production-ready solution fulfilling all requirements of **Pattern Recognition Lab: Assignment 09** (Department of Computer Science & Engineering).

Built using **Flask**, **NumPy** (from-scratch Parzen Window algorithm implementation), **Pandas**, **Scikit-Learn**, **Matplotlib**, and **Seaborn**, the application delivers both an interactive web-based ML studio and a standalone CLI pipeline that:
1. **Uploads & Inspects Any Classification Dataset:** Supports custom CSV files and bundled benchmarks (Iris 3-Class, Iris Binary, Cleveland Heart Disease, Linearly Separable 2D, Non-Linearly Separable 2D, and Naive Bayes benchmark).
2. **Pre-Training Preparation & Inspection:** Automatically detects dataset status. If preprocessing is required (missing values, duplicates, categorical columns), exposes controls for median/mode imputation, deduplication, and label encoding or exclusion. If the dataset is clean, renders an informative status message. Exposes a user choice whether to standardize numerical features via `StandardScaler`.
3. **From-Scratch Parzen Window Density Estimator (NumPy):**
   - Implements both **Hypercube** and **Gaussian** window functions.
   - Solves numerical underflow for small $h$ or high feature dimensions $d$ using the **Log-Sum-Exp** trick.
   - Tracks samples whose hypercube windows contain zero training points (empty windows) and gracefully falls back to prior probabilities.
   - Computes prior probabilities $P(\omega_j)$, class-conditional densities $p(x | \omega_j)$, and posterior probabilities $P(\omega_j | x)$ via Bayes decision theory.
4. **Default Parameters Run ($h = 1.0$, Gaussian Window):**
   - Records prior probabilities $P(\omega_j)$, window function, and window width $h=1.0$.
   - Displays a structured table of estimated densities $p(x | \omega_j)$ and posteriors $P(\omega_j | x)$ for the first five test samples.
   - Evaluates test-set Accuracy, Error Rate, Macro Precision, Macro Recall, Macro F1-Score, and plots the Confusion Matrix.
5. **Window Width ($h$) Analysis:**
   - Evaluates a multi-$h$ sweep ($h \in [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]$) for both Gaussian and Hypercube window functions.
   - Records training error rate, validation error rate, and the count of validation samples falling in empty hypercube windows.
   - Plots **$h$ vs. Error Rate** on a **logarithmic scale for $h$**, displaying all four curves (Gaussian Train/Val, Hypercube Train/Val) with the optimal point highlighted.
   - Selects the optimal configuration $(window^*, h^*)$ minimizing validation error rate, evaluates it on the test partition, and presents a side-by-side comparison table against the default model.
6. **Prediction of New Instances:**
   - Features a dynamic input form generated from feature columns, pre-filled with dataset training means.
   - Performs comprehensive input validation (non-empty, numeric format, and distribution warnings for values $>3\sigma$ from training means).
   - Applies the fitted pre-processing pipeline and outputs predicted class labels, posterior probability bars, and class-conditional densities.
7. **Multi-Classifier Integration:**
   - Integrates with the **Classifiers Web Application** developed across Labs 7 & 8, featuring **Parzen Window**, **Bayes Classifier (GaussianNB)**, **K-Nearest Neighbors (KNN with Elbow Method)**, and **Ho-Kashyap Linear Classifier**.
   - Includes a **Comprehensive Benchmark Mode** comparing all four classifiers side-by-side on the same test set with comparison tables and bar charts.

---

## 📐 Mathematical Foundations

### 1. Parzen Window Density Estimation (Kernel Density Estimation)
Given $N_j$ independent, identically distributed training samples $x_{j,1}, x_{j,2}, \dots, x_{j, N_j} \in \mathbb{R}^d$ belonging to class $\omega_j$, the Parzen Window estimate of the class-conditional probability density function $p(x | \omega_j)$ is:

$$p_n(x | \omega_j) = \frac{1}{N_j V_n} \sum_{i=1}^{N_j} \phi\left( \frac{x - x_{j,i}}{h} \right)$$

where $V_n = h^d$ is the volume of the hypercube of side length $h$, and $\phi(u)$ is a valid window (kernel) function satisfying:

$$\phi(u) \ge 0 \quad \text{and} \quad \int_{\mathbb{R}^d} \phi(u) \, du = 1$$

### 2. Window Functions

#### A. Hypercube Window Function
$$\phi(u) = \begin{cases} 1 & \text{if } |u_k| \le \frac{1}{2} \text{ for all } k = 1, \dots, d \\ 0 & \text{otherwise} \end{cases}$$

A training sample $x_{j,i}$ falls inside the window centered at test point $x$ if and only if:

$$\|x - x_{j,i}\|_\infty = \max_{1 \le k \le d} |x_k - x_{j,i,k}| \le \frac{h}{2}$$

Let $k_j(x) = \sum_{i=1}^{N_j} \mathbb{I}\left( \|x - x_{j,i}\|_\infty \le \frac{h}{2} \right)$ be the count of class $\omega_j$ samples in the hypercube. The estimated density is:

$$p(x | \omega_j) = \frac{k_j(x)}{N_j h^d}$$

> [!WARNING]
> **The Empty Window Problem & The Curse of Dimensionality:**
> In $d$-dimensional space, the hypercube volume is $V = h^d$. When $h$ is small (e.g., $h=0.05, d=4 \implies V = 6.25 \times 10^{-6}$), the probability of any training point landing inside the tiny box vanishes. If $\sum_j k_j(x) = 0$, all class densities evaluate to $0$. The application detects these empty windows and falls back to class priors $P(\omega_j)$.

#### B. Multivariate Isotropic Gaussian Window Function
$$\phi(u) = \frac{1}{(2\pi)^{d/2}} \exp\left( -\frac{1}{2} \|u\|^2 \right)$$

The density estimate becomes:

$$p(x | \omega_j) = \frac{1}{N_j (2\pi)^{d/2} h^d} \sum_{i=1}^{N_j} \exp\left( -\frac{\|x - x_{j,i}\|^2}{2 h^2} \right)$$

Because the Gaussian kernel has infinite support, $p(x | \omega_j) > 0$ everywhere, avoiding the empty-window discontinuity of the hypercube.

---

### 3. Log-Sum-Exp Trick to Prevent Underflow
When $h$ is small or feature dimensionality $d$ is high, the exponent $-\frac{\|x - x_{j,i}\|^2}{2 h^2}$ becomes an extremely large negative number (e.g., $-5000$). In standard 64-bit floating point arithmetic, $\exp(-5000)$ underflows to `0.0`, causing severe division-by-zero errors when normalizing posterior probabilities.

To guarantee complete numerical stability, computations are carried out in log-space:

$$\ln p(x | \omega_j) = -\ln(N_j) - \frac{d}{2}\ln(2\pi) - d \ln(h) + \ln \sum_{i=1}^{N_j} \exp(z_i)$$

where $z_i = -\frac{\|x - x_{j,i}\|^2}{2 h^2} \le 0$. The Log-Sum-Exp function is evaluated as:

$$\text{LSE}(z) = m + \ln \sum_{i=1}^{N_j} \exp(z_i - m), \quad \text{where } m = \max_{1 \le i \le N_j} z_i$$

Since $z_i - m \le 0$, $\exp(z_i - m) \in (0, 1]$, and for the nearest training sample $z_i = m \implies \exp(0) = 1$. The sum is at least $1$, completely eliminating underflow and NaN values.

The posterior probability $P(\omega_j | x)$ is likewise computed stably in log-space:

$$\ln \text{joint}_j = \ln p(x | \omega_j) + \ln P(\omega_j)$$

$$\ln P(\omega_j | x) = \ln \text{joint}_j - \text{LSE}_{k=1}^C (\ln \text{joint}_k)$$

$$P(\omega_j | x) = \exp\left( \ln P(\omega_j | x) \right)$$

---

### 4. Bayes Decision Rule
Using the estimated class-conditional densities and class priors $P(\omega_j) = \frac{N_j}{N}$, the Bayes decision rule assigns sample $x$ to class $\omega^*$:

$$\omega^* = \arg\max_{\omega_j} P(\omega_j | x) = \arg\max_{\omega_j} \left[ p(x | \omega_j) P(\omega_j) \right]$$

---

### 5. Effect of Window Width $h$ & Bias-Variance Trade-Off
The parameter $h$ acts as a smoothing bandwidth:
- **Small $h$ (High Variance, Low Bias):** The density estimate is a sum of sharp, isolated spikes centered at each training point. On the training set, error rate approaches $0$. On unseen test data, hypercube windows become empty and Gaussian weights drop precipitously, causing severe overfitting.
- **Large $h$ (Low Variance, High Bias):** The density estimate is heavily oversmoothed, blurring multimodal structures and flattening class boundaries. The classifier loses discriminative ability, underfitting the data.
- **Optimal $h^*$:** Balances bias and variance to minimize generalization error on the validation partition.

---

## 📂 Project Directory Structure

```text
PR LAB 9/
├── PR_Lab_Assignment_09_Parzen_Window.pdf   # Official lab assignment sheet
├── pr_lab_9.py                                 # Standalone CLI execution & verification script
├── run.sh                                      # Root executable launcher script
├── README.md                                   # Complete documentation & 4-Mark Viva Guide
├── Iris.csv                                    # Root benchmark copy
├── DatasetAnalyzer -> Classifiers              # Backward-compatibility symlink
└── Classifiers/                                # Full Flask Web Application
    ├── app.py                                  # Flask backend with Parzen Window, Bayes, KNN, Ho-Kashyap
    ├── requirements.txt                        # Python dependencies
    ├── run.sh                                  # Subdirectory launch script
    ├── test_app.py                             # Automated test suite (5 unit & integration tests)
    ├── sample_datasets/                        # Bundled preloaded CSV benchmarks
    │   ├── Iris_Data.csv                       # Fisher Iris (3 classes, 4 features)
    │   ├── Iris_Binary.csv                     # Iris Setosa vs. Versicolor
    │   ├── heart.csv                           # Cleveland Heart Disease (Binary, 13 features)
    │   ├── Linearly_Separable_2D.csv           # Synthetic separable benchmark
    │   ├── Non_Linearly_Separable_2D.csv       # Synthetic concentric benchmark
    │   └── Naive-Bayes-Classification-Data.csv # Age & Blood Pressure benchmark
    ├── static/
    │   ├── css/
    │   │   └── style.css                       # Modern responsive UI design
    │   └── plots/                              # Generated curves, heatmaps & benchmark charts
    ├── templates/
    │   └── index.html                          # Multi-stage interactive web interface
    └── uploads/                                # Cached session artifacts
```

---

## 🚀 Quick Start Guide

### 1. Environment & Dependencies
Python 3.9+ is supported. Activate your virtual environment or install dependencies:

```bash
cd "PR LAB 9/Classifiers"
pip install -r requirements.txt
```

*Required packages:* `Flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

### 2. Launching the Web Application
Launch using the root or classifier launch script:

```bash
# Option A: From root directory
cd "PR LAB 9"
./run.sh

# Option B: Directly via Python
cd "PR LAB 9/Classifiers"
python3 app.py
```

Open your browser at:
```text
http://127.0.0.1:5002/
```

### 3. Running the Standalone CLI Pipeline
To run the automated verification script on any dataset without launching a browser:

```bash
cd "PR LAB 9"
python3 pr_lab_9.py
```

To run on a custom dataset with specific options:
```bash
python3 pr_lab_9.py --dataset "Classifiers/sample_datasets/heart.csv" --target "target"
```

### 4. Running the Automated Test Suite
To execute the test suite (covering log-sum-exp stability, hypercube empty windows, classification accuracy, and full web routes):

```bash
cd "PR LAB 9/Classifiers"
python3 test_app.py
```

---

## 🔬 Benchmark Results on Fisher Iris Dataset

When running the pipeline on the standard Fisher Iris benchmark (60% Train, 20% Validation, 20% Test):

### 1. Default Parameters Run (Gaussian Window, $h = 1.0$)
- **Prior Probabilities:**
  - $P(\text{Iris-setosa}) = 0.3218$
  - $P(\text{Iris-versicolor}) = 0.3448$
  - $P(\text{Iris-virginica}) = 0.3333$
- **First 5 Test Samples Densities & Posteriors:**
  - Sample 1 (Versicolor): $p(x|\text{versicolor}) = 1.59 \times 10^{-2}, P(\text{versicolor}|x) = 64.0\%$ (Correct)
  - Sample 2 (Versicolor): $p(x|\text{versicolor}) = 1.32 \times 10^{-2}, P(\text{versicolor}|x) = 68.0\%$ (Correct)
  - Sample 3 (Virginica): $p(x|\text{virginica}) = 1.22 \times 10^{-2}, P(\text{virginica}|x) = 74.0\%$ (Correct)
  - Sample 4 (Setosa): $p(x|\text{setosa}) = 1.17 \times 10^{-2}, P(\text{setosa}|x) = 97.0\%$ (Correct)
  - Sample 5 (Virginica): $p(x|\text{virginica}) = 1.38 \times 10^{-2}, P(\text{virginica}|x) = 61.0\%$ (Correct)
- **Test Accuracy:** **86.67%** (Error Rate: 13.33%)
- **Macro F1-Score:** **86.53%**

### 2. Window Width ($h$) Sweep Analysis
| Window Width $h$ | Gaussian Train Err | Gaussian Val Err | Hypercube Train Err | Hypercube Val Err | Hypercube Empty Windows (Val) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 0.00% | 6.67% | 0.00% | 66.67% | 30 / 30 |
| 0.05 | 0.00% | 6.67% | 0.00% | 66.67% | 30 / 30 |
| 0.10 | 0.00% | 6.67% | 0.00% | 66.67% | 30 / 30 |
| 0.20 | 0.00% | 6.67% | 0.00% | 66.67% | 30 / 30 |
| 0.50 | 3.45% | **0.00%** | 0.00% | 30.00% | 13 / 30 |
| **1.00** | 8.05% | 13.33% | 2.30% | **0.00%** | 1 / 30 |
| 2.00 | 9.20% | 13.33% | 14.94% | 16.67% | 0 / 30 |
| 5.00 | 16.09% | 13.33% | 42.53% | 30.00% | 0 / 30 |

### 3. Side-by-Side Model Comparison (Test Set)
| Evaluation Metric | Default Model (Gaussian, $h = 1.0$) | Optimal Model (Hypercube, $h^* = 1.0$) | Improvement ($\Delta$) |
|---|:---:|:---:|:---:|
| **Window Function** | Gaussian | Hypercube | - |
| **Window Width ($h$)** | 1.0 | 1.0 | - |
| **Test Accuracy** | 86.67% | **96.67%** | **+10.00%** |
| **Test Error Rate** | 13.33% | **3.33%** | **-10.00%** |
| **Macro Precision** | 87.50% | **96.97%** | **+9.47%** |
| **Macro Recall** | 86.67% | **96.67%** | **+10.00%** |
| **Macro F1-Score** | 86.53% | **96.66%** | **+10.13%** |

### 4. Multi-Classifier Benchmark Comparison
| Classifier Architecture | Test Accuracy | Macro F1-Score | Remarks |
|---|:---:|:---:|---|
| **Parzen Window (Optimal)** | **96.67%** | **96.66%** | Non-parametric density estimation ($h^* = 1.0$) |
| **Gaussian Naive Bayes** | **96.67%** | **96.66%** | Parametric Gaussian with feature independence assumption |
| **K-Nearest Neighbors ($k=5$)** | **93.33%** | **93.27%** | Non-parametric instance-based classifier |

---

## 🎓 4-Mark Viva Voce Preparation Guide

### Q1. What is the fundamental difference between parametric and non-parametric density estimation?
**Answer:**
- In **parametric density estimation**, we assume the underlying distribution belongs to a known family of parameterized functions (e.g., Gaussian $\mathcal{N}(\mu, \Sigma)$) and estimate the fixed number of parameters from the data. If the true data distribution violates this assumption (e.g., multimodal, ring-shaped, or skewed data), parametric models perform poorly.
- In **non-parametric density estimation** (such as Parzen Windows and KNN), no prior functional form is assumed for the density. The shape of the density is estimated directly from the training samples, allowing arbitrary, multimodal, and non-linear distributions to be captured.

---

### Q2. What are the mathematical conditions required for a Parzen window function $\phi(u)$ to yield a valid probability density?
**Answer:**
A window function $\phi(u)$ must satisfy two conditions:
1. **Non-negativity:** $\phi(u) \ge 0$ for all $u \in \mathbb{R}^d$.
2. **Unit integral:** $\int_{\mathbb{R}^d} \phi(u) \, du = 1$.
Because $p_n(x) = \frac{1}{n h^d} \sum_{i=1}^n \phi\left(\frac{x - x_i}{h}\right)$ is a convex linear combination of normalized kernel functions, these conditions ensure that:
$$p_n(x) \ge 0 \quad \text{and} \quad \int_{\mathbb{R}^d} p_n(x) \, dx = \frac{1}{n} \sum_{i=1}^n \int_{\mathbb{R}^d} \frac{1}{h^d} \phi\left(\frac{x - x_i}{h}\right) dx = \frac{1}{n} \sum_{i=1}^n 1 = 1$$

---

### Q3. Why does the hypercube window function cause empty windows, and how does window width $h$ affect this?
**Answer:**
The hypercube window has compact (bounded) support: $\phi(u) = 1$ only if $\|u\|_\infty \le \frac{1}{2}$, and $0$ elsewhere.
In $d$ dimensions, the hypercube centered at test point $x$ spans $[x_k - h/2, x_k + h/2]$ for each dimension $k$.
If no training point falls within this box, the count $k(x) = 0$, and the estimated density $p(x) = 0$.
When $h$ is very small, the volume $V = h^d$ shrinks exponentially with $d$ (curse of dimensionality), causing almost all test samples to fall into empty windows. As $h$ increases, the window expands to encompass nearby training points, eliminating empty windows.

---

### Q4. Why is feature standardization (Z-score scaling) critical for the Parzen Window method?
**Answer:**
The Parzen Window kernel evaluates distances between points $\frac{x - x_i}{h}$. An isotropic kernel uses a single scalar bandwidth $h$ for all $d$ dimensions. If one feature has values in $[0, 1]$ (e.g., petal width) while another feature has values in $[100, 1000]$ (e.g., blood pressure), the feature with larger scale dominates the Euclidean or Chebyshev distance, rendering the small-scale feature irrelevant.
Standardization ensures that each feature has zero mean and unit variance, allowing isotropic bandwidth $h$ to act uniformly across all feature dimensions.

---

### Q5. What is numerical underflow in Gaussian Parzen Window estimation, and how does the Log-Sum-Exp trick solve it?
**Answer:**
In Gaussian Parzen estimation, the density contribution of each training sample is proportional to $\exp\left(-\frac{\|x - x_i\|^2}{2 h^2}\right)$. When $h$ is small (e.g., $h=0.01$) or $d$ is large, the squared distance divided by $2h^2$ can easily exceed $709$. Since $\exp(-709) \approx 10^{-308}$, any larger negative exponent underflows to standard floating-point `0.0`.
The Log-Sum-Exp trick solves this by subtracting the maximum exponent $m = \max_i z_i$:
$$\ln \sum_{i=1}^n \exp(z_i) = m + \ln \sum_{i=1}^n \exp(z_i - m)$$
Because $z_i - m \le 0$, the maximum term becomes $\exp(0) = 1$, ensuring the sum inside the logarithm is always $\ge 1$, which completely prevents underflow, zero division, and NaN values.

---

### Q6. How does Parzen Window classification differ from K-Nearest Neighbors (KNN)?
**Answer:**
Both are non-parametric density estimation methods, but they approach the bias-variance trade-off from opposite directions:
1. **Parzen Window Method:**
   - **Fixes the volume $V = h^d$** (by choosing window width $h$).
   - Counts the random number of samples $k(x)$ falling inside this fixed volume.
   - Disadvantage: In sparse regions, $k=0$ (empty windows); in dense regions, points may be oversmoothed.
2. **K-Nearest Neighbors Method:**
   - **Fixes the number of samples $k$**.
   - Expands or shrinks the volume $V(x)$ until it encloses exactly $k$ neighbors.
   - Disadvantage: Can lead to densities that do not integrate to 1 over all space.

---

### Q7. Explain the asymptotic convergence properties of the Parzen Window density estimator.
**Answer:**
According to the Parzen-Rosenblatt theorem, the estimated density $p_n(x)$ converges in probability to the true density $p(x)$ at points of continuity if and only if three conditions hold as sample size $n \to \infty$:
1. $\lim_{n \to \infty} V_n = \lim_{n \to \infty} h_n^d = 0$ (window width shrinks to zero to eliminate bias).
2. $\lim_{n \to \infty} n V_n = \lim_{n \to \infty} n h_n^d = \infty$ (number of points in the window grows to infinity to eliminate variance).
3. $\sup_u |\phi(u)| < \infty$ and $\lim_{\|u\| \to \infty} \|u\|^d |\phi(u)| = 0$.

---

### Q8. What happens to the Gaussian Parzen Window classifier when $h \to 0$ and when $h \to \infty$?
**Answer:**
- **When $h \to 0$:** Each Gaussian kernel converges to a Dirac delta function $\delta(x - x_i)$ centered at training sample $x_i$. For any query point $x$, the single nearest training neighbor dominates the exponent, causing the Parzen Window classifier to behave identically to the **1-Nearest Neighbor (1-NN)** classifier!
- **When $h \to \infty$:** The Gaussian kernels become infinitely wide and flat: $\phi\left(\frac{x - x_i}{h}\right) \to \text{constant}$. The estimated density $p(x | \omega_j)$ becomes identical for all $x$, and the posterior probability collapses entirely to the class prior: $P(\omega_j | x) \to P(\omega_j)$. The classifier simply predicts the majority class.

---

### Q9. Why does the hypercube window exhibit jagged, discontinuous density estimates, whereas the Gaussian window produces smooth estimates?
**Answer:**
The estimated density inherits the mathematical smoothness and differentiability of the underlying window function $\phi(u)$.
- The hypercube window is a discontinuous step function; as a point $x$ moves across the boundary where a training sample enters or exits the box $[x - h/2, x + h/2]$, $k(x)$ abruptly jumps by $\pm 1$, causing step discontinuities in $p(x)$.
- The Gaussian window is infinitely differentiable ($C^\infty$ smooth), so the resulting density estimate $p(x)$ is continuous and smooth everywhere.

---

### Q10. How is multi-class classification handled in the Parzen Window framework?
**Answer:**
The dataset is partitioned into subsets by class label: $D_1, D_2, \dots, D_C$, where $D_j$ contains $N_j$ samples of class $\omega_j$.
A separate Parzen Window density estimator $p_n(x | \omega_j)$ is constructed for each class using only that class's training data.
The prior probabilities are computed as $P(\omega_j) = \frac{N_j}{N}$.
Using Bayes rule, the posterior probability is:
$$P(\omega_j | x) = \frac{p_n(x | \omega_j) P(\omega_j)}{\sum_{k=1}^C p_n(x | \omega_k) P(\omega_k)}$$
The sample is assigned to the class with the highest posterior probability (or equivalently, the highest joint density $p(x | \omega_j) P(\omega_j)$).

---

### Q11. What is the Curse of Dimensionality in kernel density estimation, and how does it manifest in Parzen Windows?
**Answer:**
In $d$-dimensional space, volume grows exponentially with dimension: $V = h^d$.
To maintain a fixed density of points within a window of width $h$, the required number of training samples grows as $\mathcal{O}(c^d)$ (exponential in $d$).
In high dimensions, the space is exceedingly sparse: almost all points lie near the boundary of the space, and pairwise distances concentrate around the mean.
Consequently:
- For hypercube kernels, windows become almost universally empty unless $h$ is set exceedingly large.
- For Gaussian kernels, distance terms in the exponent become very large, requiring careful bandwidth tuning and numerical log-sum-exp stabilization.

---

### Q12. How does the validation error rate guide the selection of the optimal window width $h^*$?
**Answer:**
We cannot use the training error rate to select $h$, because training error monotonically decreases toward $0$ as $h \to 0$ (overfitting).
By partitioning the dataset into **Train (60%)**, **Validation (20%)**, and **Test (20%)**:
1. Models are trained on the training partition for various candidate $h$ values.
2. Each model is evaluated on the independent validation partition to measure generalization error rate.
3. The curve of $h$ vs. Validation Error Rate exhibits a U-shape: high error for small $h$ (high variance/empty windows), drops to a minimum at $h^*$, and rises for large $h$ (oversmoothing/high bias).
4. The $(window^*, h^*)$ yielding the minimum validation error rate is chosen and finally tested once on the untouched test partition to obtain an unbiased estimate of true performance.

