# 💳 Monobank ML — Spending Analysis & Prediction

A machine learning project built on personal Monobank bank statement data from 2025.

Two levels of implementation:
- **From scratch** — pure Python, NumPy, Pandas (no ML frameworks)
- **Library-based** — sklearn, XGBoost, LightGBM, CatBoost

---

## 🎯 Tasks

| Task | Type |
|------|------|
| Predict spending category for the day (0-200 / 200-600 / 600+ UAH) | Classification |
| Predict total spending amount for the day | Regression |

---

## 📁 Project Structure

```
monobank-ml/
├── data_set/
│   └── monobank_clean.csv          # anonymized bank statement
│
├── models/                         # scratch implementations (NumPy only)
│   ├── regression.py               # linear regression + neural network
│   ├── classification.py           # neural network (classification)
│   ├── decision_tree.py            # decision tree (classification)
│   ├── decision_tree_regression.py # decision tree (regression)
│   ├── knn.py                      # K-Nearest Neighbors
│   ├── random_forest.py            # random forest
│   ├── gradient_boosting.py        # Gradient Boosting
│   ├── xgboost1.py                 # XGBoost
│   ├── lightgbm1.py                # LightGBM
│   └── catboost1.py                # CatBoost
│
├── comparison.py                   # all scratch models — single comparison table
├── sklearn_comparison.py           # sklearn + libraries vs scratch
└── README.md
```

> ⚠️ Boosting files are suffixed with `1` to avoid name conflicts with the actual libraries when running `sklearn_comparison.py`.

---

## 📊 Data

- **Source:** personal Monobank statement (CSV)
- **Period:** 2025
- **Transactions:** 611
- **Days with spending:** 292
- **Names and phone numbers anonymized** before publishing

### Features

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0=Mon … 6=Sun |
| `month` | 1–12 |
| `weekend` | 1 if Sat/Sun |
| `count` | number of transactions that day |
| `yesterday` | spending the previous day (UAH) |
| `two_days_ago` | spending two days ago (UAH) |

### Classification Classes

| Class | Range | Days in test set |
|-------|-------|-----------------|
| 0 | 0–200 UAH | 23 |
| 1 | 200–600 UAH | 16 |
| 2 | 600+ UAH | 20 |

---

## 🏆 Results

### Classification — Scratch Implementations

Train: 233 days | Test: 59 days | Baseline: 33.3%

| Model | Accuracy | Train | Parameters |
|-------|----------|-------|------------|
| 🥇 Random Forest | **78.0%** | — | n=6, seed=13 |
| Decision Tree | 76.3% | — | depth=3 |
| XGBoost | 74.6% | 75.5% | lr=0.01, n=200, λ=2.0, γ=0.5 |
| Gradient Boosting | 71.2% | 78.1% | lr=0.01, n=150, depth=4 |
| LightGBM | 71.2% | 68.2% | lr=0.01, n=380, leaves=15 |
| CatBoost | 71.2% | 74.7% | lr=0.01, n=180, l2=10.0 |
| Neural Network | 67.8% | — | 6→10→6→3, seed=42 |
| KNN | 61.0% | — | k=3 |

### Classification — sklearn + Libraries (default parameters)

| Model | Accuracy |
|-------|----------|
| 🥇 Decision Tree | **76.3%** |
| XGBoost | 72.9% |
| CatBoost | 69.5% |
| Random Forest | 66.1% |
| LightGBM | 66.1% |
| Gradient Boosting | 64.4% |
| KNN | 61.0% |
| Logistic Regression | 59.3% |
| MLP | 33.9% |

### Scratch vs Library Implementation

| Model | Scratch | Library |
|-------|---------|---------|
| KNN | 61.0% | 61.0% |
| Decision Tree | 76.3% | 76.3% |
| Random Forest | **78.0%** | 66.1% |
| Neural Network | 67.8% | 33.9% |
| Gradient Boosting | 71.2% | 64.4% |
| XGBoost | **74.6%** | 72.9% |
| LightGBM | 71.2% | 66.1% |
| CatBoost | 71.2% | 69.5% |

> Scratch versions outperform library defaults not because they are better — but because parameters were tuned manually, while library models ran with default values.

### Regression

| Model | RMSE |
|-------|------|
| 🥇 Linear Regression | **1272 UAH** |
| CatBoost | 1428 UAH |
| Random Forest | 1434 UAH |
| LightGBM | 1470 UAH |
| XGBoost | 1545 UAH |
| Gradient Boosting | 1626 UAH |
| Decision Tree | 1674 UAH |
| MLP | 1756 UAH |

---

## 🔬 Architectures

**Neural Network (classification):** `6 → 10 → 6 → 3` with ReLU + Softmax

**Neural Network (regression):** `6 → 6 → 3 → 1` with ReLU

**Decision Tree:** Gini / MSE criterion, recursive splitting, max_depth=3

**Random Forest:** bagging + random feature subsampling, n=6, majority voting

**Gradient Boosting:** sequential trees on residuals `F_m = F_{m-1} + η·h_m`

**XGBoost:** GB + hessian `hᵢ = pᵢ(1-pᵢ)` + L2/γ regularization + optimal leaf value `w* = -ΣG/(ΣH+λ)`

**LightGBM:** XGBoost + leaf-wise tree growth + GOSS sampling

**CatBoost:** Ordered Target Encoding for categorical features + Symmetric Trees with cumsum optimization

---

## ⚙️ Implemented from Scratch

- Forward pass and Backpropagation, SGD, ReLU, Softmax
- Cross-Entropy and MSE loss functions, Early Stopping
- Decision Tree (Gini and MSE, recursive splitting)
- KNN (Manhattan distance, majority voting)
- Random Forest (bagging, feature subsampling)
- Gradient Boosting (residuals, sequential accumulation)
- XGBoost (hessian, Gain formula, L2 + γ, cumsum)
- LightGBM (leaf-wise growth, GOSS)
- CatBoost (Ordered Encoding, Symmetric Trees)

---

## 💡 Key Findings

**Dataset ceiling — ~78%.** Three fundamentally different algorithms (GB, LightGBM, CatBoost) with different parameters all converge to 71.2% on the test set. This signals a data problem, not a model problem.

**Regression is practically unsolvable on this data.**
RMSE of 1272 UAH against a mean of 851 UAH and std of 1535 UAH — the model is barely better than a naive baseline. Personal spending is too chaotic.

**Simple models win on small datasets.**
Random Forest (78%) and Decision Tree (76%) beat XGBoost (74.6%). Complexity does not compensate for lack of data.

**Hyperparameters matter enormously.**
XGBoost without regularization: train 95.7% → test 62.7%. With λ=2.0, γ=0.5: train 75.5% → test 74.6%. The difference is two numbers.

**Scratch ≈ library at equal parameters.**
With tuning, the gap is 1-3% — implementation details, not fundamental differences.

---

## 🛠 Tech Stack

- **Python 3.11+**
- **NumPy** — matrix operations, all scratch implementations
- **Pandas** — data loading and preprocessing
- **scikit-learn** — library models and normalization
- **XGBoost / LightGBM / CatBoost** — library boosting
