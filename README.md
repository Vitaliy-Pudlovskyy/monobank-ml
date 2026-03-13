# 💳 Monobank ML — Personal Expense Analysis & Prediction

A machine learning project built on my personal Monobank transaction history for 2025. Implemented from scratch — no ML frameworks, just Python, NumPy and Pandas.

---

## 🎯 Tasks

| Task | Type | File |
|------|------|------|
| Predict daily expense amount | Regression | `regression.py`, `decision_tree_regression.py` |
| Predict daily expense category | Classification | `classification.py`, `decision_tree.py`, `knn.py`, `random_forest.py` |
| Compare all models | Both | `comparison.py` |

---

## 📁 Project Structure

```
monobank-ml/
├── data/
│   └── monobank_clean.csv        # anonymized transaction history
├── regression.py                 # linear regression + neural network
├── classification.py             # neural network classification + interactive mode
├── decision_tree.py              # decision tree classification
├── decision_tree_regression.py   # decision tree regression
├── knn.py                        # K-Nearest Neighbors classification
├── random_forest.py              # random forest classification
├── comparison.py                 # runs all models and prints comparison table
└── README.md
```

---

## 📊 Dataset

- **Source:** personal Monobank export (CSV)
- **Period:** full year 2025
- **Transactions:** 611
- **Days with expenses:** 292
- **All personal names and phone numbers anonymized** before publishing

### Features used for models

| Feature | Description |
|---------|-------------|
| day_of_week | 0=Mon, 6=Sun |
| month | 1–12 |
| is_weekend | 1 if Sat/Sun |
| count | average number of transactions per day |
| yesterday | previous day's expenses |
| day_before | expenses two days ago |

---

## 🏆 Results

### Classification (predict expense category)

Classes: **0–200 UAH** (104 days) | **200–600 UAH** (88 days) | **600+ UAH** (100 days)  
Train: 233 days | Test: 59 days | Baseline (random): 33.3%

| Model | Accuracy |
|-------|----------|
| 🥇 Random Forest (n=6, seed=13) | **78.0%** |
| Decision Tree (depth=3) | 76.3% |
| Neural Network (seed=42) | 67.8% |
| KNN (k=3) | 61.0% |
| Baseline (random) | 33.3% |

### Regression (predict daily expense amount)

| Model | RMSE |
|-------|------|
| 🥇 Linear Regression | **1295 UAH** |
| Neural Network | 1541 UAH |
| Decision Tree | 1644 UAH |

---

## 🔬 Architecture

**Linear Regression:** `6 → 1`

**Neural Network (classification):** `6 → 10 → 6 → 3` with ReLU + Softmax

**Neural Network (regression):** `6 → 6 → 3 → 1` with ReLU

**Decision Tree:** max depth = 3, Gini impurity

**Random Forest:** 6 trees, bagging + random feature subsets, majority vote

**KNN:** k=3, Euclidean distance

---

## 🖥 Interactive Mode

`classification.py` includes an interactive prediction mode:

```
Enter data (or 'exit' to quit):
  Day of week (0=Mon, 6=Sun): 5
  Month (1-12): 3
  Yesterday's expenses (UAH): 750
  Day before (UAH): 400

  Day: Saturday, month 3
  ─────────────────────────────
  0-200 UAH      0.0%
  200-600 UAH    5.8%  █
  600+ UAH      94.2%  ██████████████████ ←
  ─────────────────────────────
  Prediction: 600+ UAH
```

---

## 🛠 Tech Stack

- **Python 3.11+** — no ML frameworks
- **NumPy** — matrix operations, neural network implementation
- **Pandas** — data loading and preprocessing

---

## ⚙️ Implemented from Scratch

- Forward pass & Backpropagation
- Gradient Descent (SGD)
- ReLU and Softmax activations
- Cross Entropy and MSE loss functions
- Early stopping
- Min-max normalization
- Train/test split
- Decision Tree (Gini impurity, recursive splitting)
- Random Forest (bagging, random subsets, majority vote)
- KNN (Euclidean distance, k-neighbors voting)

---

## 💡 Key Takeaway

> Predicting personal expenses is a genuinely hard problem — even for complex models — due to the chaotic nature of real spending data (std=1535 UAH at mean=851 UAH). This project shows a full ML pipeline from raw bank data to trained models, including an honest analysis of limitations. **Simple models on small data often beat complex ones.**

---

## 🔮 Next Steps

- [ ] Add 2026 transaction data (more training samples)
- [ ] Reimplement using scikit-learn and compare
- [ ] Try XGBoost and SVM
- [ ] Web interface for interactive prediction
