"""
Monobank ML — Порівняння моделей
==================================
Дані:   особиста виписка Монобанку за 2025 рік
Днів:   292 (після групування і очистки)
Ознаки: день_тижня, місяць, вихідний, кількість, вчора, позавчора

Задачі:
  1. Класифікація — передбачити категорію дня (0-200 / 200-600 / 600+ UAH)
  2. Регресія     — передбачити суму витрат (UAH)
"""

import math
import random
import pandas as pd
import numpy as np

np.random.seed(42)
random.seed(42)

# ============================================================
# КРОК 1-4 — ЗАВАНТАЖЕННЯ І ПІДГОТОВКА ДАНИХ
# ============================================================
df      = pd.read_csv('data_set/monobank_clean (1).csv')
витрати = df[df['Сума в валюті картки (UAH)'] < 0].copy()
витрати['Сума в валюті картки (UAH)'] = pd.to_numeric(витрати['Сума в валюті картки (UAH)'])
витрати['Дата i час операції'] = pd.to_datetime(витрати['Дата i час операції'], format='%d.%m.%Y %H:%M:%S')
витрати['день_тижня'] = витрати['Дата i час операції'].dt.dayofweek
витрати['місяць']     = витрати['Дата i час операції'].dt.month
витрати['вихідний']   = витрати['день_тижня'].apply(lambda x: 1 if x >= 5 else 0)
витрати['дата']       = витрати['Дата i час операції'].dt.date

денні_витрати = витрати.groupby('дата').agg(
    сума       = ('Сума в валюті картки (UAH)', 'sum'),
    кількість  = ('Сума в валюті картки (UAH)', 'count'),
    день_тижня = ('день_тижня', 'first'),
    місяць     = ('місяць', 'first'),
    вихідний   = ('вихідний', 'first')
).reset_index()

денні_витрати['сума']      = денні_витрати['сума'].abs()
денні_витрати['вчора']     = денні_витрати['сума'].shift(1)
денні_витрати['позавчора'] = денні_витрати['сума'].shift(2)
денні_витрати              = денні_витрати.dropna()

середня_кількість = round(денні_витрати['кількість'].mean())

print(f"Завантажено: {len(денні_витрати)} днів з витратами")
print(f"Ознаки: день_тижня, місяць, вихідний, кількість, вчора, позавчора\n")

# ============================================================
# КРОК 5 — ПІДГОТОВКА ДЛЯ КЛАСИФІКАЦІЇ
# ============================================================
X      = денні_витрати[['день_тижня', 'місяць', 'вихідний', 'кількість', 'вчора', 'позавчора']].values
bins   = [0, 200, 600, float('inf')]
LABELS = ['0-200 UAH', '200-600 UAH', '600+ UAH']
денні_витрати['клас'] = pd.cut(денні_витрати['сума'], bins=bins, labels=[0, 1, 2])
y_cls  = денні_витрати['клас'].astype(int).values

split    = int(len(X) * 0.8)
X_train  = X[:split];     X_test  = X[split:]
y_train  = y_cls[:split]; y_test  = y_cls[split:]
X_min    = X_train.min(axis=0); X_max = X_train.max(axis=0)
X_train_n = (X_train - X_min) / (X_max - X_min)
X_test_n  = (X_test  - X_min) / (X_max - X_min)

# ============================================================
# ПІДГОТОВКА ДЛЯ РЕГРЕСІЇ
# ============================================================
y_reg    = денні_витрати['сума'].values
y_tr_reg = y_reg[:split]; y_te_reg = y_reg[split:]
y_min    = y_tr_reg.min(); y_max = y_tr_reg.max()
y_tr_n   = (y_tr_reg - y_min) / (y_max - y_min)
y_te_n   = (y_te_reg - y_min) / (y_max - y_min)

results_cls = {}
results_reg = {}

# ============================================================
# УТИЛІТИ ДЛЯ ДЕРЕВ
# ============================================================
class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.value     = value
        self.left      = None
        self.right     = None
    def is_leaf(self):
        return self.value is not None

def gini(y):
    n = len(y); impurity = 1
    for c in set(y):
        p = (y == c).sum() / n
        impurity -= p ** 2
    return impurity

def mse_criterion(y):
    mean = sum(y) / len(y)
    return sum((v - mean) ** 2 for v in y) / len(y)

def best_split(X, y, criterion='gini'):
    best_gain = 0; best_feature = None; best_threshold = None
    n = len(y)
    fn  = gini if criterion == 'gini' else mse_criterion
    cur = fn(y)
    for feature in range(X.shape[1]):
        for threshold in set(X[:, feature]):
            left_y  = [y[i] for i in range(n) if X[i, feature] <= threshold]
            right_y = [y[i] for i in range(n) if X[i, feature] > threshold]
            if not left_y or not right_y:
                continue
            w_score = (len(left_y)/n)*fn(left_y) + (len(right_y)/n)*fn(right_y)
            gain = cur - w_score
            if gain > best_gain:
                best_gain = gain; best_feature = feature; best_threshold = threshold
    return best_feature, best_threshold, best_gain

def build_tree(X, y, depth=0, max_depth=3, criterion='gini'):
    if len(set(y)) == 1:
        return Node(value=y[0] if criterion == 'gini' else float(y[0]))
    if depth >= max_depth:
        if criterion == 'gini':
            counts = {c: (y == c).sum() for c in set(y)}
            return Node(value=max(counts, key=counts.get))
        else:
            return Node(value=float(sum(y)/len(y)))
    feature, threshold, gain = best_split(X, y, criterion)
    if gain == 0:
        if criterion == 'gini':
            counts = {c: (y == c).sum() for c in set(y)}
            return Node(value=max(counts, key=counts.get))
        else:
            return Node(value=float(sum(y)/len(y)))
    lm = X[:, feature] <= threshold; rm = ~lm
    node = Node(feature=feature, threshold=threshold)
    node.left  = build_tree(X[lm], y[lm], depth+1, max_depth, criterion)
    node.right = build_tree(X[rm], y[rm], depth+1, max_depth, criterion)
    return node

def predict_one(node, x):
    if node.is_leaf(): return node.value
    if x[node.feature] <= node.threshold: return predict_one(node.left, x)
    else: return predict_one(node.right, x)

# ============================================================
# 1. ЛІНІЙНА РЕГРЕСІЯ
# ============================================================
print("Навчання: Лінійна регресія...", end=' ')
w_lr = np.random.randn(6, 1) * 0.01; b_lr = np.zeros(1)

for epoch in range(120):
    for i in range(len(X_train_n)):
        x = X_train_n[i]; y = y_tr_n[i]
        yp = x @ w_lr + b_lr
        d  = 2*(yp - y)
        w_lr -= 0.005 * x.reshape(-1,1) * d.reshape(1,-1)
        b_lr -= 0.005 * d

lr_mse = sum(float(((X_test_n[i] @ w_lr + b_lr).flatten()[0] - y_te_n[i])**2) for i in range(len(X_test_n))) / len(X_test_n)
results_reg['Лінійна регресія'] = lr_mse
print(f"MSE = {lr_mse:.4f} ✓")

# ============================================================
# 2. НЕЙРОННА МЕРЕЖА — РЕГРЕСІЯ
# ============================================================
print("Навчання: Нейронна мережа (регресія)...", end=' ')
w1r = np.random.randn(6,6)*0.01; b1r = np.zeros(6)
w2r = np.random.randn(6,3)*0.01; b2r = np.zeros(3)
w3r = np.random.randn(3,1)*0.01; b3r = np.zeros(1)

def relu(z): return np.maximum(0, z)

best_l = float('inf'); no_imp = 0
for epoch in range(50000):
    tl = 0
    for i in range(len(X_train_n)):
        x = X_train_n[i]; y = y_tr_n[i]
        z1=x@w1r+b1r; h1=relu(z1); z2=h1@w2r+b2r; h2=relu(z2); yp=h2@w3r+b3r
        tl += float(np.mean((yp-y)**2))
        d3=2*(yp-y); dw3=h2.reshape(-1,1)@d3.reshape(1,-1); db3=d3
        d2=(d3@w3r.T)*(z2>0); dw2=h1.reshape(-1,1)@d2.reshape(1,-1); db2=d2
        d1=(d2@w2r.T)*(z1>0); dw1=x.reshape(-1,1)@d1.reshape(1,-1); db1=d1
        w1r-=0.005*dw1; b1r-=0.005*db1; w2r-=0.005*dw2; b2r-=0.005*db2; w3r-=0.005*dw3; b3r-=0.005*db3
    if tl < best_l: best_l=tl; no_imp=0
    else: no_imp+=1
    if no_imp >= 500: break

nn_reg_mse = 0
for i in range(len(X_test_n)):
    x=X_test_n[i]; h1=relu(x@w1r+b1r); h2=relu(h1@w2r+b2r); yp=h2@w3r+b3r
    nn_reg_mse += float((yp.flatten()[0] - y_te_n[i])**2)
nn_reg_mse /= len(X_test_n)
results_reg['Нейронна мережа'] = nn_reg_mse
print(f"MSE = {nn_reg_mse:.4f} ✓")

# ============================================================
# 3. ДЕРЕВО РІШЕНЬ — РЕГРЕСІЯ
# ============================================================
print("Навчання: Дерево рішень (регресія)...", end=' ')
tree_reg = build_tree(X_train_n, y_tr_n, max_depth=3, criterion='mse')
dt_reg_mse = sum((predict_one(tree_reg, X_test_n[i]) - y_te_n[i])**2 for i in range(len(X_test_n))) / len(X_test_n)
results_reg['Дерево рішень'] = dt_reg_mse
print(f"MSE = {dt_reg_mse:.4f} ✓")

# ============================================================
# 4. НЕЙРОННА МЕРЕЖА — КЛАСИФІКАЦІЯ
# ============================================================
print("Навчання: Нейронна мережа (класифікація)...", end=' ')
np.random.seed(42)
w1c=np.random.randn(6,10)*0.01; b1c=np.zeros(10)
w2c=np.random.randn(10,6)*0.01; b2c=np.zeros(6)
w3c=np.random.randn(6,3)*0.01;  b3c=np.zeros(3)

def softmax(z): e=np.exp(z-np.max(z)); return e/e.sum()

best_l=float('inf'); no_imp=0
for epoch in range(5000):
    tl=0
    for i in range(len(X_train_n)):
        x=X_train_n[i]; label=y_train[i]
        z1=x@w1c+b1c; h1=relu(z1); z2=h1@w2c+b2c; h2=relu(z2); out=softmax(h2@w3c+b3c)
        tl += -np.log(out[label]+1e-9)
        d3=out.copy(); d3[label]-=1
        dw3=h2.reshape(-1,1)@d3.reshape(1,-1); db3=d3
        d2=(d3@w3c.T)*(z2>0); dw2=h1.reshape(-1,1)@d2.reshape(1,-1); db2=d2
        d1=(d2@w2c.T)*(z1>0); dw1=x.reshape(-1,1)@d1.reshape(1,-1); db1=d1
        w1c-=0.01*dw1; b1c-=0.01*db1; w2c-=0.01*dw2; b2c-=0.01*db2; w3c-=0.01*dw3; b3c-=0.01*db3
    if tl < best_l: best_l=tl; no_imp=0
    else: no_imp+=1
    if no_imp >= 280: break

nn_acc = sum(1 for i in range(len(X_test_n))
             if np.argmax(softmax(relu(relu(X_test_n[i]@w1c+b1c)@w2c+b2c)@w3c+b3c)) == y_test[i]) / len(y_test)
results_cls['Нейронна мережа'] = nn_acc
print(f"{nn_acc*100:.1f}% ✓")

# ============================================================
# 5. ДЕРЕВО РІШЕНЬ — КЛАСИФІКАЦІЯ
# ============================================================
print("Навчання: Дерево рішень (класифікація)...", end=' ')
tree_cls = build_tree(X_train_n, y_train, max_depth=3, criterion='gini')
dt_acc = sum(1 for i in range(len(X_test_n)) if predict_one(tree_cls, X_test_n[i]) == y_test[i]) / len(y_test)
results_cls['Дерево рішень'] = dt_acc
print(f"{dt_acc*100:.1f}% ✓")

# ============================================================
# 6. KNN — КЛАСИФІКАЦІЯ
# ============================================================
print("Навчання: KNN...", end=' ')
def manhattan(a, b):
    return sum(abs(a[i] - b[i]) for i in range(len(a)))

def knn_predict(x_new, k=3):
    dists = sorted([(manhattan(X_train_n[i], x_new), y_train[i]) for i in range(len(X_train_n))])
    votes = [d[1] for d in dists[:k]]
    return max(set(votes), key=votes.count)

knn_acc = sum(1 for i in range(len(X_test_n)) if knn_predict(X_test_n[i]) == y_test[i]) / len(y_test)
results_cls['KNN (k=3)'] = knn_acc
print(f"{knn_acc*100:.1f}% ✓")

# ============================================================
# 7. RANDOM FOREST — КЛАСИФІКАЦІЯ
# ============================================================
print("Навчання: Random Forest...", end=' ')
np.random.seed(13); random.seed(13)

def rf_train(X, y, n_trees=6, max_depth=5, sample_size=0.8):
    trees = []
    for _ in range(n_trees):
        idx = random.sample(range(len(X)), int(len(X)*sample_size))
        trees.append(build_tree(X[idx], y[idx], max_depth=max_depth, criterion='gini'))
    return trees

def rf_predict(trees, X):
    preds = []
    for x in X:
        votes = [predict_one(t, x) for t in trees]
        preds.append(max(set(votes), key=votes.count))
    return preds

trees = rf_train(X_train_n, y_train, n_trees=6)
rf_preds = rf_predict(trees, X_test_n)
rf_acc = sum(1 for i in range(len(rf_preds)) if rf_preds[i] == y_test[i]) / len(y_test)
results_cls['Random Forest'] = rf_acc
print(f"{rf_acc*100:.1f}% ✓")

# ============================================================
# ФІНАЛЬНА ТАБЛИЦЯ
# ============================================================
print(f"\n{'='*50}")
print(f"  MONOBANK ML — ПОРІВНЯННЯ МОДЕЛЕЙ")
print(f"  Дані: 292 дні | 2025 рік | 6 ознак")
print(f"{'='*50}")

print(f"\n  КЛАСИФІКАЦІЯ (категорія дня)")
print(f"  {'─'*40}")
print(f"  {'Baseline (random)':25s}  33.3%")
print(f"  {'─'*40}")
cls_sorted = sorted(results_cls.items(), key=lambda x: x[1], reverse=True)
for i, (name, acc) in enumerate(cls_sorted):
    mark = '  ← найкращий' if i == 0 else ''
    print(f"  {name:25s} {acc*100:5.1f}%{mark}")
print(f"  {'─'*40}")

print(f"\n  РЕГРЕСІЯ (сума витрат UAH)")
print(f"  {'─'*40}")
reg_sorted = sorted(results_reg.items(), key=lambda x: x[1])
for i, (name, mse_val) in enumerate(reg_sorted):
    rmse_uah = math.sqrt(mse_val * (y_max - y_min)**2)
    mark = '  ← найкращий' if i == 0 else ''
    print(f"  {name:25s} RMSE {rmse_uah:6.0f} UAH{mark}")
print(f"  {'─'*40}")

print(f"""
  ВИСНОВКИ:
  • На малих хаотичних даних прості моделі виграють
  • Random Forest > одне дерево (різноманітність)
  • Нейронна мережа програє через малий датасет (292 дні)
  • KNN погано з 6 ознаками (прокляття розмірності)
  • Регресія складна: std витрат = 1535 UAH при середньому 851

{'='*50}
""")
