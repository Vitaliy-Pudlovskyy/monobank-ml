"""
Monobank Витрати — XGBoost (з нуля)
=====================================
Реалізація з нуля (тільки NumPy і Pandas).

Відмінність від Gradient Boosting:
  1. Оптимальне значення листа: w* = -ΣG / (ΣH + λ)
  2. Критерій розбиття через Gain замість MSE
  3. Два параметри регуляризації: λ (lambda) і γ (gamma)
  4. Гессіан hᵢ — для класифікації не константа

Gain = ½ · [GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ
"""

import math
import pandas as pd
import numpy as np

np.random.seed(42)

# ============================================================
# КРОК 1-4 — ДАНІ (стандартний pipeline)
# ============================================================
df      = pd.read_csv('../data_set/monobank_clean (1).csv')
витрати = df[df['Сума в валюті картки (UAH)'] < 0].copy()
витрати['Сума в валюті картки (UAH)'] = pd.to_numeric(витрати['Сума в валюті картки (UAH)'])
витрати['Дата i час операції'] = pd.to_datetime(
    витрати['Дата i час операції'], format='%d.%m.%Y %H:%M:%S'
)
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
середня_кількість          = round(денні_витрати['кількість'].mean())

print(f"Завантажено: {len(денні_витрати)} днів з витратами")

# ============================================================
# КРОК 5 — ПІДГОТОВКА
# ============================================================
X = денні_витрати[['день_тижня', 'місяць', 'вихідний', 'кількість', 'вчора', 'позавчора']].values

bins   = [0, 200, 600, float('inf')]
LABELS = ['0-200 UAH', '200-600 UAH', '600+ UAH']
денні_витрати['клас'] = pd.cut(денні_витрати['сума'], bins=bins, labels=[0, 1, 2])
y_cls = денні_витрати['клас'].astype(int).values
y_reg = денні_витрати['сума'].values

split    = int(len(X) * 0.8)
X_train  = X[:split];       X_test  = X[split:]
y_tr_cls = y_cls[:split];   y_te_cls = y_cls[split:]
y_tr_reg = y_reg[:split];   y_te_reg = y_reg[split:]

X_min = X_train.min(axis=0); X_max = X_train.max(axis=0)
X_tr  = (X_train - X_min) / (X_max - X_min)
X_te  = (X_test  - X_min) / (X_max - X_min)

y_min  = y_tr_reg.min(); y_max = y_tr_reg.max()
y_tr_n = (y_tr_reg - y_min) / (y_max - y_min)
y_te_n = (y_te_reg - y_min) / (y_max - y_min)

print(f"Train: {len(X_tr)} | Test: {len(X_te)}\n")


# ============================================================
# КЛАС 1 — XGBoostTree (одне дерево)
# ============================================================
class XGBoostTree:
    """
    Одне дерево в XGBoost.

    Головна відмінність від DecisionStump:
      - приймає градієнти G і гессіани H замість y
      - значення листа: w* = -ΣG / (ΣH + λ)
      - критерій розбиття: Gain формула (не MSE)
      - gamma γ автоматично обрізає непотрібні розбиття
    """

    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.lambda_   = lambda_   # L2 регуляризація
        self.gamma     = gamma     # мінімальний gain для розбиття
        self.root      = None

    def fit(self, X, G, H):
        """
        X — ознаки
        G — градієнти (перша похідна втрат по F)
        H — гессіани  (друга похідна втрат по F)
        """
        self.root = self._build(X, G, H, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])

    # ---------- внутрішні методи ----------

    def _leaf_value(self, G, H):
        """
        Оптимальне значення листа (аналітична формула XGBoost):
            w* = -ΣG / (ΣH + λ)

        Без λ це просто середнє залишків.
        З λ > 0 — значення "стискається" до нуля (регуляризація).
        """
        return -G.sum() / (H.sum() + self.lambda_)

    def _gain(self, G, H, G_L, H_L, G_R, H_R):
        """
        Gain від розбиття — наскільки корисне це розбиття.

        Gain = ½ · [GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ

        Якщо Gain < 0 — розбиття шкідливе, не робимо його.
        Саме γ контролює цей поріг (pruning).
        """
        def score(g, h):
            # Оцінка одного вузла: g²/(h+λ)
            return (g ** 2) / (h + self.lambda_)

        gain = 0.5 * (score(G_L, H_L) + score(G_R, H_R) - score(G, H)) - self.gamma
        return gain

    def _best_split(self, X, G, H):
        best_gain      = 0      # шукаємо gain > 0 (gain < 0 = не розбиваємо)
        best_feature   = None
        best_threshold = None

        G_total = G.sum()
        H_total = H.sum()

        for feature in range(X.shape[1]):
            # Сортуємо по значенню ознаки — ефективніший перебір порогів
            sorted_idx = np.argsort(X[:, feature])
            X_sorted   = X[sorted_idx, feature]
            G_sorted   = G[sorted_idx]
            H_sorted   = H[sorted_idx]

            # Накопичуємо суми зліва направо (cumsum = cumulative sum)
            # cumsum([1,2,3,4]) = [1, 3, 6, 10]
            G_left_cum = np.cumsum(G_sorted)
            H_left_cum = np.cumsum(H_sorted)

            for i in range(len(X_sorted) - 1):
                # Пропускаємо однакові значення — поріг між ними немає сенсу
                if X_sorted[i] == X_sorted[i + 1]:
                    continue

                G_L = G_left_cum[i];          H_L = H_left_cum[i]
                G_R = G_total - G_left_cum[i]; H_R = H_total - H_left_cum[i]

                gain = self._gain(G_total, H_total, G_L, H_L, G_R, H_R)

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    # Поріг = середнє між поточним і наступним значенням
                    best_threshold = (X_sorted[i] + X_sorted[i + 1]) / 2

        return best_feature, best_threshold, best_gain

    def _build(self, X, G, H, depth):
        # Умови зупинки
        if depth >= self.max_depth or len(G) < 2:
            return {'leaf': True, 'value': self._leaf_value(G, H)}

        feature, threshold, gain = self._best_split(X, G, H)

        # gain <= 0 або γ заблокував розбиття
        if feature is None or gain <= 0:
            return {'leaf': True, 'value': self._leaf_value(G, H)}

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            'leaf':      False,
            'feature':   feature,
            'threshold': threshold,
            'left':  self._build(X[left_mask],  G[left_mask],  H[left_mask],  depth + 1),
            'right': self._build(X[right_mask], G[right_mask], H[right_mask], depth + 1),
        }

    def _predict_one(self, node, x):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        return self._predict_one(node['right'], x)


# ============================================================
# КЛАС 2 — XGBoostRegressor
# ============================================================
class XGBoostRegressor:
    """
    XGBoost для регресії (MSE).

    Градієнт і гессіан для MSE L = ½(y - F)²:
        gᵢ = F(xᵢ) - yᵢ    ← похідна по F
        hᵢ = 1               ← друга похідна (константа для MSE)

    Тому для регресії XGBoost ≈ GB, але з кращим критерієм розбиття.
    Різниця помітна більше в класифікації.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, lambda_=1.0, gamma=0.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.lambda_       = lambda_
        self.gamma         = gamma
        self.trees         = []
        self.F0            = None

    def fit(self, X, y, verbose=False):
        self.F0 = float(np.mean(y))
        F = np.full(len(y), self.F0)

        for m in range(self.n_estimators):
            # Градієнти і гессіани для MSE
            G = F - y        # gᵢ = F(xᵢ) - yᵢ
            H = np.ones(len(y))  # hᵢ = 1 (константа)

            tree = XGBoostTree(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma
            )
            tree.fit(X, G, H)
            self.trees.append(tree)

            F = F + self.learning_rate * tree.predict(X)

            if verbose and (m + 1) % 20 == 0:
                mse = float(np.mean((y - F) ** 2))
                print(f"  дерево {m+1:3d}/{self.n_estimators} | train MSE: {mse:.4f}")

        return self

    def predict(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return F


# ============================================================
# КЛАС 3 — XGBoostClassifier
# ============================================================
class XGBoostClassifier:
    """
    XGBoost для класифікації (3 класи, one-vs-rest).

    Градієнт і гессіан для cross-entropy:
        gᵢ = pᵢ - yᵢ           ← ймовірність мінус реальний клас
        hᵢ = pᵢ · (1 - pᵢ)     ← НЕ константа! залежить від впевненості

    Саме hᵢ робить XGBoost розумнішим:
      - якщо модель впевнена (p ≈ 1 або p ≈ 0) → hᵢ ≈ 0 → маленький крок
      - якщо модель невпевнена (p ≈ 0.5)        → hᵢ ≈ 0.25 → більший крок
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, lambda_=1.0, gamma=0.0, n_classes=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.lambda_       = lambda_
        self.gamma         = gamma
        self.n_classes     = n_classes
        self.trees         = [[] for _ in range(n_classes)]
        self.F0            = None

    @staticmethod
    def _softmax(F):
        e = np.exp(F - F.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y, verbose=False):
        n = len(y)
        self.F0 = math.log(1 / self.n_classes)
        F = np.full((n, self.n_classes), self.F0)

        for m in range(self.n_estimators):
            probs = self._softmax(F)   # (n, 3)

            for k in range(self.n_classes):
                y_k = (y == k).astype(float)

                # Градієнт: pᵢ - yᵢ
                G = probs[:, k] - y_k

                # Гессіан: pᵢ · (1 - pᵢ)  ← ось де XGBoost розумніший за GB
                H = probs[:, k] * (1 - probs[:, k])
                # Мінімальний поріг щоб не ділити на ~0
                H = np.maximum(H, 1e-6)

                tree = XGBoostTree(
                    max_depth=self.max_depth,
                    lambda_=self.lambda_,
                    gamma=self.gamma
                )
                tree.fit(X, G, H)
                self.trees[k].append(tree)

                F[:, k] += self.learning_rate * tree.predict(X)

            if verbose and (m + 1) % 20 == 0:
                preds = np.argmax(probs, axis=1)
                acc   = float(np.mean(preds == y))
                print(f"  дерево {m+1:3d}/{self.n_estimators} | train acc: {acc*100:.1f}%")

        return self

    def predict_proba(self, X):
        n = len(X)
        F = np.full((n, self.n_classes), self.F0)
        for k in range(self.n_classes):
            for tree in self.trees[k]:
                F[:, k] += self.learning_rate * tree.predict(X)
        return self._softmax(F)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# НАВЧАННЯ — РЕГРЕСІЯ
# ============================================================
print("=" * 50)
print("XGBOOST — РЕГРЕСІЯ")
print("=" * 50)

xgb_reg = XGBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    lambda_=1.0,   # L2 регуляризація
    gamma=0.1      # мінімальний gain
)
print("Навчання...")
xgb_reg.fit(X_tr, y_tr_n, verbose=True)

pred_te_n    = xgb_reg.predict(X_te)
test_mse     = float(np.mean((pred_te_n - y_te_n) ** 2))
test_rmse_uah = math.sqrt(test_mse * (y_max - y_min) ** 2)

print(f"\nTest RMSE: {test_rmse_uah:.0f} UAH")
print(f"(Gradient Boosting: 1613 UAH)")
print(f"(Лінійна регресія:  1295 UAH)")
print(f"(Нейронна мережа:   1541 UAH)")

# ============================================================
# НАВЧАННЯ — КЛАСИФІКАЦІЯ
# ============================================================
print("\n" + "=" * 50)
print("XGBOOST — КЛАСИФІКАЦІЯ")
print("=" * 50)

xgb_cls = XGBoostClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=3,
    lambda_=2.0,
    gamma=0.5
)
print("Навчання...")
xgb_cls.fit(X_tr, y_tr_cls, verbose=True)

test_pred = xgb_cls.predict(X_te)
test_acc  = float(np.mean(test_pred == y_te_cls))

print(f"\nTest точність: {test_acc*100:.1f}%")
print(f"(Random Forest:     78.0%)")
print(f"(Gradient Boosting: 71.2%)")
print(f"(Нейронна мережа:   67.8%)")
print(f"(Baseline:          33.3%)")

# ============================================================
# ПОРІВНЯННЯ λ і γ — покажемо вплив регуляризації
# ============================================================
print(f"\n{'='*50}")
print("ВПЛИВ РЕГУЛЯРИЗАЦІЇ НА ТОЧНІСТЬ:")
print(f"{'─'*50}")
print(f"  {'λ':>4}  {'γ':>4}  {'Точність':>10}")
print(f"  {'─'*30}")

for lam, gam in [(0.0, 0.0), (0.1, 0.0), (1.0, 0.0), (1.0, 0.1), (2.0, 0.5)]:
    model = XGBoostClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=3, lambda_=lam, gamma=gam
    )
    model.fit(X_tr, y_tr_cls)
    acc = float(np.mean(model.predict(X_te) == y_te_cls))
    print(f"  λ={lam:>3}  γ={gam:>3}  →  {acc*100:5.1f}%")

# ============================================================
# ДЕТАЛЬНІ РЕЗУЛЬТАТИ
# ============================================================
print(f"\n{'='*50}")
print("ДЕТАЛЬНІ ПЕРЕДБАЧЕННЯ (тест):")
print(f"{'='*50}")

proba = xgb_cls.predict_proba(X_te)
for i in range(min(5, len(X_te))):
    true_label = LABELS[y_te_cls[i]]
    pred_label = LABELS[test_pred[i]]
    mark       = '✓' if test_pred[i] == y_te_cls[i] else '✗'
    print(f"  {mark} Реально: {true_label:12s} | "
          f"Передбачено: {pred_label:12s} | "
          f"Ймов: {proba[i][0]*100:.0f}% / {proba[i][1]*100:.0f}% / {proba[i][2]*100:.0f}%")

# ============================================================
# ІНТЕРАКТИВНИЙ РЕЖИМ
# ============================================================
day_names = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пятниця', 'Субота', 'Неділя']

print(f"\n{'='*50}")
print("  ПЕРЕДБАЧЕННЯ ВИТРАТ — XGBOOST")
print(f"{'='*50}")

while True:
    print("\nВведіть дані (або 'exit' для виходу):")
    try:
        inp = input("  День тижня (0=Пн, 6=Нд): ")
        if inp.lower() == 'exit':
            break
        день_тижня = int(inp)
        місяць     = int(input("  Місяць (1-12): "))
        вихідний   = 1 if день_тижня >= 5 else 0
        вчора      = float(input("  Витрати вчора (UAH): "))
        позавчора  = float(input("  Витрати позавчора (UAH): "))

        x_input = np.array(
            [день_тижня, місяць, вихідний, середня_кількість, вчора, позавчора],
            dtype=float
        )
        x_norm = (x_input - X_min) / (X_max - X_min)

        probs    = xgb_cls.predict_proba(x_norm.reshape(1, -1))[0]
        pred_cls = int(np.argmax(probs))
        pred_sum_n = xgb_reg.predict(x_norm.reshape(1, -1))[0]
        pred_sum   = float(pred_sum_n) * (y_max - y_min) + y_min

        print(f"\n  День: {day_names[день_тижня]}, місяць {місяць}")
        print(f"  ─────────────────────────────")
        for i, label in enumerate(LABELS):
            bar  = '█' * int(probs[i] * 20)
            mark = ' ←' if i == pred_cls else ''
            print(f"  {label:12s} {probs[i]*100:5.1f}%  {bar}{mark}")
        print(f"  ─────────────────────────────")
        print(f"  Категорія:    {LABELS[pred_cls]}")
        print(f"  Сума (регр.): {pred_sum:.0f} UAH")

    except (ValueError, KeyboardInterrupt):
        print("  Помилка вводу")