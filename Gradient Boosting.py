"""
Monobank Витрати — Gradient Boosting
======================================
Реалізація з нуля (тільки NumPy і Pandas).

Ідея: будуємо дерева послідовно, кожне наступне виправляє
помилки попереднього. На відміну від Random Forest (паралельний),
бустинг — послідовний процес.

Формула оновлення:
    F_m(x) = F_{m-1}(x) + η · h_m(x)
де:
    h_m — нове дерево, навчене на залишках
    η   — learning rate (крок навчання)
"""


import math
import pandas as pd
import numpy as np

np.random.seed(42)

# ============================================================
# КРОК 1-4 — ЗАВАНТАЖЕННЯ І ПІДГОТОВКА ДАНИХ (стандартний pipeline)
# ============================================================
df = pd.read_csv('data_set/monobank_clean (1).csv')
витрати = df[df['Сума в валюті картки (UAH)'] < 0].copy()
витрати['Сума в валюті картки (UAH)'] = pd.to_numeric(витрати['Сума в валюті картки (UAH)'])
витрати['Дата i час операції'] = pd.to_datetime(
    витрати['Дата i час операції'], format='%d.%m.%Y %H:%M:%S'
)
витрати['день_тижня'] = витрати['Дата i час операції'].dt.dayofweek
витрати['місяць'] = витрати['Дата i час операції'].dt.month
витрати['вихідний'] = витрати['день_тижня'].apply(lambda x: 1 if x >= 5 else 0)
витрати['дата'] = витрати['Дата i час операції'].dt.date

денні_витрати = витрати.groupby('дата').agg(
    сума=('Сума в валюті картки (UAH)', 'sum'),
    кількість=('Сума в валюті картки (UAH)', 'count'),
    день_тижня=('день_тижня', 'first'),
    місяць=('місяць', 'first'),
    вихідний=('вихідний', 'first')
).reset_index()

денні_витрати['сума'] = денні_витрати['сума'].abs()
денні_витрати['вчора'] = денні_витрати['сума'].shift(1)
денні_витрати['позавчора'] = денні_витрати['сума'].shift(2)
денні_витрати = денні_витрати.dropna()
середня_кількість = round(денні_витрати['кількість'].mean())

print(f"Завантажено: {len(денні_витрати)} днів з витратами")

# ============================================================
# КРОК 5 — ПІДГОТОВКА FEATURES
# ============================================================
X = денні_витрати[['день_тижня', 'місяць', 'вихідний', 'кількість', 'вчора', 'позавчора']].values

# --- Для класифікації ---
bins = [0, 200, 600, float('inf')]
LABELS = ['0-200 UAH', '200-600 UAH', '600+ UAH']
денні_витрати['клас'] = pd.cut(денні_витрати['сума'], bins=bins, labels=[0, 1, 2])
y_cls = денні_витрати['клас'].astype(int).values

# --- Для регресії ---
y_reg = денні_витрати['сума'].values

# --- Розбивка train/test ---
split = int(len(X) * 0.8)
X_train = X[:split];
X_test = X[split:]
y_tr_cls = y_cls[:split];
y_te_cls = y_cls[split:]
y_tr_reg = y_reg[:split];
y_te_reg = y_reg[split:]

# --- Нормалізація ---
X_min = X_train.min(axis=0);
X_max = X_train.max(axis=0)
X_tr = (X_train - X_min) / (X_max - X_min)
X_te = (X_test - X_min) / (X_max - X_min)

# Нормалізація y тільки для регресії
y_min = y_tr_reg.min();
y_max = y_tr_reg.max()
y_tr_n = (y_tr_reg - y_min) / (y_max - y_min)
y_te_n = (y_te_reg - y_min) / (y_max - y_min)

print(f"Train: {len(X_tr)} | Test: {len(X_te)}")
print(f"Класи: {[(y_te_cls == i).sum() for i in range(3)]}\n")

# ============================================================
# КЛАС 1 — DecisionStump (слабкий учень)
# ============================================================
class DecisionStump:
    """
    Просте дерево рішень з обмеженою глибиною.
    """

    def __init__(self, max_depth = 3):
        self.max_depth = max_depth


    def fit(self, X, y):
        self.root =  self._build(X, y , depth = 0)

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in  X])

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2:
            return {'leaf':True, 'value': float(np.mean(y))}

        feature , threshold, gain = self._best_split(X, y)

        if gain == 0:
            return {'leaf': True, 'value': float(np.mean(y))}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {'leaf':      False,
            'feature':   feature,
            'threshold': threshold,
            'left':  self._build(X[left_mask],  y[left_mask],  depth + 1),
            'right': self._build(X[right_mask], y[right_mask], depth + 1),
        }
    def _best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        n = len(y)
        current_mse = np.mean((y-np.mean(y)) ** 2)

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:,feature]):
                left_y = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] > threshold]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                weighted_mse = (
                        (len(left_y) / n) * np.mean((left_y - np.mean(left_y)) ** 2) +
                        (len(right_y) / n) * np.mean((right_y - np.mean(right_y)) ** 2)
                )
                gain = current_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold


        return best_feature, best_threshold, best_gain

    def _predict_one(self, node, x):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        return self._predict_one(node['right'], x)


class GradientBoostingRegressor:
    """
    Gradient Boosting для регресії (передбачення суми витрат).

    Алгоритм:
    1. F₀ = mean(y)
    2. Для кожного дерева m:
       a. r = y - F_{m-1}(X)      ← залишки
       b. Навчаємо дерево на r
       c. F_m = F_{m-1} + lr * дерево.predict(X)
    """

    def __init__(self, n_estimators = 100, learning_rate = 0.1, max_depth = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []  # список навчених дерев
        self.F0 = None # початкове передбачення

    def fit(self, X, y, verbose=False):
        # Крок 0: початкове передбачення = середнє
        self.F0 = np.mean(y)
        F = np.full(len(y), self.F0)  # [F0, F0, F0, ...] для кожного прикладу

        for m in range(self.n_estimators):
            # Крок a: рахуємо залишки
            residuals = y - F

            # Крок b: навчаємо дерево НА ЗАЛИШКАХ
            tree = DecisionStump(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Крок c: оновлюємо передбачення
            update = tree.predict(X)
            F = F + self.learning_rate * update

            self.trees.append(tree)

            if verbose and (m + 1) % 20 == 0:
                mse = np.mean((y - F) ** 2)
                print(f"  дерево {m + 1:3d}/{self.n_estimators} | MSE: {mse:.4f}")

        return self

    def predict(self, X):
        # Починаємо з F0
        F = np.full(len(X), self.F0)

        # Додаємо внесок кожного дерева
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return F


class GradientBoostingClassifier:
    """
    Gradient Boosting для класифікації (3 класи: 0-200 / 200-600 / 600+).

    Будуємо окремий ансамбль дерев для КОЖНОГО класу.
    Тобто маємо: trees[0], trees[1], trees[2] — по n_estimators дерев кожен.

    Залишки для класу k:
        r_k = (y == k) - p_k
    де p_k — поточна ймовірність класу k (через softmax).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, n_classes=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_classes = n_classes
        # trees[k] = список дерев для класу k
        self.trees = [[] for _ in range(n_classes)]
        self.F0 = None

    @staticmethod
    def _softmax(F):
        # F має форму (n_samples, n_classes)
        e = np.exp(F - F.max(axis=1, keepdims=True))  # стабільний softmax
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y, verbose=False):
        n = len(y)

        # Крок 0: рівномірні логіти (log(1/3) для 3 класів)
        self.F0 = np.log(1 / self.n_classes)
        F = np.full((n, self.n_classes), self.F0)

        for m in range(self.n_estimators):
            probs = self._softmax(F)  # (n, 3) — ймовірності

            for k in range(self.n_classes):
                # Залишок для класу k: "чи правда що це клас k" мінус "наша впевненість"
                # (y == k) дає [0,1,0,1,...] — індикатор класу
                residuals = (y == k).astype(float) - probs[:, k]

                tree = DecisionStump(max_depth=self.max_depth)
                tree.fit(X, residuals)

                F[:, k] += self.learning_rate * tree.predict(X)
                self.trees[k].append(tree)

            if verbose and (m + 1) % 20 == 0:
                preds = np.argmax(probs, axis=1)
                acc = np.mean(preds == y)
                print(f"  дерево {m + 1:3d}/{self.n_estimators} | train acc: {acc * 100:.1f}%")

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
print("GRADIENT BOOSTING — РЕГРЕСІЯ")
print("=" * 50)

gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)
print("Навчання...")
gb_reg.fit(X_tr, y_tr_n, verbose=True)

# Передбачення і оцінка
pred_te_n = gb_reg.predict(X_te)
test_mse = float(np.mean((pred_te_n - y_te_n) ** 2))
# Переводимо назад в UAH
test_rmse_uah = math.sqrt(test_mse * (y_max - y_min) ** 2)

print(f"\nTest RMSE: {test_rmse_uah:.0f} UAH")
print(f"(Лінійна регресія була: 1295 UAH)")
print(f"(Нейронна мережа була:  1541 UAH)")
print(f"(Дерево рішень було:    1644 UAH)")

# ============================================================
# НАВЧАННЯ — КЛАСИФІКАЦІЯ
# ============================================================
print("\n" + "=" * 50)
print("GRADIENT BOOSTING — КЛАСИФІКАЦІЯ")
print("=" * 50)

gb_cls = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.01,
    max_depth=4
)
print("Навчання...")
gb_cls.fit(X_tr, y_tr_cls, verbose=True)

test_pred = gb_cls.predict(X_te)
test_acc = float(np.mean(test_pred == y_te_cls))

print(f"\nTest точність: {test_acc * 100:.1f}%")
print(f"(Random Forest був:    78.0%)")
print(f"(Дерево рішень було:   76.3%)")
print(f"(Нейронна мережа була: 67.8%)")
print(f"(KNN було:             61.0%)")
print(f"(Baseline:             33.3%)")

# ============================================================
# ДЕТАЛЬНІ РЕЗУЛЬТАТИ
# ============================================================
print(f"\n{'=' * 50}")
print("ДЕТАЛЬНІ ПЕРЕДБАЧЕННЯ (тест):")
print(f"{'=' * 50}")

proba = gb_cls.predict_proba(X_te)
for i in range(min(5, len(X_te))):
    true_label = LABELS[y_te_cls[i]]
    pred_label = LABELS[test_pred[i]]
    mark = '✓' if test_pred[i] == y_te_cls[i] else '✗'
    print(f"  {mark} Реально: {true_label:12s} | "
          f"Передбачено: {pred_label:12s} | "
          f"Ймовірності: {proba[i][0] * 100:.0f}% / {proba[i][1] * 100:.0f}% / {proba[i][2] * 100:.0f}%")

# ============================================================
# ІНТЕРАКТИВНИЙ РЕЖИМ
# ============================================================
day_names = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пятниця', 'Субота', 'Неділя']

print(f"\n{'=' * 50}")
print("  ПЕРЕДБАЧЕННЯ ВИТРАТ — GRADIENT BOOSTING")
print(f"{'=' * 50}")

while True:
    print("\nВведіть дані (або 'exit' для виходу):")
    try:
        inp = input("  День тижня (0=Пн, 6=Нд): ")
        if inp.lower() == 'exit':
            break
        день_тижня = int(inp)
        місяць = int(input("  Місяць (1-12): "))
        вихідний = 1 if день_тижня >= 5 else 0
        вчора = float(input("  Витрати вчора (UAH): "))
        позавчора = float(input("  Витрати позавчора (UAH): "))

        x_input = np.array(
            [день_тижня, місяць, вихідний, середня_кількість, вчора, позавчора],
            dtype=float
        )
        x_norm = (x_input - X_min) / (X_max - X_min)

        # Класифікація
        probs = gb_cls.predict_proba(x_norm.reshape(1, -1))[0]
        pred_cls = int(np.argmax(probs))

        # Регресія
        pred_sum_n = gb_reg.predict(x_norm.reshape(1, -1))[0]
        pred_sum = float(pred_sum_n) * (y_max - y_min) + y_min

        print(f"\n  День: {day_names[день_тижня]}, місяць {місяць}")
        print(f"  ─────────────────────────────")
        for i, label in enumerate(LABELS):
            bar = '█' * int(probs[i] * 20)
            mark = ' ←' if i == pred_cls else ''
            print(f"  {label:12s} {probs[i] * 100:5.1f}%  {bar}{mark}")
        print(f"  ─────────────────────────────")
        print(f"  Категорія:    {LABELS[pred_cls]}")
        print(f"  Сума (регр.): {pred_sum:.0f} UAH")

    except (ValueError, KeyboardInterrupt):
        print("  Помилка вводу")