"""
Monobank Витрати — CatBoost (з нуля)
======================================
Реалізація з нуля (тільки NumPy і Pandas).

Ключові відмінності від інших бустингів:
  1. Ordered Target Encoding — категоріальні ознаки без data leakage
  2. Symmetric Trees — одне розбиття на весь рівень дерева
  3. Cumsum оптимізація — знаходження порогу за O(n) замість O(n²)
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

y_min  = y_tr_reg.min(); y_max = y_tr_reg.max()
y_tr_n = (y_tr_reg - y_min) / (y_max - y_min)
y_te_n = (y_te_reg - y_min) / (y_max - y_min)

CAT_FEATURES = [0, 1]  # день_тижня і місяць

print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")


# ============================================================
# ORDERED TARGET ENCODING
# ============================================================
def ordered_target_encoding(X_train, y_train, X_test, cat_features, prior=0.5):
    """
    Кодує категоріальні ознаки без data leakage.

    Для прикладу i використовуємо статистику тільки по j < i.
    Формула: (сума_y_до_i + prior * global_mean) / (кількість_до_i + prior)

    prior захищає від нових/рідких категорій — при малій кількості
    прикладів encoding тягнеться до глобального середнього.
    """
    X_tr_enc = X_train.astype(float).copy()
    X_te_enc = X_test.astype(float).copy()

    global_mean = float(np.mean(y_train))

    for feat in cat_features:
        # --- Train: ordered (тільки попередні приклади) ---
        running_sum   = {}
        running_count = {}

        for i in range(len(X_train)):
            cat_val = int(X_train[i, feat])
            s = running_sum.get(cat_val, 0.0)
            c = running_count.get(cat_val, 0)

            X_tr_enc[i, feat] = (s + prior * global_mean) / (c + prior)

            running_sum[cat_val]   = s + float(y_train[i])
            running_count[cat_val] = c + 1

        # --- Test: вся статистика train ---
        final_sum   = {}
        final_count = {}
        for i in range(len(X_train)):
            cat_val = int(X_train[i, feat])
            final_sum[cat_val]   = final_sum.get(cat_val, 0.0) + float(y_train[i])
            final_count[cat_val] = final_count.get(cat_val, 0) + 1

        for i in range(len(X_test)):
            cat_val = int(X_test[i, feat])
            s = final_sum.get(cat_val, 0.0)
            c = final_count.get(cat_val, 0)
            X_te_enc[i, feat] = (s + prior * global_mean) / (c + prior)

    return X_tr_enc, X_te_enc


# ============================================================
# КЛАС 1 — SymmetricTree (з cumsum оптимізацією)
# ============================================================
class SymmetricTree:
    """
    Симетричне дерево — на кожному рівні одне розбиття для всіх вузлів.

    Оптимізація через cumsum:
      Замість циклу по кожному порогу окремо —
      np.cumsum по відсортованих значеннях дає GL і GR
      для ВСІХ порогів одночасно за O(n).

    Це перетворює O(n_thresholds) внутрішній цикл на O(1) векторну операцію.
    """

    def __init__(self, max_depth=4, lambda_l2=3.0):
        self.max_depth   = max_depth
        self.lambda_l2   = lambda_l2
        self.splits      = []
        self.leaf_values = None

    def _best_split_for_level(self, X, G, H, node_ids, n_nodes):
        """
        Знаходить найкраще (feature, threshold) для всього рівня.
        Використовує cumsum для векторизації по порогах.
        """
        lam       = self.lambda_l2
        best_gain = 0
        best_feat = None
        best_thr  = None

        for feature in range(X.shape[1]):
            sorted_idx = np.argsort(X[:, feature])
            X_s   = X[sorted_idx, feature]
            G_s   = G[sorted_idx]
            H_s   = H[sorted_idx]
            nid_s = node_ids[sorted_idx]

            # Масив gainів по позиціях (сума по всіх вузлах)
            total_gains = np.zeros(len(X_s) - 1)

            for node in range(n_nodes):
                mask = nid_s == node
                if mask.sum() < 2:
                    continue

                pos    = np.where(mask)[0]   # позиції цього вузла у відсортованому масиві
                G_node = G_s[mask]
                H_node = H_s[mask]

                # cumsum дає GL для всіх порогів всередині вузла одночасно
                G_cum = np.cumsum(G_node)[:-1]
                H_cum = np.cumsum(H_node)[:-1]
                GT    = G_node.sum()
                HT    = H_node.sum()
                GR    = GT - G_cum
                HR    = HT - H_cum

                # Gain для кожного порогу — повністю векторизовано
                gains = 0.5 * (
                    G_cum**2 / (H_cum + lam) +
                    GR**2    / (HR   + lam) -
                    GT**2    / (HT   + lam)
                )

                # Забороняємо розбиття між однаковими значеннями
                valid = X_s[pos[:-1]] != X_s[pos[1:]]
                gains[~valid] = 0.0

                total_gains[pos[:-1]] += gains

            if total_gains.max() > best_gain:
                best_gain = total_gains.max()
                best_feat = feature
                idx       = int(np.argmax(total_gains))
                best_thr  = (X_s[idx] + X_s[idx + 1]) / 2

        return best_feat, best_thr, best_gain

    def fit(self, X, G, H):
        n        = len(G)
        node_ids = np.zeros(n, dtype=int)
        self.splits = []

        for depth in range(self.max_depth):
            feat, thr, gain = self._best_split_for_level(
                X, G, H, node_ids, 2 ** depth
            )
            if feat is None or gain == 0:
                break

            self.splits.append((feat, thr))

            # Оновлення node_ids — без Python циклу
            # ліві дочірні: node*2, праві: node*2+1
            node_ids = node_ids * 2
            node_ids[X[:, feat] > thr] += 1

        # Значення листків
        n_leaves         = 2 ** len(self.splits)
        self.leaf_values = np.zeros(n_leaves)
        lam              = self.lambda_l2

        for leaf in range(n_leaves):
            mask = node_ids == leaf
            if mask.sum() == 0:
                continue
            self.leaf_values[leaf] = -G[mask].sum() / (H[mask].sum() + lam)

    def predict(self, X):
        """Передбачення — d операцій замість рекурсії."""
        node_ids = np.zeros(len(X), dtype=int)
        for feat, thr in self.splits:
            node_ids = node_ids * 2
            node_ids[X[:, feat] > thr] += 1
        return self.leaf_values[node_ids]


# ============================================================
# КЛАС 2 — CatBoostRegressor
# ============================================================
class CatBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=4, l2_leaf_reg=3.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.l2_leaf_reg   = l2_leaf_reg
        self.trees         = []
        self.F0            = None

    def fit(self, X, y, verbose=False):
        self.F0 = float(np.mean(y))
        F = np.full(len(y), self.F0)

        for m in range(self.n_estimators):
            # MSE: G = F - y, H = 1
            G = F - y
            H = np.ones(len(y))

            tree = SymmetricTree(max_depth=self.max_depth, lambda_l2=self.l2_leaf_reg)
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
# КЛАС 3 — CatBoostClassifier
# ============================================================
class CatBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=4, l2_leaf_reg=3.0, n_classes=3):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.l2_leaf_reg   = l2_leaf_reg
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
            probs = self._softmax(F)

            for k in range(self.n_classes):
                # Cross-entropy: G = p - y, H = p*(1-p)
                G = probs[:, k] - (y == k).astype(float)
                H = np.maximum(probs[:, k] * (1 - probs[:, k]), 1e-6)

                tree = SymmetricTree(
                    max_depth=self.max_depth,
                    lambda_l2=self.l2_leaf_reg
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
# ORDERED ENCODING
# ============================================================
print("Ordered Target Encoding...")
X_tr_cls, X_te_cls = ordered_target_encoding(
    X_train, y_tr_cls, X_test, CAT_FEATURES, prior=0.5
)
X_tr_reg, X_te_reg = ordered_target_encoding(
    X_train, y_tr_n, X_test, CAT_FEATURES, prior=0.5
)
print(f"  день_тижня до:    {X_train[:3, 0]}")
print(f"  день_тижня після: {X_tr_cls[:3, 0].round(3)}\n")

# ============================================================
# НАВЧАННЯ — РЕГРЕСІЯ
# ============================================================
print("=" * 50)
print("CATBOOST — РЕГРЕСІЯ")
print("=" * 50)

cb_reg = CatBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    l2_leaf_reg=3.0
)
print("Навчання...")
cb_reg.fit(X_tr_reg, y_tr_n, verbose=True)

pred_te_n     = cb_reg.predict(X_te_reg)
test_mse      = float(np.mean((pred_te_n - y_te_n) ** 2))
test_rmse_uah = math.sqrt(test_mse * (y_max - y_min) ** 2)

print(f"\nTest RMSE: {test_rmse_uah:.0f} UAH")
print(f"(LightGBM:         1465 UAH)")
print(f"(XGBoost:          1401 UAH)")
print(f"(Лінійна регресія: 1295 UAH)")

# ============================================================
# НАВЧАННЯ — КЛАСИФІКАЦІЯ
# ============================================================
print("\n" + "=" * 50)
print("CATBOOST — КЛАСИФІКАЦІЯ")
print("=" * 50)

cb_cls = CatBoostClassifier(
    n_estimators=180,
    learning_rate=0.01,
    max_depth=4,
    l2_leaf_reg=10.0
)
print("Навчання...")
cb_cls.fit(X_tr_cls, y_tr_cls, verbose=True)

test_pred = cb_cls.predict(X_te_cls)
test_acc  = float(np.mean(test_pred == y_te_cls))

print(f"\nTest точність: {test_acc*100:.1f}%")
print(f"(Random Forest:     78.0%)")
print(f"(Дерево рішень:     76.3%)")
print(f"(XGBoost:           74.6%)")
print(f"(GB / LightGBM:     71.2%)")
print(f"(Нейронна мережа:   67.8%)")
print(f"(Baseline:          33.3%)")


# ============================================================
# ДЕТАЛЬНІ РЕЗУЛЬТАТИ
# ============================================================
print(f"\n{'='*50}")
print("ДЕТАЛЬНІ ПЕРЕДБАЧЕННЯ (тест):")
print(f"{'='*50}")
proba = cb_cls.predict_proba(X_te_cls)
for i in range(min(5, len(X_te_cls))):
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

# Статистика для кодування нового прикладу
prior       = 0.5
gm_cls      = float(np.mean(y_tr_cls))
gm_reg      = float(np.mean(y_tr_n))
day_cls     = {}; month_cls = {}
day_reg     = {}; month_reg = {}

for i in range(len(X_train)):
    d = int(X_train[i, 0]); m = int(X_train[i, 1])
    day_cls[d]   = day_cls.get(d, []);   day_cls[d].append(float(y_tr_cls[i]))
    month_cls[m] = month_cls.get(m, []); month_cls[m].append(float(y_tr_cls[i]))
    day_reg[d]   = day_reg.get(d, []);   day_reg[d].append(float(y_tr_n[i]))
    month_reg[m] = month_reg.get(m, []); month_reg[m].append(float(y_tr_n[i]))

def enc(vals, gm): return (sum(vals) + prior * gm) / (len(vals) + prior)

print(f"\n{'='*50}")
print("  ПЕРЕДБАЧЕННЯ ВИТРАТ — CATBOOST")
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

        x_cls = np.array([[
            enc(day_cls.get(день_тижня, []), gm_cls),
            enc(month_cls.get(місяць, []), gm_cls),
            вихідний, середня_кількість, вчора, позавчора
        ]])
        x_reg = np.array([[
            enc(day_reg.get(день_тижня, []), gm_reg),
            enc(month_reg.get(місяць, []), gm_reg),
            вихідний, середня_кількість, вчора, позавчора
        ]])

        probs    = cb_cls.predict_proba(x_cls)[0]
        pred_cls = int(np.argmax(probs))
        pred_sum = float(cb_reg.predict(x_reg)[0]) * (y_max - y_min) + y_min

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