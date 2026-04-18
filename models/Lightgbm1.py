"""
Monobank Витрати — LightGBM (з нуля)
======================================
Реалізація з нуля (тільки NumPy і Pandas).

Ключові відмінності від XGBoost:
  1. Leaf-wise замість level-wise — розбиваємо найгірший листок
  2. GOSS — навчаємось тільки на важливих прикладах
  3. num_leaves замість max_depth як головний параметр
"""

import math
import pandas as pd
import numpy as np

np.random.seed(42)

# ============================================================
# КРОК 1-4 — ДАНІ (стандартний pipeline)
# ============================================================
df = pd.read_csv('../data_set/monobank_clean (1).csv')
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
# КРОК 5 — ПІДГОТОВКА
# ============================================================
X = денні_витрати[['день_тижня', 'місяць', 'вихідний', 'кількість', 'вчора', 'позавчора']].values

bins = [0, 200, 600, float('inf')]
LABELS = ['0-200 UAH', '200-600 UAH', '600+ UAH']
денні_витрати['клас'] = pd.cut(денні_витрати['сума'], bins=bins, labels=[0, 1, 2])
y_cls = денні_витрати['клас'].astype(int).values
y_reg = денні_витрати['сума'].values

split = int(len(X) * 0.8)
X_train = X[:split];
X_test = X[split:]
y_tr_cls = y_cls[:split];
y_te_cls = y_cls[split:]
y_tr_reg = y_reg[:split];
y_te_reg = y_reg[split:]

X_min = X_train.min(axis=0);
X_max = X_train.max(axis=0)
X_tr = (X_train - X_min) / (X_max - X_min)
X_te = (X_test - X_min) / (X_max - X_min)

y_min = y_tr_reg.min();
y_max = y_tr_reg.max()
y_tr_n = (y_tr_reg - y_min) / (y_max - y_min)
y_te_n = (y_te_reg - y_min) / (y_max - y_min)

print(f"Train: {len(X_tr)} | Test: {len(X_te)}\n")


# ============================================================
# GOSS — відбір важливих прикладів
# ============================================================
def goss_sample(G, top_rate=0.2, other_rate=0.1):
    """
    Gradient-based One-Side Sampling.

    Ідея: приклади з великим |градієнтом| — важливі (модель їх не знає).
    Приклади з малим |градієнтом| — модель вже добре навчилась на них.

    Кроки:
      1. Сортуємо за |G| — від більшого до меншого
      2. Беремо топ top_rate (20%) — всі
      3. З решти беремо випадкові other_rate (10%)
      4. Малим множимо вагу: (1 - top_rate) / other_rate
         щоб компенсувати що їх менше

    Повертає: індекси вибраних прикладів + їх ваги
    """
    n = len(G)

    # Кількість прикладів у кожній групі
    n_large = max(1, int(n * top_rate))
    n_small = max(1, int(n * other_rate))

    # Сортуємо за абсолютним значенням градієнта (великі спочатку)
    sorted_idx = np.argsort(-np.abs(G))

    # Топ n_large — великі градієнти, беремо всі
    large_idx = sorted_idx[:n_large]

    # Решта — малі градієнти, беремо випадкові n_small
    small_pool = sorted_idx[n_large:]
    small_idx = np.random.choice(small_pool, size=min(n_small, len(small_pool)), replace=False)

    # Ваги: великі = 1.0, малі = компенсаційна вага
    weight_small = (1.0 - top_rate) / other_rate

    selected_idx = np.concatenate([large_idx, small_idx])
    weights = np.ones(len(selected_idx))
    weights[n_large:] = weight_small  # малі отримують більшу вагу

    return selected_idx, weights
# ============================================================
# КЛАС 1 — LightGBMTree (leaf-wise дерево)
# ============================================================
class LightGBMTree:
    """
    Дерево що будується leaf-wise (а не level-wise).

    Level-wise (XGBoost):
        Розбиваємо ВСІ листки на поточному рівні.
        Дерево завжди симетричне.

    Leaf-wise (LightGBM):
        Шукаємо листок з найбільшим gain і розбиваємо ТІЛЬКИ його.
        Дерево асиметричне — глибше там де важливіше.

    Параметр num_leaves замість max_depth:
        Зупиняємось коли кількість листків досягла num_leaves.
    """

    def __init__(self, num_leaves=31, min_data_in_leaf=5, lambda_l2=0.1):
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.leaves = []  # список листків (індекси прикладів)
        self.nodes = {}  # id_вузла → вузол
        self.root_id = 0
        self._node_counter = 0

    def fit(self, X, G, H, weights=None):
        """
        X       — ознаки
        G       — градієнти
        H       — гессіани
        weights — ваги прикладів (від GOSS)
        """
        n = len(G)
        if weights is None:
            weights = np.ones(n)

        # Зважені градієнти і гессіани
        Gw = G * weights
        Hw = H * weights

        # Починаємо з одного кореневого листка (всі індекси)
        root_indices = np.arange(n)
        root_id = self._new_leaf(root_indices, Gw, Hw)
        self.root_id = root_id

        # Поки не досягли num_leaves — шукаємо найкращий листок для розбиття
        while len(self.leaves) < self.num_leaves:
            best_gain = 0
            best_leaf_id = None
            best_feature = None
            best_threshold = None
            best_left_idx = None
            best_right_idx = None

            # Перебираємо всі поточні листки
            for leaf_id in self.leaves:
                leaf = self.nodes[leaf_id]
                indices = leaf['indices']

                # Замало прикладів — не розбиваємо
                if len(indices) < 2 * self.min_data_in_leaf:
                    continue

                X_leaf = X[indices]
                G_leaf = Gw[indices]
                H_leaf = Hw[indices]

                feat, thr, gain, l_idx, r_idx = self._best_split(
                    X_leaf, G_leaf, H_leaf, indices
                )

                if feat is not None and gain > best_gain:
                    best_gain = gain
                    best_leaf_id = leaf_id
                    best_feature = feat
                    best_threshold = thr
                    best_left_idx = l_idx
                    best_right_idx = r_idx

            # Немає корисного розбиття — зупиняємось
            if best_leaf_id is None:
                break

            # Розбиваємо найкращий листок
            self._split_leaf(
                best_leaf_id, best_feature, best_threshold,
                best_left_idx, best_right_idx, Gw, Hw
            )

        # Фінальні значення листків
        for leaf_id in self.leaves:
            leaf = self.nodes[leaf_id]
            G_sum = Gw[leaf['indices']].sum()
            H_sum = Hw[leaf['indices']].sum()
            leaf['value'] = -G_sum / (H_sum + self.lambda_l2)

    def predict(self, X):
        return np.array([self._predict_one(self.root_id, x) for x in X])

    # ---------- внутрішні методи ----------

    def _new_node_id(self):
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _new_leaf(self, indices, Gw, Hw):
        """Створити новий листок і додати його до списку листків."""
        nid = self._new_node_id()
        self.nodes[nid] = {
            'is_leaf': True,
            'indices': indices,
            'value': 0.0,
            'feature': None,
            'threshold': None,
            'left_id': None,
            'right_id': None,
        }
        self.leaves.append(nid)
        return nid

    def _split_leaf(self, leaf_id, feature, threshold, left_idx, right_idx, Gw, Hw):
        """Перетворити листок на внутрішній вузол, створити два дочірні листки."""
        node = self.nodes[leaf_id]

        # Видаляємо з листків — більше не листок
        self.leaves.remove(leaf_id)

        # Створюємо дочірні листки
        left_id = self._new_leaf(left_idx, Gw, Hw)
        right_id = self._new_leaf(right_idx, Gw, Hw)

        # Оновлюємо вузол
        node['is_leaf'] = False
        node['feature'] = feature
        node['threshold'] = threshold
        node['left_id'] = left_id
        node['right_id'] = right_id

    def _leaf_score(self, G_sum, H_sum):
        """Оцінка листка для формули Gain: G²/(H+λ)"""
        return (G_sum ** 2) / (H_sum + self.lambda_l2)

    def _best_split(self, X_leaf, G_leaf, H_leaf, global_indices):
        """Знайти найкраще розбиття для одного листка."""
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        G_total = G_leaf.sum()
        H_total = H_leaf.sum()
        n = len(G_leaf)

        for feature in range(X_leaf.shape[1]):
            sorted_idx = np.argsort(X_leaf[:, feature])
            X_s = X_leaf[sorted_idx, feature]
            G_s = G_leaf[sorted_idx]
            H_s = H_leaf[sorted_idx]

            G_left_cum = np.cumsum(G_s)
            H_left_cum = np.cumsum(H_s)

            for i in range(self.min_data_in_leaf - 1,
                           n - self.min_data_in_leaf):
                if X_s[i] == X_s[i + 1]:
                    continue

                G_L = G_left_cum[i]
                H_L = H_left_cum[i]
                G_R = G_total - G_L
                H_R = H_total - H_L

                gain = 0.5 * (
                        self._leaf_score(G_L, H_L) +
                        self._leaf_score(G_R, H_R) -
                        self._leaf_score(G_total, H_total)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (X_s[i] + X_s[i + 1]) / 2

                    # Переводимо назад до глобальних індексів
                    left_local = sorted_idx[:i + 1]
                    right_local = sorted_idx[i + 1:]
                    best_left_idx = global_indices[left_local]
                    best_right_idx = global_indices[right_local]

        return best_feature, best_threshold, best_gain, best_left_idx, best_right_idx

    def _predict_one(self, node_id, x):
        node = self.nodes[node_id]
        if node['is_leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left_id'], x)
        return self._predict_one(node['right_id'], x)


# ============================================================
# КЛАС 2 — LightGBMRegressor
# ============================================================
class LightGBMRegressor:
    """
    LightGBM для регресії.
    Градієнт і гессіан для MSE — ті самі що в XGBoost.
    Різниця в тому ЯК будується дерево (leaf-wise + GOSS).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 num_leaves=15, min_data_in_leaf=5,
                 lambda_l2=0.1, top_rate=0.2, other_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.trees = []
        self.F0 = None

    def fit(self, X, y, verbose=False):
        self.F0 = float(np.mean(y))
        F = np.full(len(y), self.F0)

        for m in range(self.n_estimators):
            G = F - y
            H = np.ones(len(y))

            # GOSS: відбираємо важливі приклади
            idx, weights = goss_sample(G, self.top_rate, self.other_rate)

            tree = LightGBMTree(
                num_leaves=self.num_leaves,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l2=self.lambda_l2
            )
            # Навчаємо тільки на вибраних прикладах
            tree.fit(X[idx], G[idx], H[idx], weights)
            self.trees.append(tree)

            F = F + self.learning_rate * tree.predict(X)

            if verbose and (m + 1) % 20 == 0:
                mse = float(np.mean((y - F) ** 2))
                print(f"  дерево {m + 1:3d}/{self.n_estimators} | train MSE: {mse:.4f}")

        return self

    def predict(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return F


# ============================================================
# КЛАС 3 — LightGBMClassifier
# ============================================================
class LightGBMClassifier:
    """
    LightGBM для класифікації (3 класи).
    Градієнт і гессіан — ті самі що в XGBoost (cross-entropy).
    Головна різниця — leaf-wise дерева + GOSS.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 num_leaves=15, min_data_in_leaf=5,
                 lambda_l2=0.1, top_rate=0.2, other_rate=0.1,
                 n_classes=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.n_classes = n_classes
        self.trees = [[] for _ in range(n_classes)]
        self.F0 = None

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
                G = probs[:, k] - (y == k).astype(float)
                H = np.maximum(probs[:, k] * (1 - probs[:, k]), 1e-6)

                # GOSS на градієнтах класу k
                idx, weights = goss_sample(G, self.top_rate, self.other_rate)

                tree = LightGBMTree(
                    num_leaves=self.num_leaves,
                    min_data_in_leaf=self.min_data_in_leaf,
                    lambda_l2=self.lambda_l2
                )
                tree.fit(X[idx], G[idx], H[idx], weights)
                self.trees[k].append(tree)

                F[:, k] += self.learning_rate * tree.predict(X)

            if verbose and (m + 1) % 20 == 0:
                preds = np.argmax(probs, axis=1)
                acc = float(np.mean(preds == y))
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

#
# # ============================================================
# # НАВЧАННЯ — РЕГРЕСІЯ
# # ============================================================
print("=" * 50)
print("LIGHTGBM — РЕГРЕСІЯ")
print("=" * 50)

lgb_reg = LightGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_data_in_leaf=5,
    lambda_l2=0.1,
    top_rate=0.2,
    other_rate=0.1
)
print("Навчання...")
lgb_reg.fit(X_tr, y_tr_n, verbose=True)

pred_te_n = lgb_reg.predict(X_te)
test_mse = float(np.mean((pred_te_n - y_te_n) ** 2))
test_rmse_uah = math.sqrt(test_mse * (y_max - y_min) ** 2)

print(f"\nTest RMSE: {test_rmse_uah:.0f} UAH")
print(f"(XGBoost:           1401 UAH)")
print(f"(Лінійна регресія:  1295 UAH)")
print(f"(Нейронна мережа:   1541 UAH)")

# ============================================================
# НАВЧАННЯ — КЛАСИФІКАЦІЯ
# ============================================================
print("\n" + "=" * 50)
print("LIGHTGBM — КЛАСИФІКАЦІЯ")
print("=" * 50)

lgb_cls = LightGBMClassifier(
    n_estimators=380,
    learning_rate=0.01,
    num_leaves=15,
    min_data_in_leaf=20,
    lambda_l2=1.5,
    top_rate=0.2,
    other_rate=0.1
)
print("Навчання...")
lgb_cls.fit(X_tr, y_tr_cls, verbose=True)

test_pred = lgb_cls.predict(X_te)
test_acc = float(np.mean(test_pred == y_te_cls))

print(f"\nTest точність: {test_acc * 100:.1f}%")
print(f"(Random Forest:     78.0%)")
print(f"(Дерево рішень:     76.3%)")
print(f"(XGBoost:           74.6%)")
print(f"(Gradient Boosting: 71.2%)")
print(f"(Нейронна мережа:   67.8%)")
print(f"(Baseline:          33.3%)")

# ============================================================
# ВПЛИВ num_leaves
# ============================================================
print(f"\n{'=' * 50}")
print("ВПЛИВ num_leaves НА ТОЧНІСТЬ:")
print(f"{'─' * 50}")
print(f"  {'num_leaves':>12}  {'Точність':>10}")
print(f"  {'─' * 30}")

for nl in [4, 8, 15, 31, 63]:
    model = LightGBMClassifier(
        n_estimators=100, learning_rate=0.05,
        num_leaves=nl, min_data_in_leaf=5, lambda_l2=0.1
    )
    model.fit(X_tr, y_tr_cls)
    acc = float(np.mean(model.predict(X_te) == y_te_cls))
    print(f"  num_leaves={nl:>3}  →  {acc * 100:5.1f}%")

# ============================================================
# ДЕТАЛЬНІ РЕЗУЛЬТАТИ
# ============================================================
print(f"\n{'=' * 50}")
print("ДЕТАЛЬНІ ПЕРЕДБАЧЕННЯ (тест):")
print(f"{'=' * 50}")

proba = lgb_cls.predict_proba(X_te)
for i in range(min(5, len(X_te))):
    true_label = LABELS[y_te_cls[i]]
    pred_label = LABELS[test_pred[i]]
    mark = '✓' if test_pred[i] == y_te_cls[i] else '✗'
    print(f"  {mark} Реально: {true_label:12s} | "
          f"Передбачено: {pred_label:12s} | "
          f"Ймов: {proba[i][0] * 100:.0f}% / {proba[i][1] * 100:.0f}% / {proba[i][2] * 100:.0f}%")

# ============================================================
# ІНТЕРАКТИВНИЙ РЕЖИМ
# ============================================================
day_names = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пятниця', 'Субота', 'Неділя']

print(f"\n{'=' * 50}")
print("  ПЕРЕДБАЧЕННЯ ВИТРАТ — LIGHTGBM")
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

        probs = lgb_cls.predict_proba(x_norm.reshape(1, -1))[0]
        pred_cls = int(np.argmax(probs))

        pred_sum_n = lgb_reg.predict(x_norm.reshape(1, -1))[0]
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