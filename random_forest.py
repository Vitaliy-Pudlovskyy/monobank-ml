import pandas as pd
import numpy as np
import random


np.random.seed(13)
random.seed(13)
# ============================================================
# КРОК 1 — ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================
df = pd.read_csv('data_set/monobank_clean (1).csv')

# ============================================================
# КРОК 2 — ФІЛЬТРАЦІЯ: тільки витрати
# ============================================================
витрати = df[df['Сума в валюті картки (UAH)'] < 0].copy()

# ============================================================
# КРОК 3 — ПІДГОТОВКА КОЛОНОК
# ============================================================
витрати['Сума в валюті картки (UAH)'] = pd.to_numeric(витрати['Сума в валюті картки (UAH)'])

витрати['Дата i час операції'] = pd.to_datetime(
    витрати['Дата i час операції'],
    format='%d.%m.%Y %H:%M:%S'
)

витрати['день_тижня'] = витрати['Дата i час операції'].dt.dayofweek
витрати['місяць']     = витрати['Дата i час операції'].dt.month
витрати['вихідний']   = витрати['день_тижня'].apply(lambda x: 1 if x >= 5 else 0)

# ============================================================
# КРОК 4 — ГРУПУВАННЯ ПО ДНЯХ
# ============================================================
витрати['дата'] = витрати['Дата i час операції'].dt.date

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

# Середня кількість транзакцій — використовується як дефолт в інтерактивному режимі
середня_кількість = round(денні_витрати['кількість'].mean())

print(f"Всього днів з витратами: {len(денні_витрати)}")

# ============================================================
# КРОК 5 — КЛАСИ І РОЗДІЛЕННЯ
# ============================================================
X = денні_витрати[['день_тижня', 'місяць', 'вихідний', 'кількість', 'вчора', 'позавчора']].values

bins   = [0, 200, 600, float('inf')]
LABELS = ['0-200 UAH', '200-600 UAH', '600+ UAH']
денні_витрати['клас'] = pd.cut(денні_витрати['сума'], bins=bins, labels=[0, 1, 2])
y = денні_витрати['клас'].astype(int).values

print("Розподіл класів:")
for i, label in enumerate(LABELS):
    print(f"  {label}: {(y == i).sum()} днів")

split   = int(len(X) * 0.8)
X_train = X[:split]; X_test = X[split:]
y_train = y[:split]; y_test = y[split:]

X_min = X_train.min(axis=0); X_max = X_train.max(axis=0)
X_train = (X_train - X_min) / (X_max - X_min)
X_test  = (X_test  - X_min) / (X_max - X_min)

print(f"Train: {len(X_train)} днів | Test: {len(X_test)} днів")


class Node:
    def __init__(self, feature=None , threshold=None, value=None):
        self.feature    = feature
        self.threshold = threshold
        self.value     = value
        self.left      = None
        self.right     = None

    def is_leaf(self):
        return  self.value is not None


def gini(y):
    n = len(y)
    unieque_classes = set(y)
    impurity = 1
    for c in unieque_classes:
        p = (y == c).sum()/n
        impurity -= p ** 2
    return impurity


def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    n = len(y)
    current_gini = gini(y)

    for feature in range(X.shape[1]):  # для кожної ознаки
        thresholds = set(X[:, feature])  # всі унікальні значення

        for threshold in thresholds:  # для кожного порогу
            # розділити на ліву і праву групу
            left_y = [y[i] for i in range(n) if X[i, feature] <= threshold]
            right_y = [y[i] for i in range(n) if X[i, feature] > threshold]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # порахувати зважений gini
            weighted_gini = (len(left_y) / n) * gini(left_y) +\
            (len(right_y) / n)* gini(right_y)

            # порахувати gain
            gain = current_gini - weighted_gini

            # якщо краще — зберегти
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def build_tree(X, y, depth=0, max_depth=5):
    if len(set(y)) == 1:
        return Node(value=y[0])

    if depth >= max_depth:
        counts = {c: (y == c).sum() for c in set(y)}
        return Node(value=max(counts, key=counts.get))

    feature, threshold, gain = best_split(X, y)

    if gain == 0:
        counts = {c: (y == c).sum() for c in set(y)}
        return Node(value=max(counts, key=counts.get))

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

    node = Node(feature=feature, threshold=threshold)
    node.left = left
    node.right = right
    return node


def predict_one(node, x):
    if node.is_leaf():
        return node.value

    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


def random_forest_train(X, y, n_trees=4, max_depth=5, sample_size=0.8):
    trees = []
    for _ in range(n_trees):
        # випадкова підвибірка даних
        indices = random.sample(range(len(X)), int(len(X) * sample_size))
        X_sample = X[indices]
        y_sample = y[indices]

        # будуємо одне дерево
        tree = build_tree(X_sample, y_sample, max_depth=max_depth)
        trees.append(tree)

    return trees


def random_forest_predict(trees, X):
    predictions = []
    for x in X:
        votes = [predict_one(tree, x) for tree in trees]
        result = max(set(votes), key = votes.count)
        predictions.append(result)
    return predictions

for n in [6]:
    trees = random_forest_train(X_train, y_train, n_trees=n)
    test_pred = random_forest_predict(trees, X_test)
    test_acc  = sum(test_pred[i] == y_test[i] for i in range(len(y_test))) / len(y_test)
    print(f"n_trees={n:3d} → {test_acc*100:.1f}%")


