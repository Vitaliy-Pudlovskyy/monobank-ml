import pandas as pd
import numpy as np
import math


np.random.seed(42)

# ============================================================
# КРОК 1 — ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================
df = pd.read_csv('../data_set/monobank_clean (1).csv')

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


y = денні_витрати['сума'].values

split   = int(len(X) * 0.8)
X_train = X[:split]; X_test = X[split:]
y_train = y[:split]; y_test = y[split:]

X_min = X_train.min(axis=0); X_max = X_train.max(axis=0)
y_min = y_train.min();        y_max = y_train.max()

X_train = (X_train - X_min) / (X_max - X_min)
X_test  = (X_test  - X_min) / (X_max - X_min)
y_train = (y_train - y_min) / (y_max - y_min)
y_test  = (y_test  - y_min) / (y_max - y_min)

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


def mse(y):
    mean = sum(y) / len(y)
    return sum((val - mean)**2 for val in y) / len(y)


def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    n = len(y)
    current_mse = mse(y)

    for feature in range(X.shape[1]):  # для кожної ознаки
        thresholds = set(X[:, feature])  # всі унікальні значення

        for threshold in thresholds:  # для кожного порогу
            # розділити на ліву і праву групу
            left_y = [y[i] for i in range(n) if X[i, feature] <= threshold]
            right_y = [y[i] for i in range(n) if X[i, feature] > threshold]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # порахувати зважений gini
            weighted_mse = (len(left_y) / n) * mse(left_y) +\
            (len(right_y) / n)* mse(right_y)

            # порахувати gain
            gain = current_mse - weighted_mse

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
        return Node(value=float(sum(y) / len(y)))

    feature, threshold, gain = best_split(X, y)

    if gain == 0:
        return Node(value=float(sum(y) / len(y)))

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

def predict(node , X):
    return [predict_one(node, x) for x in X]

tree = build_tree(X_train, y_train, max_depth=3)

train_pred = predict(tree, X_train)
test_pred  = predict(tree, X_test)

# Замість train_acc/test_acc:
train_mse = sum((train_pred[i] - y_train[i])**2 for i in range(len(y_train))) / len(y_train)
test_mse  = sum((test_pred[i]  - y_test[i])**2  for i in range(len(y_test)))  / len(y_test)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE:  {test_mse:.4f}")
print(f"Лінійна регресія була: 2.3640")


day_names = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пятниця', 'Субота', 'Неділя']

print("\n" + "=" * 45)
print("  ПЕРЕДБАЧЕННЯ ВИТРАТ — ДЕРЕВО РІШЕНЬ")
print("=" * 45)
test_mse_real  = test_mse  * (y_max - y_min)**2
train_mse_real = train_mse * (y_max - y_min)**2

print(f"\n{'='*40}")
print(f"Train RMSE: {math.sqrt(train_mse_real):.0f} UAH")
print(f"Test RMSE:  {math.sqrt(test_mse_real):.0f} UAH")
print(f"(помиляємось в середньому на X гривень)")
print(f"{'='*40}")

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

        x_input = np.array([день_тижня, місяць, вихідний, середня_кількість, вчора, позавчора], dtype=float)
        x_norm  = (x_input - X_min) / (X_max - X_min)

        result = predict_one(tree, x_norm)

        print(f"\n  День: {day_names[день_тижня]}, місяць {місяць}")
        print(f"  Передбачення: {result:.0f} UAH")

    except (ValueError, KeyboardInterrupt):
        print("  Помилка вводу")

