import math

import pandas as pd
import numpy as np

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


def manhattan(a, b):
    return sum(abs(a[i] - b[i]) for i in range(len(a)))

def knn_predict(X_train, y_train, x_new, k=5):
    distances = []
    for i in range(len(X_train)):
        dist = manhattan(X_train[i],x_new )
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    votes = [label for _, label in k_nearest]
    return max(set(votes), key = votes.count)

def predict(X_train, y_train, X_test, k=3):
    return [knn_predict(X_train, y_train, x, k) for x in X_test]



test_pred  = predict(X_train, y_train, X_test, k=3)
test_acc   = sum(test_pred[i] == y_test[i] for i in range(len(y_test))) / len(y_test)
print(f"Test точність KNN: {test_acc*100:.1f}%")
