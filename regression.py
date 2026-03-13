"""
Monobank Витрати — Регресія
============================
Задача: передбачити суму витрат на день.
Дані: особиста виписка Монобанку за 2025 рік.

Висновок: обидві моделі (лінійна регресія і нейронна мережа)
показали схожі результати через малу кількість даних (292 дні)
і великий розкид витрат (std=1535 при середньому=851 UAH).
Задача потребує більше даних або зміни підходу.
"""

import pandas as pd
import numpy as np

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

витрати['день_тижня'] = витрати['Дата i час операції'].dt.dayofweek  # 0=Пн, 6=Нд
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

print(f"Всього днів з витратами: {len(денні_витрати)}")

# ============================================================
# КРОК 5 — НОРМАЛІЗАЦІЯ І РОЗДІЛЕННЯ
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

# ============================================================
# МОДЕЛЬ 1 — ЛІНІЙНА РЕГРЕСІЯ (NumPy)
# ============================================================
w_lr = np.random.randn(6, 1) * 0.01
b_lr = np.zeros(1)

def forward_lr(x):
    return x @ w_lr + b_lr

def loss_lr(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))

def backward_lr(x, y_true, y_pred):
    delta = 2 * (y_pred - y_true)
    dw    = x.reshape(-1, 1) * delta.reshape(1, -1)
    db    = delta
    return dw, db

def update_lr(dw, db, lr=0.005):
    global w_lr, b_lr
    w_lr = w_lr - lr * dw
    b_lr = b_lr - lr * db

print("\n--- Лінійна регресія ---")
for epoch in range(120):
    total_loss = 0
    for i in range(len(X_train)):
        x      = X_train[i]
        y      = y_train[i]
        y_pred = forward_lr(x)
        total_loss += loss_lr(y_pred, y)
        dw, db = backward_lr(x, y, y_pred)
        update_lr(dw, db)

    if epoch % 20 == 0:
        print(f"  epoch {epoch:4d} | loss: {total_loss:.4f}")

print("\nЛінійна регресія — передбачення vs реальність:")
for i in range(5):
    y_pred     = forward_lr(X_test[i])
    pred_real  = float(y_pred.flatten()[0]) * (y_max - y_min) + y_min
    true_real  = y_test[i] * (y_max - y_min) + y_min
    print(f"  Передбачено: {pred_real:6.0f} UAH | Реально: {true_real:6.0f} UAH")

# ============================================================
# МОДЕЛЬ 2 — НЕЙРОННА МЕРЕЖА (NumPy)
# ============================================================
w1 = np.random.randn(6, 6) * 0.01;  b1 = np.zeros(6)
w2 = np.random.randn(6, 3) * 0.01;  b2 = np.zeros(3)
w3 = np.random.randn(3, 1) * 0.01;  b3 = np.zeros(1)

def relu(z):
    return np.maximum(0, z)

def forward_nn(x):
    z1 = x @ w1 + b1;  h1 = relu(z1)
    z2 = h1 @ w2 + b2; h2 = relu(z2)
    z3 = h2 @ w3 + b3
    return z3, h1, h2, z1, z2

def loss_nn(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))

def backward_nn(x, y_true, y_pred, h1, h2, z1, z2):
    delta3 = 2 * (y_pred - y_true)
    dw3    = h2.reshape(-1, 1) @ delta3.reshape(1, -1)
    db3    = delta3

    delta2 = (delta3 @ w3.T) * (z2 > 0)
    dw2    = h1.reshape(-1, 1) @ delta2.reshape(1, -1)
    db2    = delta2

    delta1 = (delta2 @ w2.T) * (z1 > 0)
    dw1    = x.reshape(-1, 1) @ delta1.reshape(1, -1)
    db1    = delta1
    return dw1, db1, dw2, db2, dw3, db3

def update_nn(dw1, db1, dw2, db2, dw3, db3, lr=0.005):
    global w1, b1, w2, b2, w3, b3
    w1 -= lr * dw1;  b1 -= lr * db1
    w2 -= lr * dw2;  b2 -= lr * db2
    w3 -= lr * dw3;  b3 -= lr * db3

print("\n--- Нейронна мережа (early stopping) ---")
best_loss  = float('inf')
patience   = 500
no_improve = 0

for epoch in range(50000):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]; y = y_train[i]
        y_pred, h1, h2, z1, z2 = forward_nn(x)
        total_loss += loss_nn(y_pred, y)
        dw1, db1, dw2, db2, dw3, db3 = backward_nn(x, y, y_pred, h1, h2, z1, z2)
        update_nn(dw1, db1, dw2, db2, dw3, db3)

    if total_loss < best_loss:
        best_loss = total_loss; no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"  Early stopping на епосі {epoch}")
        break

    if epoch % 2000 == 0:
        print(f"  epoch {epoch:5d} | loss: {total_loss:.4f}")

print("\nНейронна мережа — передбачення vs реальність:")
for i in range(5):
    y_pred, *_ = forward_nn(X_test[i])
    pred_real  = float(y_pred[0]) * (y_max - y_min) + y_min
    true_real  = y_test[i] * (y_max - y_min) + y_min
    print(f"  Передбачено: {pred_real:6.0f} UAH | Реально: {true_real:6.0f} UAH")

# ============================================================
# ПІДСУМОК
# ============================================================
test_loss_lr = sum(loss_lr(forward_lr(X_test[i]), y_test[i]) for i in range(len(X_test)))
test_loss_nn = sum(loss_nn(forward_nn(X_test[i])[0], y_test[i]) for i in range(len(X_test)))

print(f"\n{'='*40}")
print(f"Test MSE — Лінійна регресія: {test_loss_lr:.4f}")
print(f"Test MSE — Нейронна мережа:  {test_loss_nn:.4f}")
print(f"{'='*40}")