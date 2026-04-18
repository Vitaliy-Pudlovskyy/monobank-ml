"""
Monobank Витрати — Класифікація
================================
Задача: передбачити категорію витрат на день
  0 → малий день    (0–200 UAH)
  1 → середній день (200–600 UAH)
  2 → великий день  (600+ UAH)

Архітектура: 6 → 10 → 6 → 3 (Softmax)
Точність: 66.1% (випадкове вгадування: 33.3%)
"""

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

# ============================================================
# МОДЕЛЬ — НЕЙРОННА МЕРЕЖА
# ============================================================
w1, b1 = np.random.randn(6, 10) * 0.01, np.zeros(10)
w2, b2 = np.random.randn(10, 6) * 0.01, np.zeros(6)
w3, b3 = np.random.randn(6, 3)  * 0.01, np.zeros(3)


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def cross_entropy(y_pred, y_true):
    return -np.log(y_pred[y_true] + 1e-9)


def forward(x):
    z1 = x @ w1 + b1;  h1 = relu(z1)
    z2 = h1 @ w2 + b2; h2 = relu(z2)
    z3 = h2 @ w3 + b3
    out = softmax(z3)
    return out, h1, h2, z1, z2


def backward(x, y_true, out, h1, h2, z1, z2):
    delta3 = out.copy()
    delta3[y_true] -= 1

    dw3 = h2.reshape(-1, 1) @ delta3.reshape(1, -1)
    db3 = delta3

    delta2 = (delta3 @ w3.T) * (z2 > 0)
    dw2 = h1.reshape(-1, 1) @ delta2.reshape(1, -1)
    db2 = delta2

    delta1 = (delta2 @ w2.T) * (z1 > 0)
    dw1 = x.reshape(-1, 1) @ delta1.reshape(1, -1)
    db1 = delta1
    return dw1, db1, dw2, db2, dw3, db3


def update(dw1, db1, dw2, db2, dw3, db3, lr=0.01):
    global w1, b1, w2, b2, w3, b3
    w1 -= lr * dw1;  b1 -= lr * db1
    w2 -= lr * dw2;  b2 -= lr * db2
    w3 -= lr * dw3;  b3 -= lr * db3


# ============================================================
# НАВЧАННЯ З EARLY STOPPING
# ============================================================
best_loss  = float('inf')
patience   = 280
no_improve = 0

print("\nНавчання:")
for epoch in range(5000):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]; y = y_train[i]
        out, h1, h2, z1, z2 = forward(x)
        total_loss += cross_entropy(out, y)
        dw1, db1, dw2, db2, dw3, db3 = backward(x, y, out, h1, h2, z1, z2)
        update(dw1, db1, dw2, db2, dw3, db3, lr=0.01)

    if total_loss < best_loss:
        best_loss = total_loss; no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"  Early stopping на епосі {epoch}")
        break

    if epoch % 500 == 0:
        print(f"  epoch {epoch:4d} | loss: {total_loss:.4f}")

# ============================================================
# ОЦІНКА
# ============================================================
correct = 0
for i in range(len(X_test)):
    out, _, _, _, _ = forward(X_test[i])
    if np.argmax(out) == y_test[i]:
        correct += 1

print(f"\nТочність: {correct}/{len(X_test)} = {correct/len(X_test)*100:.1f}%")
print(f"(Випадкове вгадування: 33.3%)\n")

# ============================================================
# ІНТЕРАКТИВНИЙ РЕЖИМ
# ============================================================
day_names = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пятниця', 'Субота', 'Неділя']

print("=" * 45)
print("  ПЕРЕДБАЧЕННЯ ВИТРАТ НА ДЕНЬ")
print("=" * 45)

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

        # Нормалізація вводу (кількість = середнє по датасету)
        x_input = np.array([день_тижня, місяць, вихідний, середня_кількість, вчора, позавчора], dtype=float)
        x_norm  = (x_input - X_min) / (X_max - X_min)

        out, _, _, _, _ = forward(x_norm)
        predicted = np.argmax(out)

        print(f"\n  День: {day_names[день_тижня]}, місяць {місяць}")
        print(f"  ─────────────────────────────")
        for i, label in enumerate(LABELS):
            bar  = '█' * int(out[i] * 20)
            mark = ' ←' if i == predicted else ''
            print(f"  {label:12s} {out[i]*100:5.1f}%  {bar}{mark}")
        print(f"  ─────────────────────────────")
        print(f"  Передбачення: {LABELS[predicted]}")

    except (ValueError, KeyboardInterrupt):
        print("  Помилка вводу")
