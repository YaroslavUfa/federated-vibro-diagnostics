import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Загружаем данные
X = np.load("X_mel.npy")
y = np.load("y_labels.npy")
X = X[..., np.newaxis]  # (6, 64, 41, 1)
y = y.astype(np.float32)

print("Data shape:", X.shape, "Labels:", y)

# === ФУНКЦИЯ: Создать модель ===
def create_model():
    model = keras.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=(64, 41, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# === ФУНКЦИЯ: Обучить модель на части данных ===
def train_one_epoch(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)

# === ФУНКЦИЯ: Получить веса ===
def get_weights(model):
    return [w.numpy() for w in model.trainable_weights]

# === ФУНКЦИЯ: Установить веса ===
def set_weights(model, weights):
    for w, new_w in zip(model.trainable_weights, weights):
        w.assign(new_w)

# === ФУНКЦИЯ: Усреднить веса двух моделей ===
def average_weights(weights_a, weights_b):
    return [(w_a + w_b) / 2.0 for w_a, w_b in zip(weights_a, weights_b)]

# === FEDERATED LEARNING СИМУЛЯЦИЯ ===
print("\n=== Federated Learning Simulation ===\n")

# Разделяем данные: смешанные (перекрывающиеся классы)
indices_a = [0, 1, 3]
indices_b = [2, 4, 5]

X_a = X[indices_a]
y_a = y[indices_a]
X_b = X[indices_b]
y_b = y[indices_b]

print(f"Client A data: {y_a} (labels)")
print(f"Client B data: {y_b} (labels)\n")

# Создаём две модели
model_a = create_model()
model_b = create_model()
set_weights(model_b, get_weights(model_a))

print("Initial weights set equal for both models\n")

# Сохраняем метрики
history_a_before = []
history_b_before = []
history_a_after = []
history_b_after = []
history_global = []

# Симулируем 10 раундов
for round_num in range(10):
    print(f"--- Round {round_num + 1} ---")
    
    train_one_epoch(model_a, X_a, y_a)
    acc_a = model_a.evaluate(X_a, y_a, verbose=0)[1]
    
    train_one_epoch(model_b, X_b, y_b)
    acc_b = model_b.evaluate(X_b, y_b, verbose=0)[1]
    
    # Усредняем веса
    w_a = get_weights(model_a)
    w_b = get_weights(model_b)
    w_avg = average_weights(w_a, w_b)
    
    set_weights(model_a, w_avg)
    set_weights(model_b, w_avg)
    
    # Проверяем точность после
    acc_a_after = model_a.evaluate(X_a, y_a, verbose=0)[1]
    acc_b_after = model_b.evaluate(X_b, y_b, verbose=0)[1]
    
    # Глобальная точность
    global_acc = model_a.evaluate(X, y, verbose=0)[1]
    
    # Сохраняем
    history_a_before.append(acc_a)
    history_b_before.append(acc_b)
    history_a_after.append(acc_a_after)
    history_b_after.append(acc_b_after)
    history_global.append(global_acc)
    
    print(f"Model A - Before: {acc_a:.3f}, After: {acc_a_after:.3f}")
    print(f"Model B - Before: {acc_b:.3f}, After: {acc_b_after:.3f}")
    print(f"Global: {global_acc:.3f}")
    print()

print("=== FL Simulation Complete ===\n")

# === ГРАФИК ===
plt.figure(figsize=(12, 6))

rounds = range(1, 11)
plt.plot(rounds, history_global, 'o-', label='Global Accuracy', linewidth=2, markersize=8)
plt.plot(rounds, history_a_after, 's-', label='Model A (After)', linewidth=2, markersize=6)
plt.plot(rounds, history_b_after, '^-', label='Model B (After)', linewidth=2, markersize=6)

plt.xlabel('FL Round', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Federated Learning: Accuracy Over Rounds', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.1])
plt.tight_layout()
plt.show()
