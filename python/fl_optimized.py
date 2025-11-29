import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Загружаем ВСЕ данные
X = np.load("X_mel_all.npy")
y = np.load("y_labels_all.npy")
X = X[..., np.newaxis]
y = y.astype(np.float32)

print(f"Total data: X={X.shape}, y={y.shape}")
print(f"Class distribution: {(y==0).sum()} healthy, {(y==1).sum()} faulty\n")

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# === УЛУЧШЕННАЯ МОДЕЛЬ ===
def create_model_v2():
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 41, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_one_epoch(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=3, batch_size=4, verbose=0)

def get_weights(model):
    return [w.numpy() for w in model.trainable_weights]

def set_weights(model, weights):
    for w, new_w in zip(model.trainable_weights, weights):
        w.assign(new_w)

def average_weights(weights_a, weights_b):
    return [(w_a + w_b) / 2.0 for w_a, w_b in zip(weights_a, weights_b)]

# === FEDERATED LEARNING ===
print("=== Federated Learning (Optimized) ===\n")

# Разделяем данные так, чтобы у каждого было оба класса
n = len(X_train)
indices_a = np.concatenate([
    np.where(y_train == 0)[0][:n//4],
    np.where(y_train == 1)[0][:n//4]
])
indices_b = np.concatenate([
    np.where(y_train == 0)[0][n//4:n//2],
    np.where(y_train == 1)[0][n//4:n//2]
])

X_a = X_train[indices_a]
y_a = y_train[indices_a]
X_b = X_train[indices_b]
y_b = y_train[indices_b]

print(f"Client A: {X_a.shape} (class dist: {(y_a==0).sum()}/{(y_a==1).sum()})")
print(f"Client B: {X_b.shape} (class dist: {(y_b==0).sum()}/{(y_b==1).sum()})\n")

# Инициализируем модели
model_a = create_model_v2()
model_b = create_model_v2()
set_weights(model_b, get_weights(model_a))

# Сохраняем метрики
history = {
    'a_local': [], 'b_local': [],
    'a_global': [], 'b_global': [],
    'test_accuracy': []
}

# 8 раундов FL
for round_num in range(8):
    print(f"--- Round {round_num + 1} ---")
    
    # Обучение
    train_one_epoch(model_a, X_a, y_a)
    train_one_epoch(model_b, X_b, y_b)
    
    # Локальные точности
    acc_a_local = model_a.evaluate(X_a, y_a, verbose=0)[1]
    acc_b_local = model_b.evaluate(X_b, y_b, verbose=0)[1]
    
    # Усреднение весов (FedAvg)
    w_a = get_weights(model_a)
    w_b = get_weights(model_b)
    w_avg = average_weights(w_a, w_b)
    
    set_weights(model_a, w_avg)
    set_weights(model_b, w_avg)
    
    # Глобальные точности
    acc_a_global = model_a.evaluate(X_train, y_train, verbose=0)[1]
    acc_b_global = model_b.evaluate(X_train, y_train, verbose=0)[1]
    
    # Точность на тестовом наборе
    test_acc = model_a.evaluate(X_test, y_test, verbose=0)[1]
    
    history['a_local'].append(acc_a_local)
    history['b_local'].append(acc_b_local)
    history['a_global'].append(acc_a_global)
    history['b_global'].append(acc_b_global)
    history['test_accuracy'].append(test_acc)
    
    print(f"A: Local={acc_a_local:.3f} → Global={acc_a_global:.3f}")
    print(f"B: Local={acc_b_local:.3f} → Global={acc_b_global:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}\n")

# === ВИЗУАЛИЗАЦИЯ ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rounds = range(1, 9)

# График 1: Локальная vs Глобальная точность
axes[0].plot(rounds, history['a_local'], 'o-', label='A Local', linewidth=2)
axes[0].plot(rounds, history['a_global'], 's-', label='A Global', linewidth=2)
axes[0].plot(rounds, history['b_local'], '^--', label='B Local', linewidth=2)
axes[0].plot(rounds, history['b_global'], 'D--', label='B Global', linewidth=2)
axes[0].set_xlabel('FL Round', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Local vs Global Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.3, 1.05])

# График 2: Тестовая точность
axes[1].plot(rounds, history['test_accuracy'], 'go-', linewidth=3, markersize=8, label='Test Accuracy')
axes[1].fill_between(rounds, 0.5, history['test_accuracy'], alpha=0.2, color='green')
axes[1].set_xlabel('FL Round', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Test Set Accuracy (Unseen Data)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.3, 1.05])
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Guess')

plt.tight_layout()
plt.show()

print("=== FL Complete ===")
