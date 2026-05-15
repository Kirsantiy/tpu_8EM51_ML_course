import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# ─────────────────────────────────────────────
# 1. Загрузка данных из data/
# ─────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv').astype(np.float32).values
y_train = pd.read_csv('data/y_train.csv').squeeze().astype(np.float32).values

X_val   = pd.read_csv('data/X_val.csv').astype(np.float32).values
y_val   = pd.read_csv('data/y_val.csv').squeeze().astype(np.float32).values

X_test  = pd.read_csv('data/X_test.csv').astype(np.float32).values
y_test  = pd.read_csv('data/y_test.csv').squeeze().astype(np.float32).values

n_features = X_train.shape[1]
print(f"Признаков в X: {n_features}")

# ─────────────────────────────────────────────
# 2. Архитектура сети
# ─────────────────────────────────────────────
print("\n--- 2. Построение архитектуры ---")
model = Sequential([
    # Входной слой
    Dense(16, activation='relu', input_shape=(n_features,)),

    # Скрытые слои
    Dense(32, activation='relu'),
    Dense(16,  activation='relu'),
    Dense(8,  activation='relu'),

    # Выходной слой (регрессия — без активации)
    Dense(1, activation=None)
])

model.compile(
    optimizer=Adam(),
    loss=MeanSquaredError(),
    metrics=['RootMeanSquaredError']
)

model.summary()

# ─────────────────────────────────────────────
# 3. Callbacks
# ─────────────────────────────────────────────
# TensorBoard логи — запусти после обучения: tensorboard --logdir logs/nn
os.makedirs('logs/nn', exist_ok=True)
tensorboard_callback = TensorBoard(
    log_dir='logs/nn',
    histogram_freq=1,      # гистограммы весов каждую эпоху
    write_graph=True,      # граф вычислений модели
    write_images=False
)

callbacks = [
    # Останавливает обучение если val_loss не улучшается 20 эпох подряд.
    # restore_best_weights=True — возвращает веса лучшей эпохи после остановки
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Уменьшает learning_rate в factor раз если val_loss не улучшается patience эпох.
    # Помогает "дотянуться" до минимума когда крупные шаги уже не работают
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    tensorboard_callback
]

# ─────────────────────────────────────────────
# 4. Обучение
# ─────────────────────────────────────────────
print("\n--- 3. Обучение модели ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 5. Метрики
# ─────────────────────────────────────────────
def print_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"--- Метрики для {name} ---")
    print(f"R^2 Score (Точность): {r2:.5f}")
    print(f"MAE (Средняя ошибка): {mae:.5f}")
    print(f"RMSE (Кв. ошибка): {rmse:.5f}\n")
    return {'r2': round(r2, 5), 'mae': round(mae, 5), 'rmse': round(rmse, 5)}

y_pred_train = model.predict(X_train, verbose=0).flatten()
y_pred_val   = model.predict(X_val,   verbose=0).flatten()
y_pred_test  = model.predict(X_test,  verbose=0).flatten()

print("\n--- 4. Итоговые метрики ---")
metrics_train = print_metrics(y_train, y_pred_train, "Train")
metrics_val   = print_metrics(y_val,   y_pred_val,   "Val")
metrics_test  = print_metrics(y_test,  y_pred_test,  "Test")

# ─────────────────────────────────────────────
# 6. Сохранение метрик в metrics/nn.json
# ─────────────────────────────────────────────
os.makedirs('metrics', exist_ok=True)
metrics = {
    'train': metrics_train,
    'val':   metrics_val,
    'test':  metrics_test
}
with open('metrics/nn.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Метрики сохранены: metrics/nn.json")

# ─────────────────────────────────────────────
# 7. График Loss и RMSE по эпохам
# ─────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Loss
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss (MSE) по эпохам', fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RMSE
#print(history.history.keys())
axes[1].plot(history.history['RootMeanSquaredError'],     label='Train RMSE')
axes[1].plot(history.history['val_RootMeanSquaredError'], label='Val RMSE')
axes[1].set_title('RMSE по эпохам', fontsize=14, fontweight='bold', pad=12)
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('RMSE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/nn_training_history.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/nn_training_history.png")

# ─────────────────────────────────────────────
# 8. Сохранение модели в models/nn.keras
# ─────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
model.save('models/nn.keras')
print("✅ Модель сохранена: models/nn.keras")

# ─────────────────────────────────────────────
# 9. График реальных vs предсказанных значений
# ─────────────────────────────────────────────
target_column = 'selling_price'

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Предсказания модели")
plt.plot(y_test, y_test, color="red", linestyle="--", label="Идеальное предсказание")
plt.title(f"Реальные vs Предсказанные значения для '{target_column}' (Neural Network)")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/nn_real_vs_pred.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/nn_real_vs_pred.png")

# ─────────────────────────────────────────────
# 10. Гистограммы весов по слоям + интерпретация
# ─────────────────────────────────────────────
trainable_layers = [(layer.name, layer.get_weights())
                    for layer in model.layers
                    if layer.get_weights()]

n_layers = len(trainable_layers)
fig, axes = plt.subplots(n_layers, 2, figsize=(14, 4 * n_layers))
fig.patch.set_facecolor('white')
fig.suptitle('Гистограммы весов и биасов по слоям', fontsize=15, fontweight='bold', y=1.01)

interpretations = []

for i, (name, weights) in enumerate(trainable_layers):
    W, b = weights[0], weights[1]

    # Веса
    axes[i, 0].hist(W.flatten(), bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    axes[i, 0].set_title(f'{name} — Веса (W)', fontsize=12, fontweight='bold')
    axes[i, 0].set_xlabel('Значение веса')
    axes[i, 0].set_ylabel('Частота')
    axes[i, 0].grid(True, alpha=0.3)

    # Биасы
    axes[i, 1].hist(b.flatten(), bins=30, color='coral', alpha=0.8, edgecolor='white')
    axes[i, 1].set_title(f'{name} — Биасы (b)', fontsize=12, fontweight='bold')
    axes[i, 1].set_xlabel('Значение биаса')
    axes[i, 1].set_ylabel('Частота')
    axes[i, 1].grid(True, alpha=0.3)

    # Интерпретация
    w_std  = W.std()
    w_mean = W.mean()
    dead   = (W == 0).sum()
    total  = W.size
    interp = (
        f"  [{name}] mean={w_mean:.4f}, std={w_std:.4f}, "
        f"нулевых весов: {dead}/{total}"
    )
    if w_std < 0.01:
        interp += " ⚠️  веса очень малы — слой почти не обучился"
    elif w_std > 2.0:
        interp += " ⚠️  веса очень велики — возможен взрыв градиентов"
    else:
        interp += " ✅ распределение в норме"
    interpretations.append(interp)

plt.tight_layout()
plt.savefig('figures/nn_weight_histograms.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/nn_weight_histograms.png")

print("\n--- Интерпретация весов ---")
for line in interpretations:
    print(line)

# ─────────────────────────────────────────────
# 11. Подсказка по TensorBoard
# ─────────────────────────────────────────────
print("\n--- TensorBoard ---")
print("Логи сохранены в: logs/nn/")
print("Для просмотра запусти в терминале:")
print("  tensorboard --logdir logs/nn")
print("Затем открой в браузере: http://localhost:6006")
print("Вкладки: Scalars (loss/rmse), Histograms (веса), Graphs (архитектура)")
