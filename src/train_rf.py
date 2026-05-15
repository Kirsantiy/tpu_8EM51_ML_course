import numpy as np
import pandas as pd
import json
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. Загрузка данных из data/
# ─────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

X_val   = pd.read_csv('data/X_val.csv')
y_val   = pd.read_csv('data/y_val.csv').squeeze()

X_test  = pd.read_csv('data/X_test.csv')
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# ─────────────────────────────────────────────
# 2. Обучение модели
# ─────────────────────────────────────────────
random_forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
    max_leaf_nodes=150
)
random_forest.fit(X_train, y_train)

y_pred_train = random_forest.predict(X_train)
y_pred_val   = random_forest.predict(X_val)
y_pred_test  = random_forest.predict(X_test)

# ─────────────────────────────────────────────
# 3. Метрики
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

metrics_train = print_metrics(y_train, y_pred_train, "Train")
metrics_val   = print_metrics(y_val,   y_pred_val,   "Val")
metrics_test  = print_metrics(y_test,  y_pred_test,  "Test")

# ─────────────────────────────────────────────
# 4. Сохранение метрик в metrics/rf.json
# ─────────────────────────────────────────────
os.makedirs('metrics', exist_ok=True)
metrics = {
    'train': metrics_train,
    'val':   metrics_val,
    'test':  metrics_test
}
with open('metrics/rf.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Метрики сохранены: metrics/rf.json")

# ─────────────────────────────────────────────
# 5. Сохранение модели в models/rf.pkl
# ─────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
with open('models/rf.pkl', 'wb') as f:
    pickle.dump(random_forest, f)
print("✅ Модель сохранена: models/rf.pkl")

# ─────────────────────────────────────────────
# 6. График реальных vs предсказанных значений
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt
target_column = 'selling_price'

os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Предсказания модели")
plt.plot(y_test, y_test, color="red", linestyle="--", label="Идеальное предсказание")
plt.title(f"Реальные vs Предсказанные значения для '{target_column}' (Random Forest)")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/rf_real_vs_pred.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/rf_real_vs_pred.png")

# ─────────────────────────────────────────────
# 7. Feature Importances
# ─────────────────────────────────────────────
importances = pd.Series(random_forest.feature_importances_, index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Random Forest — Feature Importances', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Важность признака')
plt.tight_layout()
plt.savefig('figures/rf_feature_importances.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/rf_feature_importances.png")
