import numpy as np
import pandas as pd
import json
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
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
tree_model = DecisionTreeRegressor(
    splitter='best',
    max_depth=10,
    min_samples_leaf=5,
    max_leaf_nodes=30
)
tree_model.fit(X_train, y_train)

y_pred_train = tree_model.predict(X_train)
y_pred_val   = tree_model.predict(X_val)
y_pred_test  = tree_model.predict(X_test)

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
# 4. Сохранение метрик в metrics/dt.json
# ─────────────────────────────────────────────
os.makedirs('metrics', exist_ok=True)
metrics = {
    'train': metrics_train,
    'val':   metrics_val,
    'test':  metrics_test
}
with open('metrics/dt.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Метрики сохранены: metrics/dt.json")

# ─────────────────────────────────────────────
# 5. Визуализация первых 3-х уровней дерева
# ─────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(36, 12))
fig.patch.set_facecolor('white')

plot_tree(
    tree_model,
    feature_names=X_train.columns.tolist(),
    filled=True,
    rounded=True,
    fontsize=20,
    max_depth=2,
    precision=3,
    ax=ax
)

ax.set_title(
    'Decision Tree Regressor (первые 3 уровня)',
    fontsize=22,
    fontweight='bold',
    pad=14
)

plt.savefig(
    'figures/dt_tree.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print("✅ Визуализация сохранена: figures/dt_tree.png")

# ─────────────────────────────────────────────
# 6. Сохранение модели в models/dt.pkl
# ─────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
with open('models/dt.pkl', 'wb') as f:
    pickle.dump(tree_model, f)
print("✅ Модель сохранена: models/dt.pkl")

# ─────────────────────────────────────────────
# 7. График реальных vs предсказанных значений
# ─────────────────────────────────────────────
target_column = 'selling_price'

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Предсказания модели")
plt.plot(y_test, y_test, color="red", linestyle="--", label="Идеальное предсказание")
plt.title(f"Реальные vs Предсказанные значения для '{target_column}' (Decision Tree)")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/dt_real_vs_pred.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/dt_real_vs_pred.png")

# ─────────────────────────────────────────────
# 8. Feature Importances
# ─────────────────────────────────────────────
importances = pd.Series(tree_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Decision Tree — Feature Importances', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Важность признака')
plt.tight_layout()
plt.savefig('figures/dt_feature_importances.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/dt_feature_importances.png")
