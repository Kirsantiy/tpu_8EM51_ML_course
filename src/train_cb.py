import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import catboost as cb
import optuna
import warnings
from optuna.integration import CatBoostPruningCallback
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

# ─────────────────────────────────────────────
# 1. Загрузка данных из data/
# ─────────────────────────────────────────────
X_train = pd.read_csv('data/X_train_cb.csv')
y_train = pd.read_csv('data/y_train_cb.csv').squeeze()

X_val   = pd.read_csv('data/X_val_cb.csv')
y_val   = pd.read_csv('data/y_val_cb.csv').squeeze()

X_test  = pd.read_csv('data/X_test_cb.csv')
y_test  = pd.read_csv('data/y_test_cb.csv').squeeze()

cat_features = ['fuel', 'seller_type', 'transmission']

train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
val_pool   = cb.Pool(X_val,   y_val,   cat_features=cat_features)
test_pool  = cb.Pool(X_test,  y_test,  cat_features=cat_features)

# ─────────────────────────────────────────────
# 2. Оптимизация гиперпараметров через Optuna + Pruning
# ─────────────────────────────────────────────
def trial_callback(study, trial):
    pruned  = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    status = "✂️ PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else "✅ OK"
    print(
        f"Trial {trial.number:>3} | {status:<12} | "
        f"RMSE: {trial.value:.5f} | "
        f"Лучший: {study.best_value:.5f} | "
        f"Завершено: {complete} | Обрезано: {pruned}"
    )

def objective(trial):
    params = {
        'iterations':            trial.suggest_int('iterations', 2000, 8000),
        'learning_rate':         trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'depth':                 trial.suggest_int('depth', 6, 11),
        'l2_leaf_reg':           trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'bootstrap_type':        'Bernoulli',
        'subsample':             trial.suggest_float('subsample', 0.2, 1),
        'loss_function':         'RMSE',
        'eval_metric':           'RMSE',
        'one_hot_max_size':       10,
        'task_type':             'CPU',
        'devices':               '0',
        'early_stopping_rounds': 200,
        'verbose':               False,
    }

    pruning_callback = CatBoostPruningCallback(trial, 'RMSE')

    model = cb.CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        callbacks=[pruning_callback],
        verbose=False
    )

    pruning_callback.check_pruned()

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

print("--- 2. Подбор гиперпараметров (Optuna, 200 trials, MedianPruner) ---")
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=200)
)
study.optimize(objective, n_trials=200, callbacks=[trial_callback])

best_params = study.best_params
pruned  = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"✅ Лучшие параметры: {best_params}")
print(f"   Завершено trials: {complete} | Обрезано (pruned): {pruned}")

# ─────────────────────────────────────────────
# 3. Обучение финальной модели с лучшими параметрами
# ─────────────────────────────────────────────
print("\n--- 3. Обучение финальной модели ---")
final_params = {
    **best_params,
    'bootstrap_type':        'Bernoulli',
    'loss_function':         'RMSE',
    'eval_metric':           'RMSE',
    'one_hot_max_size':       10,
    'task_type':             'GPU',
    'devices':               '0',
    'early_stopping_rounds': 200,
    'verbose':               100
}

catboost_model = cb.CatBoostRegressor(**final_params)
catboost_model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=100
)

y_pred_train = catboost_model.predict(X_train)
y_pred_val   = catboost_model.predict(X_val)
y_pred_test  = catboost_model.predict(X_test)

# ─────────────────────────────────────────────
# 4. Метрики
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
# 5. Сохранение метрик в metrics/cb.json
# ─────────────────────────────────────────────
os.makedirs('metrics', exist_ok=True)
metrics = {
    'best_params': best_params,
    'train': metrics_train,
    'val':   metrics_val,
    'test':  metrics_test
}
with open('metrics/cb.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Метрики сохранены: metrics/cb.json")

# ─────────────────────────────────────────────
# 6. Сохранение модели в models/cb.cbm
# ─────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
catboost_model.save_model('models/cb.cbm')
print("✅ Модель сохранена: models/cb.cbm")

# ─────────────────────────────────────────────
# 7. Графики Optuna — Hyperparameter Importances и Optimization History
# ─────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

# График 1 — Hyperparameter Importances
ax1 = plot_param_importances(study)
fig1 = ax1.get_figure()
fig1.patch.set_facecolor('white')
fig1.tight_layout()
fig1.savefig('figures/cb_param_importances.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("✅ График сохранён: figures/cb_param_importances.png")

# График 2 — Optimization History
ax2 = plot_optimization_history(study)
ax2.set_title('CatBoost — Optimization History', fontsize=14, fontweight='bold', pad=12)
fig2 = ax2.get_figure()
fig2.patch.set_facecolor('white')
fig2.tight_layout()
fig2.savefig('figures/cb_optimization_history.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("✅ График сохранён: figures/cb_optimization_history.png")

# ─────────────────────────────────────────────
# 8. График реальных vs предсказанных значений
# ─────────────────────────────────────────────
target_column = 'selling_price'

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Предсказания модели")
plt.plot(y_test, y_test, color="red", linestyle="--", label="Идеальное предсказание")
plt.title(f"Реальные vs Предсказанные значения для '{target_column}' (CatBoost)")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/cb_real_vs_pred.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/cb_real_vs_pred.png")

# ─────────────────────────────────────────────
# 9. Feature Importances
# ─────────────────────────────────────────────
importances = pd.Series(
    catboost_model.get_feature_importance(),
    index=X_train.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('CatBoost — Feature Importances', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Важность признака')
plt.tight_layout()
plt.savefig('figures/cb_feature_importances.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/cb_feature_importances.png")
