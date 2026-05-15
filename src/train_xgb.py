import numpy as np
import pandas as pd
import json
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

# ─────────────────────────────────────────────
# 1. Загрузка данных из data/
# ─────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

X_val   = pd.read_csv('data/X_val.csv')
y_val   = pd.read_csv('data/y_val.csv').squeeze()

X_test  = pd.read_csv('data/X_test.csv')
y_test  = pd.read_csv('data/y_test.csv').squeeze()

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# ─────────────────────────────────────────────
# 2. Оптимизация гиперпараметров через Optuna + Pruning
# ─────────────────────────────────────────────
def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':       trial.suggest_int('max_depth', 5, 11),
        'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample':       trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'tree_method':     'hist',
        'device':          'cuda',
        'random_state':    42,
        'objective':       'reg:squarederror',
        'eval_metric':     'rmse',
        'verbosity':       0
    }

    # Pruning callback — репортит RMSE на val после каждого раунда
    pruning_callback = XGBoostPruningCallback(trial, 'validation-rmse')

    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'validation')],
        callbacks=[pruning_callback],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    y_pred = xgb_model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def trial_callback(study, trial):
    pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    rmse_str = f"{trial.value:.5f}" if trial.value is not None else "    —   "
    status   = "✂️  PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else "✅ OK"
    print(
        f"Trial {trial.number:>3} | {status:<12} | "
        f"RMSE: {rmse_str} | "
        f"Лучший: {study.best_value:.5f} | "
        f"Завершено: {complete} | Обрезано: {pruned}"
    )

print("--- 2. Подбор гиперпараметров (Optuna, 200 trials, MedianPruner) ---")
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=100)
)
study.optimize(objective, n_trials=200, callbacks=[trial_callback])

best_params = study.best_params
pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"\n✅ Лучшие параметры: {best_params}")
print(f"   Завершено trials: {complete} | Обрезано (pruned): {pruned}")

# ─────────────────────────────────────────────
# 3. Обучение финальной модели с лучшими параметрами
# ─────────────────────────────────────────────
print("\n--- 3. Обучение финальной модели ---")
final_params = {
    **best_params,
    'tree_method':  'hist',
    'device':       'cuda',
    'random_state': 42,
    'objective':    'reg:squarederror',
    'eval_metric':  'rmse',
    'verbosity':    1
}

xgb_model = xgb.train(
    final_params,
    dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=100
)

y_pred_train = xgb_model.predict(dtrain)
y_pred_val   = xgb_model.predict(dval)
y_pred_test  = xgb_model.predict(dtest)

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
# 5. Сохранение метрик в metrics/xgb.json
# ─────────────────────────────────────────────
os.makedirs('metrics', exist_ok=True)
metrics = {
    'best_params': best_params,
    'train': metrics_train,
    'val':   metrics_val,
    'test':  metrics_test
}
with open('metrics/xgb.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✅ Метрики сохранены: metrics/xgb.json")

# ─────────────────────────────────────────────
# 6. Сохранение модели в models/xgb.json
# ─────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
xgb_model.save_model('models/xgb.json')
print("✅ Модель сохранена: models/xgb.json")

# ─────────────────────────────────────────────
# 7. Графики Optuna — Hyperparameter Importances и Optimization History
# ─────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)

# График 1 — Hyperparameter Importances
ax1 = plot_param_importances(study)
fig1 = ax1.get_figure()
fig1.patch.set_facecolor('white')
fig1.tight_layout()
fig1.savefig('figures/xgb_param_importances.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("✅ График сохранён: figures/xgb_param_importances.png")

# График 2 — Optimization History
ax2 = plot_optimization_history(study)
ax2.set_title('XGBoost — Optimization History', fontsize=14, fontweight='bold', pad=12)
fig2 = ax2.get_figure()
fig2.patch.set_facecolor('white')
fig2.tight_layout()
fig2.savefig('figures/xgb_optimization_history.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("✅ График сохранён: figures/xgb_optimization_history.png")

# ─────────────────────────────────────────────
# 8. График реальных vs предсказанных значений
# ─────────────────────────────────────────────
target_column = 'selling_price'

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Предсказания модели")
plt.plot(y_test, y_test, color="red", linestyle="--", label="Идеальное предсказание")
plt.title(f"Реальные vs Предсказанные значения для '{target_column}' (XGBoost)")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('figures/xgb_real_vs_pred.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/xgb_real_vs_pred.png")

# ─────────────────────────────────────────────
# 9. Feature Importances
# ─────────────────────────────────────────────
scores = xgb_model.get_score(importance_type='gain')
importances = pd.Series(scores).reindex(X_train.columns, fill_value=0).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('XGBoost — Feature Importances (gain)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Важность признака (gain)')
plt.tight_layout()
plt.savefig('figures/xgb_feature_importances.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ График сохранён: figures/xgb_feature_importances.png")
