import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# === КОНСТАНТЫ ===
FILTERED_DATA_FILE = 'enhanced_filtered_training_data.csv'
MODEL_PATH = 'enhanced_energy_model.cbm'

def mape_scorer(y_true, y_pred):
    """Кастомный scorer для MAPE"""
    return -mean_absolute_percentage_error(y_true, y_pred)  # GridSearch максимизирует, поэтому минус

def train_model_with_optimization():
    """Обучение модели с подбором гиперпараметров"""
    print("=== ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ ===")
    df = pd.read_csv(FILTERED_DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Определяем признаки
    feature_columns = [
        'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'month', 'week_of_year',
        'temperature', 'humidity', 'wind_speed',
        'is_holiday', 'is_working_weekend', 'is_regular_weekend', 'is_working_day',
        'is_weekend_or_holiday', 'is_weekend',
        'is_winter', 'is_spring', 'is_summer', 'is_autumn'
    ]
    
    # Добавляем все лаги, скользящие и EWM признаки
    lag_cols = [col for col in df.columns if 'consumption_lag_' in col]
    rolling_cols = [col for col in df.columns if 'consumption_rolling_' in col]
    ewm_cols = [col for col in df.columns if 'consumption_ewm_' in col]
    
    feature_columns += lag_cols + rolling_cols + ewm_cols

    X = df[feature_columns]
    y = df['consumption']

    # Определяем категориальные признаки
    categorical_features = [
        'hour', 'dayofweek', 'month', 'week_of_year',
        'is_holiday', 'is_working_weekend', 'is_regular_weekend', 'is_working_day',
        'is_weekend_or_holiday', 'is_weekend',
        'is_winter', 'is_spring', 'is_summer', 'is_autumn'
    ]

    # Настройка кросс-валидации для временных рядов
    tscv = TimeSeriesSplit(n_splits=5)

    # Сетка гиперпараметров (можно расширить)
    param_grid = {
        'iterations': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'depth': [6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'random_seed': [42]
    }

    # Создаем модель
    model = CatBoostRegressor(
        loss_function='MAPE',
        cat_features=categorical_features,
        verbose=False
    )

    # Поиск лучших параметров
    print("Начинаем подбор гиперпараметров...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(mape_scorer, greater_is_better=False),
        cv=tscv,
        n_jobs=-1,  # Использовать все ядра CPU
        verbose=1
    )

    grid_search.fit(X, y)

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший MAPE на кросс-валидации: {-grid_search.best_score_:.4f}")

    # Обучаем финальную модель на лучших параметрах
    best_model = grid_search.best_estimator_
    best_model.fit(X, y, verbose=False)

    # Сохраняем модель
    best_model.save_model(MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

    return best_model, feature_columns

if __name__ == "__main__":
    model, features = train_model_with_optimization()
