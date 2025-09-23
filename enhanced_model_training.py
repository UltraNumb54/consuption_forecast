import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# === КОНСТАНТЫ ===
FILTERED_DATA_FILE = 'enhanced_filtered_training_data.csv'
MODEL_PATH = 'enhanced_energy_model.cbm'
FEATURES_IMPORTANCE_PATH = 'enhanced_features_importance.png'

def mape_scorer(y_true, y_pred):
    """Кастомный scorer для MAPE"""
    return -mean_absolute_percentage_error(y_true, y_pred)  # GridSearch максимизирует, поэтому минус

def train_model_with_optimization():
    """Обучение модели с подбором гиперпараметров и прогресс-барами"""
    print("=== ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ С ПОДБОРОМ ГИПЕРПАРАМЕТРОВ ===")
    df = pd.read_csv(FILTERED_DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    print(f"Всего данных: {len(df)}")
    print(f"Период: {df['datetime'].min()} - {df['datetime'].max()}")

    # Определяем признаки
    base_features = [
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
    
    feature_columns = base_features + lag_cols + rolling_cols + ewm_cols

    # Проверяем, что все колонки существуют
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Предупреждение: следующие колонки отсутствуют в данных: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]

    X = df[feature_columns]
    y = df['consumption']

    # Определяем категориальные признаки
    categorical_features = [
        'hour', 'dayofweek', 'month', 'week_of_year',
        'is_holiday', 'is_working_weekend', 'is_regular_weekend', 'is_working_day',
        'is_weekend_or_holiday', 'is_weekend',
        'is_winter', 'is_spring', 'is_summer', 'is_autumn'
    ]
    
    # Фильтруем только существующие категориальные признаки
    categorical_features = [col for col in categorical_features if col in X.columns]

    # Настройка кросс-валидации для временных рядов
    tscv = TimeSeriesSplit(n_splits=5)

    # Сетка гиперпараметров (можно расширить)
    param_grid = {
        'iterations': [500],          # Для скорости оставим 500
        'learning_rate': [0.05, 0.1],
        'depth': [6, 8],
        'l2_leaf_reg': [3, 5],
        'random_seed': [42]
    }

    # Создаем модель
    model = CatBoostRegressor(
        loss_function='MAPE',
        cat_features=categorical_features,
        verbose=False
    )

    # Поиск лучших параметров с ПРОГРЕСС-БАРОМ (verbose=2)
    print("\n🔍 Начинаем подбор гиперпараметров...")
    print("Это может занять несколько минут. Пожалуйста, подождите...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(mape_scorer, greater_is_better=False),
        cv=tscv,
        n_jobs=-1,  # Использовать все ядра CPU
        verbose=2   # <-- ПРОГРЕСС-БАР! Показывает ход выполнения
    )

    grid_search.fit(X, y)

    print(f"\n🏆 Лучшие параметры: {grid_search.best_params_}")
    # Исправлено: -grid_search.best_score_ даст ПОЛОЖИТЕЛЬНЫЙ MAPE
    print(f"📈 Лучший MAPE на кросс-валидации: {-grid_search.best_score_:.4f} (или {(-grid_search.best_score_)*100:.2f}%)")

    # ===== ИСПРАВЛЕНИЕ ОШИБКИ use_best_model =====
    # Разделяем данные на финальные train/val (80/20) для обучения с early stopping
    split_idx = int(len(X) * 0.8)
    X_train_final = X.iloc[:split_idx]
    y_train_final = y.iloc[:split_idx]
    X_val_final = X.iloc[split_idx:]
    y_val_final = y.iloc[split_idx:]

    # Обучаем финальную модель на лучших параметрах с ПРОГРЕСС-БАРОМ
    print("\n🚀 Обучение финальной модели с лучшими параметрами...")
    best_model = CatBoostRegressor(
        **grid_search.best_params_,
        loss_function='MAPE',
        eval_metric='MAPE',  # Для красивого вывода в прогресс-баре
        cat_features=categorical_features,
        early_stopping_rounds=30,  # Предотвращение переобучения
        use_best_model=True,       # Использовать лучшую итерацию
        verbose=50                 # <-- ПРОГРЕСС-БАР! Вывод каждые 50 итераций
    )

    # ОБЯЗАТЕЛЬНО передаем eval_set!
    best_model.fit(
        X_train_final, y_train_final,
        eval_set=(X_val_final, y_val_final)  # <-- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ!
    )
    # =============================================

    # Оценка на всем датасете (для информации)
    y_pred_full = best_model.predict(X)
    mae_full = mean_absolute_error(y, y_pred_full)
    mape_full = mean_absolute_percentage_error(y, y_pred_full) * 100
    print(f"\n📊 Оценка на всем обучающем наборе:")
    print(f"   MAE: {mae_full:.3f}")
    print(f"   MAPE: {mape_full:.2f}%")

    # Анализ переобучения (сравниваем ошибку на train и val)
    y_pred_train = best_model.predict(X_train_final)
    y_pred_val = best_model.predict(X_val_final)
    mae_train = mean_absolute_error(y_train_final, y_pred_train)
    mae_val = mean_absolute_error(y_val_final, y_pred_val)
    overfitting_ratio = mae_val / mae_train if mae_train > 0 else float('inf')
    print(f"\n⚠️  Коэффициент переобучения (MAE val / MAE train): {overfitting_ratio:.2f}")

    # Вывод важности признаков
    print(f"\n=== ТОП-15 САМЫХ ВАЖНЫХ ПРИЗНАКОВ ===")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(15).iterrows():
        print(f"{i+1:2d}. {row['feature']:<30} : {row['importance']:>8.2f}")

    # Сохраняем модель
    best_model.save_model(MODEL_PATH)
    print(f"\n✅ Модель успешно сохранена в {MODEL_PATH}")

    # Сохраняем график важности признаков
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Важность')
        plt.title('Топ-15 самых важных признаков')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FEATURES_IMPORTANCE_PATH, dpi=300, bbox_inches='tight')
        print(f"📊 График важности признаков сохранен в {FEATURES_IMPORTANCE_PATH}")
    except Exception as e:
        print(f"⚠️  Не удалось сохранить график важности признаков: {e}")

    return best_model, feature_columns

if __name__ == "__main__":
    model, features = train_model_with_optimization()
    print("\n🎉 Обучение завершено успешно!")
