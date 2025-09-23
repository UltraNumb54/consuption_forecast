import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# === КОНСТАНТЫ ===
MODEL_PATH = 'enhanced_energy_model.cbm'
HISTORICAL_DATA_FILE = 'enhanced_processed_energy_data.csv'  # Используем полный датасет
TEST_START_DATE = '2025-09-01'
TEST_DAYS = 10  # Тестируем на 10 дней для надежности

def load_model():
    """Загрузка обученной модели"""
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

def prepare_features_for_prediction(df_historical, prediction_datetime, weather_data, feature_columns):
    """Подготовка признаков для одного предсказания"""
    features = {}
    
    # Базовые признаки
    features['hour'] = prediction_datetime.hour
    features['hour_sin'] = np.sin(2 * np.pi * prediction_datetime.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * prediction_datetime.hour / 24)
    features['dayofweek'] = prediction_datetime.weekday()
    features['month'] = prediction_datetime.month
    features['week_of_year'] = prediction_datetime.isocalendar().week

    # Погодные данные
    features['temperature'] = weather_data.get('temperature', 10.0)
    features['humidity'] = weather_data.get('humidity', 70.0)
    features['wind_speed'] = weather_data.get('wind_speed', 3.0)

    # Календарные признаки
    date_only = prediction_datetime.date()
    calendar_data = df_historical[df_historical['date_only'] == str(date_only)]
    if len(calendar_data) > 0:
        row = calendar_data.iloc[0]
        features['is_holiday'] = row.get('is_holiday', 0)
        features['is_working_weekend'] = row.get('is_working_weekend', 0)
        features['is_regular_weekend'] = row.get('is_regular_weekend', 0)
        features['is_working_day'] = row.get('is_working_day', 0)
        features['is_weekend_or_holiday'] = row.get('is_weekend_or_holiday', 0)
        features['is_weekend'] = row.get('is_weekend', 0)
    else:
        is_weekend = int(prediction_datetime.weekday() in [5, 6])
        features.update({
            'is_holiday': 0,
            'is_working_weekend': 0,
            'is_regular_weekend': is_weekend,
            'is_working_day': 1 - is_weekend,
            'is_weekend_or_holiday': is_weekend,
            'is_weekend': is_weekend,
        })

    # Времена года
    month = prediction_datetime.month
    season = month % 12 // 3 + 1
    features['is_winter'] = int(season == 1)
    features['is_spring'] = int(season == 2)
    features['is_summer'] = int(season == 3)
    features['is_autumn'] = int(season == 4)

    # Лаговые признаки
    for lag in [1, 2, 3, 24, 48, 72, 120, 168]:
        lag_time = prediction_datetime - timedelta(hours=lag)
        lag_data = df_historical[df_historical['datetime'] == lag_time]
        if len(lag_data) > 0:
            features[f'consumption_lag_{lag}'] = lag_data['consumption'].iloc[0]
        else:
            features[f'consumption_lag_{lag}'] = np.nan

    # Скользящие и EWM признаки
    for prefix in ['consumption_rolling_mean_', 'consumption_rolling_std_', 'consumption_ewm_mean_']:
        for window in [3, 6, 12, 24, 720]:
            if f'{prefix}{window}' in feature_columns:
                if 'ewm' in prefix:
                    # Для EWM берем последние `window` значений и считаем вручную
                    recent_data = df_historical[df_historical['datetime'] < prediction_datetime].tail(window)
                    if len(recent_data) > 0:
                        ewm = recent_data['consumption'].ewm(span=window).mean().iloc[-1]
                        features[f'{prefix}{window}'] = ewm
                    else:
                        features[f'{prefix}{window}'] = np.nan
                else:
                    # Для обычных скользящих берем последние `window` значений
                    recent_data = df_historical[df_historical['datetime'] < prediction_datetime].tail(window)
                    if len(recent_data) >= window:
                        if 'mean' in prefix:
                            features[f'{prefix}{window}'] = recent_data['consumption'].mean()
                        else:
                            features[f'{prefix}{window}'] = recent_data['consumption'].std()
                    else:
                        features[f'{prefix}{window}'] = np.nan

    return features

def september_test():
    """Тестирование на сентябре 2025"""
    print("=== ТЕСТИРОВАНИЕ НА СЕНТЯБРЕ 2025 ===")
    model = load_model()
    print("Модель загружена")

    df_historical = pd.read_csv(HISTORICAL_DATA_FILE)
    df_historical['datetime'] = pd.to_datetime(df_historical['datetime'])
    df_historical['date_only'] = df_historical['datetime'].dt.date.astype(str)

    test_start = datetime.strptime(TEST_START_DATE, '%Y-%m-%d')
    test_end = test_start + timedelta(days=TEST_DAYS)
    print(f"Тестирование с {test_start} по {test_end}")

    training_data = df_historical[df_historical['datetime'] < test_start]
    test_data = df_historical[(df_historical['datetime'] >= test_start) & (df_historical['datetime'] < test_end)]

    print(f"Обучающие данные: {len(training_data)} записей")
    print(f"Тестовые данные: {len(test_data)} записей")

    if len(test_data) == 0:
        print("Нет тестовых данных!")
        return

    # Получаем список признаков из модели (или можно загрузить из training скрипта)
    sample_features = prepare_features_for_prediction(training_data.iloc[:1], test_start, {}, [])
    feature_columns = list(sample_features.keys())

    predictions = []
    actual_values = []
    prediction_times = []

    for idx, row in test_data.iterrows():
        prediction_time = row['datetime']
        actual_consumption = row['consumption']

        weather_data = {
            'temperature': row.get('temperature', 10.0),
            'humidity': row.get('humidity', 70.0),
            'wind_speed': row.get('wind_speed', 3.0),
        }

        features = prepare_features_for_prediction(training_data, prediction_time, weather_data, feature_columns)
        features_df = pd.DataFrame([features])

        # Заполнение пропусков
        for col in features_df.columns:
            if features_df[col].isna().any():
                if col in training_data.columns:
                    median_val = training_data[col].median()
                    features_df[col] = features_df[col].fillna(median_val)
                else:
                    features_df[col] = features_df[col].fillna(0)

        try:
            prediction = model.predict(features_df)[0]
            predictions.append(prediction)
            actual_values.append(actual_consumption)
            prediction_times.append(prediction_time)
            print(f"{prediction_time}: предсказано {prediction:.1f}, реально {actual_consumption:.1f}")
        except Exception as e:
            print(f"Ошибка при предсказании {prediction_time}: {e}")
            continue

    if len(predictions) > 0:
        mae = mean_absolute_error(actual_values, predictions)
        mape = mean_absolute_percentage_error(actual_values, predictions) * 100
        mean_actual = np.mean(actual_values)

        print(f"\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
        print(f"Количество предсказаний: {len(predictions)}")
        print(f"MAE: {mae:.3f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Среднее потребление: {mean_actual:.1f} МВт")
        print(f"Точность (±2.5% от среднего): {mae < (mean_actual * 0.025)}")

        # Визуализация
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(prediction_times, actual_values, 'b-', label='Реальное', linewidth=2)
        plt.plot(prediction_times, predictions, 'r--', label='Предсказанное', linewidth=2)
        plt.title('Прогноз потребления на сентябрь 2025')
        plt.ylabel('Потребление (МВт)')
        plt.legend()
        plt.grid(True, alpha=00.3)

        plt.subplot(2, 1, 2)
        errors = np.abs(np.array(actual_values) - np.array(predictions))
        plt.plot(prediction_times, errors, 'g-', alpha=0.7)
        plt.axhline(y=mean_actual * 0.025, color='r', linestyle='--', label='Порог ±2.5%')
        plt.title(f'Абсолютные ошибки (MAE = {mae:.3f})')
        plt.ylabel('Ошибка (МВт)')
        plt.xlabel('Время')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('enhanced_september_test.png', dpi=300, bbox_inches='tight')
        plt.show()

        results_df = pd.DataFrame({
            'datetime': prediction_times,
            'actual_consumption': actual_values,
            'predicted_consumption': predictions,
            'absolute_error': errors
        })
        results_df.to_csv('enhanced_september_test_detailed.csv', index=False)
        print("Детальные результаты сохранены в enhanced_september_test_detailed.csv")

if __name__ == "__main__":
    september_test()
