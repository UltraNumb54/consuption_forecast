import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === КОНСТАНТЫ ===
CONSUMPTION_FILE = 'consumption_data.csv'
WEATHER_FILE = 'weather_data.csv'
CALENDAR_FILE = 'russian_production_calendar_2017_2025.csv'
OUTPUT_FILE = 'enhanced_processed_energy_data.csv'

def preprocess_consumption_data():
    """Предобработка данных по потреблению"""
    print("Загрузка данных по потреблению...")
    try:
        df_consumption = pd.read_csv(CONSUMPTION_FILE, sep=';', parse_dates=['date'])
    except FileNotFoundError:
        print(f"Ошибка: Файл {CONSUMPTION_FILE} не найден!")
        return None

    # Преобразуем дату + час в datetime
    df_consumption['datetime'] = pd.to_datetime(df_consumption['date']) + pd.to_timedelta(df_consumption['hour'], unit='h')
    
    # Удаляем строки с пропусками в ключевых полях
    df_consumption = df_consumption.dropna(subset=['consumption', 'temperature'])
    
    # Удаляем температурный прогноз
    if 'temperature_forecast' in df_consumption.columns:
        df_consumption = df_consumption.drop(columns=['temperature_forecast'])
    
    # Создаем базовые признаки
    df_consumption['date_only'] = df_consumption['datetime'].dt.date
    df_consumption['hour'] = df_consumption['datetime'].dt.hour
    df_consumption['dayofweek'] = df_consumption['datetime'].dt.dayofweek
    df_consumption['month'] = df_consumption['datetime'].dt.month
    df_consumption['year'] = df_consumption['datetime'].dt.year
    df_consumption['week_of_year'] = df_consumption['datetime'].dt.isocalendar().week
    
    # Циклические признаки для часа
    df_consumption['hour_sin'] = np.sin(2 * np.pi * df_consumption['hour'] / 24)
    df_consumption['hour_cos'] = np.cos(2 * np.pi * df_consumption['hour'] / 24)
    
    # Признак времени года
    df_consumption['season'] = df_consumption['month'] % 12 // 3 + 1
    df_consumption['is_winter'] = (df_consumption['season'] == 1).astype(int)
    df_consumption['is_spring'] = (df_consumption['season'] == 2).astype(int)
    df_consumption['is_summer'] = (df_consumption['season'] == 3).astype(int)
    df_consumption['is_autumn'] = (df_consumption['season'] == 4).astype(int)
    
    # Исправлено: astype(int) на булевом Series - это нормально, игнорируем предупреждение IDE
    df_consumption['is_weekend'] = df_consumption['dayofweek'].isin([5, 6]).astype(int)

    print(f"Данные по потреблению обработаны. Размер: {len(df_consumption)}")
    return df_consumption

def preprocess_weather_data():
    """Предобработка данных по погоде (без температуры)"""
    print("Загрузка данных по погоде...")
    try:
        with open(WEATHER_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('"Местное время'):
                header_line = line.strip()
                data_start = i + 1
                break

        if header_line is None:
            print("Ошибка: Не найдена строка с заголовком в файле погоды")
            return None

        df_weather = pd.read_csv(
            WEATHER_FILE, 
            sep=';', 
            skiprows=data_start,
            on_bad_lines='skip',
            encoding='utf-8'
        )

        if len(df_weather) == 0:
            print("Ошибка: Файл погоды пуст!")
            return None

    except FileNotFoundError:
        print(f"Ошибка: Файл {WEATHER_FILE} не найден!")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке файла погоды: {e}")
        return None

    if len(df_weather.columns) >= 13:
        column_names = [
            'datetime', 'T', 'P0', 'P', 'U', 'DD', 'Ff', 'ff10', 
            'WW', 'WW2', 'clouds', 'VV', 'Td'
        ]
        actual_columns = df_weather.columns[:13]
        df_weather = df_weather[actual_columns]
        df_weather.columns = column_names[:len(actual_columns)]
    else:
        expected_names = ['datetime', 'T', 'P0', 'P', 'U', 'DD', 'Ff', 'ff10', 'WW', 'WW2', 'clouds', 'VV', 'Td']
        for i, col in enumerate(df_weather.columns[:len(expected_names)]):
            df_weather = df_weather.rename(columns={col: expected_names[i]})

    try:
        df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], dayfirst=True)
    except:
        try:
            df_weather['datetime'] = pd.to_datetime(df_weather.iloc[:, 0], dayfirst=True)
        except Exception as e:
            print(f"Ошибка при преобразовании даты: {e}")
            return None

    # Берем только влажность и ветер. Температура (T) не берётся.
    required_columns = ['datetime', 'U', 'Ff']
    existing_columns = [col for col in required_columns if col in df_weather.columns]
    if len(existing_columns) < 2:
        print("Ошибка: Не найдены необходимые колонки (U, Ff) в файле погоды")
        return None

    df_weather = df_weather[existing_columns]
    df_weather = df_weather.rename(columns={'U': 'humidity', 'Ff': 'wind_speed'})

    # Усредним по часам
    df_weather['datetime'] = df_weather['datetime'].dt.floor('H')
    df_weather = df_weather.groupby('datetime').mean().reset_index()

    print(f"Данные по погоде обработаны. Размер: {len(df_weather)}")
    return df_weather

def load_calendar():
    """Загрузка существующего производственного календаря"""
    print("Загрузка производственного календаря...")
    try:
        df_calendar = pd.read_csv(CALENDAR_FILE)
        df_calendar['date'] = pd.to_datetime(df_calendar['date']).dt.date
        return df_calendar
    except FileNotFoundError:
        print(f"Ошибка: Файл {CALENDAR_FILE} не найден!")
        return None

# Замена изменяемых значений по умолчанию (list) на None
def create_lag_features(df, lag_hours=None):
    """Создание лаговых признаков"""
    if lag_hours is None:
        lag_hours = [1, 2, 3, 24, 48, 72, 120, 168]  # Создается НОВЫЙ список при каждом вызове
    
    print("Создание лаговых признаков...")
    for lag in lag_hours:
        df[f'consumption_lag_{lag}'] = df['consumption'].shift(lag)
    return df

# Замена изменяемых значений по умолчанию (list) на None
def create_rolling_features(df, windows=None):
    """Создание скользящих признаков (среднее и стандартное отклонение)"""
    if windows is None:
        windows = [3, 6, 12, 24, 720]  # Создается НОВЫЙ список при каждом вызове
    
    print("Создание скользящих признаков...")
    for window in windows:
        df[f'consumption_rolling_mean_{window}'] = df['consumption'].rolling(window=window).mean()
        df[f'consumption_rolling_std_{window}'] = df['consumption'].rolling(window=window).std()
    return df

# Замена изменяемых значений по умолчанию (list) на None
def create_ewm_features(df, spans=None):
    """Создание признаков экспоненциального сглаживания"""
    if spans is None:
        spans = [3, 6, 12, 24]  # Создается НОВЫЙ список при каждом вызове
    
    print("Создание признаков экспоненциального сглаживания...")
    for span in spans:
        df[f'consumption_ewm_mean_{span}'] = df['consumption'].ewm(span=span).mean()
    return df

def main():
    """Основная функция предобработки"""
    print("=== НАЧАЛО УЛУЧШЕННОЙ ПРЕДОБРАБОТКИ ДАННЫХ ===")

    df_consumption = preprocess_consumption_data()
    if df_consumption is None:
        return

    df_weather = preprocess_weather_data()
    if df_weather is None:
        return

    df_calendar = load_calendar()
    if df_calendar is None:
        return

    print("Объединение датасетов...")
    # Объединяем с погодой (только влажность и ветер)
    df_merged = pd.merge(df_consumption, df_weather, on='datetime', how='left')

    df_merged['date_for_merge'] = pd.to_datetime(df_merged['date_only'])
    df_calendar['date_for_merge'] = pd.to_datetime(df_calendar['date'])
    df_final = pd.merge(df_merged, df_calendar, left_on='date_for_merge', right_on='date_for_merge', how='left')

    df_final['is_holiday'] = (df_final['day_type'] == 'non-working holiday').astype(int)
    df_final['is_working_weekend'] = (df_final['day_type'] == 'working weekend').astype(int)
    df_final['is_regular_weekend'] = (df_final['day_type'] == 'weekend').astype(int)
    df_final['is_working_day'] = (df_final['day_type'] == 'working day').astype(int)
    df_final['is_weekend_or_holiday'] = (
        (df_final['day_type'] == 'weekend') | 
        (df_final['day_type'] == 'non-working holiday')
    ).astype(int)

    df_final = df_final.sort_values('datetime')
    df_final = create_lag_features(df_final)
    df_final = create_rolling_features(df_final)
    df_final = create_ewm_features(df_final)

    initial_size = len(df_final)
    df_final = df_final.dropna()
    final_size = len(df_final)
    print(f"Удалено строк с пропусками: {initial_size - final_size}")

    columns_to_drop = ['date_for_merge_x', 'date_for_merge_y', 'T']
    df_final = df_final.drop(columns=[col for col in columns_to_drop if col in df_final.columns])

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"=== УЛУЧШЕННАЯ ПРЕДОБРАБОТКА ЗАВЕРШЕНА ===")
    print(f"Финальный датасет сохранен в {OUTPUT_FILE}")
    print(f"Размер: {len(df_final)} строк, {len(df_final.columns)} колонок")

if __name__ == "__main__":
    main()
