# src/preprocess.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def preprocess_consumption(df):
    """
    Предобработка датасета потребления.
    """
    logging.info("Начало предобработки потребления.")
    # Преобразуем дату + час в datetime
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    # Удаляем строки с пропусками в ключевых полях
    df = df.dropna(subset=['consumption', 'temperature'])
    # Удаляем температурный прогноз, если он есть
    if 'temperature_forecast' in df.columns:
        df = df.drop(columns=['temperature_forecast'])
    # Создаем базовые признаки
    df['date_only'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    # Циклические признаки для часа
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # Признак времени года
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_winter'] = (df['season'] == 1).astype(int)
    df['is_spring'] = (df['season'] == 2).astype(int)
    df['is_summer'] = (df['season'] == 3).astype(int)
    df['is_autumn'] = (df['season'] == 4).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    logging.info("Предобработка потребления завершена.")
    return df

def preprocess_weather(df):
    """
    Предобработка датасета погоды.
    """
    logging.info("Начало предобработки погоды.")
    # Берем только влажность и ветер. Температура (T) не берётся.
    required_cols = ['datetime', 'U', 'Ff']
    available_cols = [col for col in required_cols if col in df.columns]
    if len(available_cols) < 2:
        raise ValueError(f"Не найдены необходимые колонки (U, Ff) в файле погоды. Найдено: {available_cols}")

    df_weather_subset = df[available_cols].copy()
    df_weather_subset = df_weather_subset.rename(columns={'U': 'humidity', 'Ff': 'wind_speed'})
    # Усредним по часам
    df_weather_subset['datetime'] = df_weather_subset['datetime'].dt.floor('H')
    df_weather_subset = df_weather_subset.groupby('datetime').mean().reset_index()
    logging.info("Предобработка погоды завершена.")
    return df_weather_subset

def create_lag_features(df, lag_hours=None):
    """Создание лаговых признаков"""
    if lag_hours is None:
        lag_hours = [1, 2, 3, 24, 48, 72, 120, 168]
    logging.info(f"Создание лаговых признаков для лагов: {lag_hours}")
    for lag in lag_hours:
        df[f'consumption_lag_{lag}'] = df['consumption'].shift(lag)
    return df

def create_rolling_features(df, windows=None):
    """Создание скользящих признаков (среднее и стандартное отклонение)"""
    if windows is None:
        windows = [3, 6, 12, 24, 720]
    logging.info(f"Создание скользящих признаков для окон: {windows}")
    for window in windows:
        df[f'consumption_rolling_mean_{window}'] = df['consumption'].rolling(window=window).mean()
        df[f'consumption_rolling_std_{window}'] = df['consumption'].rolling(window=window).std()
    return df

def create_ewm_features(df, spans=None):
    """Создание признаков экспоненциального сглаживания"""
    if spans is None:
        spans = [3, 6, 12, 24]
    logging.info(f"Создание EWM признаков для span: {spans}")
    for span in spans:
        df[f'consumption_ewm_mean_{span}'] = df['consumption'].ewm(span=span).mean()
    return df

def handle_missing_values(df):
    """Обработка пропусков в данных"""
    logging.info("Начало обработки пропусков.")
    # Колонки для обработки
    weather_cols = ['temperature', 'humidity', 'wind_speed']
    lag_cols = [col for col in df.columns if 'consumption_lag_' in col]
    rolling_cols = [col for col in df.columns if 'consumption_rolling_' in col]
    ewm_cols = [col for col in df.columns if 'consumption_ewm_' in col]

    # Интерполяция погодных данных
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby(df['datetime'].dt.date)[col].transform(
                lambda x: x.interpolate(method='linear', limit=3, limit_direction='both')
            )
            hourly_mean = df.groupby('hour')[col].transform('mean')
            df[col] = df[col].fillna(hourly_mean)

    # Интерполяция лаговых и скользящих признаков
    for col in lag_cols + rolling_cols + ewm_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    # Финальное заполнение медианой
    all_cols = weather_cols + lag_cols + rolling_cols + ewm_cols
    for col in all_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    logging.info("Обработка пропусков завершена.")
    return df

def load_calendar(df_calendar):
    """
    Загрузка и подготовка календаря.
    """
    logging.info("Начало подготовки календаря.")
    df_calendar['date'] = pd.to_datetime(df_calendar['date']).dt.date
    logging.info("Подготовка календаря завершена.")
    return df_calendar

def main_preprocess(df_consumption, df_weather, df_calendar):
    """
    Основная функция предобработки, объединяющая все шаги.
    """
    logging.info("=== НАЧАЛО ГЛАВНОЙ ПРЕДОБРАБОТКИ ===")
    # 1. Предобработка каждого датасета
    df_con = preprocess_consumption(df_consumption)
    df_w = preprocess_weather(df_weather)
    df_cal = load_calendar(df_calendar)

    # 2. Объединение
    logging.info("Объединение датасетов...")
    df_merged = pd.merge(df_con, df_w, on='datetime', how='left')
    df_merged['date_for_merge'] = pd.to_datetime(df_merged['date_only'])
    df_cal['date_for_merge'] = pd.to_datetime(df_cal['date'])
    df_final = pd.merge(df_merged, df_cal, left_on='date_for_merge', right_on='date_for_merge', how='left')

    # 3. Создание признаков из календаря
    df_final['is_holiday'] = (df_final['day_type'] == 'non-working holiday').astype(int)
    df_final['is_working_weekend'] = (df_final['day_type'] == 'working weekend').astype(int)
    df_final['is_regular_weekend'] = (df_final['day_type'] == 'weekend').astype(int)
    df_final['is_working_day'] = (df_final['day_type'] == 'working day').astype(int)
    df_final['is_weekend_or_holiday'] = (
        (df_final['day_type'] == 'weekend') |
        (df_final['day_type'] == 'non-working holiday')
    ).astype(int)

    df_final = df_final.sort_values('datetime')

    # 4. Создание сложных признаков
    df_final = create_lag_features(df_final)
    df_final = create_rolling_features(df_final)
    df_final = create_ewm_features(df_final)

    # 5. Обработка пропусков
    df_final = handle_missing_values(df_final)

    # 6. Финальная очистка
    columns_to_drop = ['date_for_merge_x', 'date_for_merge_y', 'T', 'date_only', 'date'] # Удаляем временные и исходные даты
    df_final = df_final.drop(columns=[col for col in columns_to_drop if col in df_final.columns])

    # Удаляем оставшиеся строки с пропусками в критических колонках
    critical_cols = ['consumption', 'temperature', 'humidity', 'wind_speed']
    df_final = df_final.dropna(subset=[col for col in critical_cols if col in df_final.columns])

    logging.info("=== ГЛАВНАЯ ПРЕДОБРАБОТКА ЗАВЕРШЕНА ===")
    return df_final

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Этот файл должен быть запущен из контекста pipeline_runner.py, передав ему данные
    # или загружать их сам, если используется отдельно.
    # Пример:
    # from data_ingestion import load_consumption_data, load_weather_data, load_calendar_data
    # df_con = load_consumption_data()
    # df_w = load_weather_data()
    # df_cal = load_calendar_data()
    # df_processed = main_preprocess(df_con, df_w, df_cal)
    # print(df_processed.head())
