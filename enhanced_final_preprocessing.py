import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === КОНСТАНТЫ ===
PROCESSED_DATA_FILE = 'enhanced_processed_energy_data.csv'  # ИЗМЕНЕНО
FILTERED_DATA_FILE = 'enhanced_filtered_training_data.csv'  # ИЗМЕНЕНО

def load_and_filter_quality_data():
    """Загрузка и фильтрация данных по качеству"""
    print("=== ФИЛЬТРАЦИЯ ДАННЫХ ПО КАЧЕСТВУ ===")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Исходные данные: {len(df)} записей")

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    # Меняйте этот список для экспериментов
    problematic_years = [2017]  # Например, оставляем 2023, убираем только 2017
    df_filtered = df[~df['year'].isin(problematic_years)]

    df_filtered = df_filtered[
        ~((df_filtered['year'] == 2025) & (df_filtered['month'] == 9))
    ]
    print(f"После фильтрации проблемных лет: {len(df_filtered)} записей")

    df_filtered['year_month'] = df_filtered['datetime'].dt.to_period('M')
    monthly_stats = df_filtered.groupby('year_month').agg({
        'consumption': 'count',
        'temperature': lambda x: x.notna().sum(),
        'humidity': lambda x: x.notna().sum(),
        'wind_speed': lambda x: x.notna().sum()
    }).rename(columns={'consumption': 'total_records'})
    monthly_stats['temp_completeness'] = monthly_stats['temperature'] / monthly_stats['total_records']
    monthly_stats['humidity_completeness'] = monthly_stats['humidity'] / monthly_stats['total_records']
    monthly_stats['wind_completeness'] = monthly_stats['wind_speed'] / monthly_stats['total_records']
    
    # Фильтр: все погодные параметры должны быть заполнены > 80%
    good_months = monthly_stats[
        (monthly_stats['temp_completeness'] > 0.8) &
        (monthly_stats['humidity_completeness'] > 0.8) &
        (monthly_stats['wind_completeness'] > 0.8)
    ].index

    df_filtered = df_filtered[df_filtered['year_month'].isin(good_months)]
    print(f"После фильтрации по полноте: {len(df_filtered)} записей")
    return df_filtered

def handle_missing_values(df):
    """Обработка пропусков в данных"""
    print("=== ОБРАБОТКА ПРОПУСКОВ ===")
    missing_stats = df.isnull().sum()
    for col, count in missing_stats[missing_stats > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

    # Колонки для обработки
    weather_cols = ['humidity', 'wind_speed']  # Температура уже чистая
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

    print("Пропуски после обработки:")
    missing_stats = df.isnull().sum()
    for col, count in missing_stats[missing_stats > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

    return df

def create_final_dataset():
    """Создание финального датасета для обучения"""
    print("=== СОЗДАНИЕ ФИНАЛЬНОГО ДАТАСЕТА ===")
    df = load_and_filter_quality_data()
    df = handle_missing_values(df)

    # Удаляем оставшиеся строки с пропусками в критических колонках
    critical_cols = ['consumption', 'temperature', 'humidity', 'wind_speed']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])

    print(f"Финальный датасет: {len(df)} записей")
    df.to_csv(FILTERED_DATA_FILE, index=False)
    print(f"Финальный датасет сохранен в {FILTERED_DATA_FILE}")

    print(f"Период данных: {df['datetime'].min()} - {df['datetime'].max()}")
    print(f"Количество лет: {df['year'].nunique()}")
    print("Используемые годы:", sorted(df['year'].unique().tolist()))

    return df

if __name__ == "__main__":
    df_final = create_final_dataset()
