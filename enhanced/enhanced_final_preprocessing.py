import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === КОНСТАНТЫ ===
PROCESSED_DATA_FILE = 'enhanced_processed_energy_data.csv'  # Исходный предобработанный файл
FILTERED_TRAINING_FILE = 'enhanced_filtered_training_data.csv'  # Файл ДЛЯ ОБУЧЕНИЯ (без сентября 2025)
FILTERED_FULL_FILE = 'enhanced_filtered_full_data.csv'          # Файл ДЛЯ ТЕСТА (с сентябрем 2025)

def load_and_filter_quality_data(include_september=False):
    """Загрузка и фильтрация данных по качеству"""
    print("=== ФИЛЬТРАЦИЯ ДАННЫХ ПО КАЧЕСТВУ ===")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Исходные данные: {len(df)} записей")

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    # Меняйте этот список для экспериментов. Например, [2017] или [2017, 2023]
    problematic_years = [2017]
    df_filtered = df[~df['year'].isin(problematic_years)]
    print(f"После фильтрации проблемных лет: {len(df_filtered)} записей")

    # Исключаем сентябрь 2025 ТОЛЬКО если include_september=False
    if not include_september:
        df_filtered = df_filtered[
            ~((df_filtered['year'] == 2025) & (df_filtered['month'] == 9))
        ]
        print(f"После исключения сентября 2025: {len(df_filtered)} записей")

    # Фильтр по полноте данных (должно быть > 80% данных по погоде)
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

    print("Пропуски после обработки:")
    missing_stats = df.isnull().sum()
    for col, count in missing_stats[missing_stats > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

    return df

def create_final_dataset():
    """Создание финального датасета для обучения И полного датасета для тестирования"""
    print("=== СОЗДАНИЕ ФИНАЛЬНЫХ ДАТАСЕТОВ ===")
    
    # Загружаем ПОЛНЫЙ набор качественных данных (включая сентябрь 2025)
    df_full = load_and_filter_quality_data(include_september=True)
    df_full = handle_missing_values(df_full)
    
    # Удаляем оставшиеся строки с пропусками в критических колонках
    critical_cols = ['consumption', 'temperature', 'humidity', 'wind_speed']
    df_full = df_full.dropna(subset=[col for col in critical_cols if col in df_full.columns])
    
    print(f"Полный отфильтрованный датасет: {len(df_full)} записей")
    
    # Создаем датасет для обучения (исключаем сентябрь 2025)
    df_for_training = df_full[~((df_full['year'] == 2025) & (df_full['month'] == 9))]
    print(f"Датасет для обучения (без сентября 2025): {len(df_for_training)} записей")
    
    # Сохраняем оба файла
    df_for_training.to_csv(FILTERED_TRAINING_FILE, index=False)
    df_full.to_csv(FILTERED_FULL_FILE, index=False)
    
    print(f"Датасет для обучения сохранен в {FILTERED_TRAINING_FILE}")
    print(f"Полный датасет (для теста) сохранен в {FILTERED_FULL_FILE}")
    
    # Статистика
    print(f"\nСтатистика по датасету для обучения:")
    print(f"   Период: {df_for_training['datetime'].min()} - {df_for_training['datetime'].max()}")
    print(f"   Количество лет: {df_for_training['year'].nunique()}")
    print(f"   Используемые годы: {sorted(df_for_training['year'].unique().tolist())}")
    
    print(f"\nСтатистика по полному датасету:")
    print(f"   Период: {df_full['datetime'].min()} - {df_full['datetime'].max()}")
    print(f"   Количество лет: {df_full['year'].nunique()}")
    print(f"   Используемые годы: {sorted(df_full['year'].unique().tolist())}")

    return df_for_training, df_full

if __name__ == "__main__":
    df_train, df_full = create_final_dataset()
    print("\nФинальная предобработка завершена успешно!")
