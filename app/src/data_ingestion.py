# src/data_ingestion.py

import pandas as pd
import os
import logging
from datetime import datetime

# Пути к файлам (можно передавать как аргументы)
RAW_DATA_DIR = "../data/raw"
CONSUMPTION_FILE = "consumption_data.csv"
WEATHER_FILE = "weather_data.csv"
CALENDAR_FILE = "russian_production_calendar_2017_2025.csv"

def load_consumption_data():
    """
    Загружает и проверяет файл потребления.
    """
    file_path = os.path.join(RAW_DATA_DIR, CONSUMPTION_FILE)
    logging.info(f"Загрузка потребления из {file_path}")
    try:
        df = pd.read_csv(file_path, sep=';', parse_dates=['date'])
        logging.info(f"Файл потребления загружен. Размер: {df.shape}")
        # Базовая проверка
        required_columns = ['date', 'hour', 'consumption', 'temperature']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют столбцы в файле потребления: {missing_cols}")
        # Проверка формата столбцов (пример)
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Столбец 'date' должен быть типа datetime.")
        if not pd.api.types.is_numeric_dtype(df['hour']):
             raise ValueError("Столбец 'hour' должен быть числовым.")
        if not pd.api.types.is_numeric_dtype(df['consumption']):
             raise ValueError("Столбец 'consumption' должен быть числовым.")
        if not pd.api.types.is_numeric_dtype(df['temperature']):
             raise ValueError("Столбец 'temperature' должен быть числовым.")

        logging.info("Проверка файла потребления пройдена.")
        return df
    except FileNotFoundError:
        logging.error(f"Файл {file_path} не найден.")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Файл {file_path} пуст.")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла потребления: {e}")
        raise

def load_weather_data():
    """
    Загружает и проверяет файл погоды.
    """
    file_path = os.path.join(RAW_DATA_DIR, WEATHER_FILE)
    logging.info(f"Загрузка погоды из {file_path}")
    try:
        # Предполагаем, что заголовок начинается с "# Метеостанция"
        # Читаем первые строки, чтобы найти заголовок
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_line = None
        data_start_row = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('"Местное время'):
                header_line = line.strip()
                data_start_row = i + 1
                break

        if header_line is None:
            raise ValueError("Не найдена строка с заголовком в файле погоды.")

        df = pd.read_csv(
            file_path,
            sep=';',
            skiprows=data_start_row,
            on_bad_lines='skip',
            encoding='utf-8'
        )

        if df.empty:
            raise ValueError("Файл погоды пуст после пропуска заголовков.")

        logging.info(f"Файл погоды загружен. Размер: {df.shape}")

        # Базовая проверка - наличие нужных столбцов (T, U, Ff)
        # Файл может содержать разное количество столбцов, проверим первые 13
        if len(df.columns) >= 13:
            # Предполагаемый порядок столбцов из описания
            expected_names = ['datetime', 'T', 'P0', 'P', 'U', 'DD', 'Ff', 'ff10', 'WW', 'WW2', 'clouds', 'VV', 'Td']
            actual_columns = df.columns[:13]
            # Переименовываем первые 13 столбцов
            df = df[actual_columns]
            df.columns = expected_names[:len(actual_columns)]
        else:
            # Если меньше 13, используем текущие имена или задаём минимальные
            logging.warning(f"Файл погоды содержит менее 13 столбцов ({len(df.columns)}). Проверьте структуру.")
            # Попробуем переименовать минимальный набор, если он есть
            if 'datetime' in df.columns and 'U' in df.columns and 'Ff' in df.columns:
                 # Оставляем как есть, если есть ключевые столбцы
                 pass
            else:
                 # Или задаём имена, если известен порядок (например, первые три)
                 if len(df.columns) >= 3:
                      df = df.iloc[:, :3] # Берём первые 3 столбца
                      df.columns = ['datetime', 'T', 'U'] # Пример, нужно уточнить
                 else:
                      raise ValueError("Недостаточно столбцов для минимального набора (datetime, T, U).")

        # Преобразование datetime
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        df.dropna(subset=['datetime'], inplace=True) # Удаляем строки с некорректной датой

        logging.info("Проверка файла погоды пройдена.")
        return df

    except FileNotFoundError:
        logging.error(f"Файл {file_path} не найден.")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла погоды: {e}")
        raise

def load_calendar_data():
    """
    Загружает и проверяет файл календаря.
    """
    file_path = os.path.join(RAW_DATA_DIR, CALENDAR_FILE)
    logging.info(f"Загрузка календаря из {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Файл календаря загружен. Размер: {df.shape}")
        # Базовая проверка
        required_columns = ['date', 'day_type']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют столбцы в файле календаря: {missing_cols}")

        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df.dropna(subset=['date'], inplace=True) # Удаляем строки с некорректной датой

        logging.info("Проверка файла календаря пройдена.")
        return df
    except FileNotFoundError:
        logging.error(f"Файл {file_path} не найден.")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла календаря: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Тестовая загрузка
    try:
        df_con = load_consumption_data()
        df_w = load_weather_data()
        df_cal = load_calendar_data()
        print("Загрузка данных завершена успешно.")
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
