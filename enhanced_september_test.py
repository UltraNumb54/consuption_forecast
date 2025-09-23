import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta

# === КОНСТАНТЫ ===
MODEL_PATH = 'enhanced_energy_model.cbm'
FULL_DATA_FILE = 'enhanced_filtered_full_data.csv'  # Полный датасет, включая сентябрь 2025
FEATURES_LIST_FILE = 'model_feature_columns.pkl'    # Файл со списком признаков (сохраняется при обучении)
TEST_START_DATE = '2025-09-01'
TEST_DAYS = 3  # Прогноз на 3 дня вперед

def load_model():
    """Загрузка обученной модели"""
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    print("Модель успешно загружена")
    return model

def simple_september_test():
    """Простое тестирование на готовых данных сентября 2025"""
    print("=== ПРОСТОЕ ТЕСТИРОВАНИЕ НА СЕНТЯБРЕ 2025 ===")
    
    # Загружаем модель
    model = load_model()
    
    # Загружаем полный датасет (с сентября 2025)
    df_full = pd.read_csv(FULL_DATA_FILE)
    df_full['datetime'] = pd.to_datetime(df_full['datetime'])
    print(f"Полный датасет загружен: {len(df_full)} записей")
    print(f"Период данных: {df_full['datetime'].min()} - {df_full['datetime'].max()}")

    # Загружаем список признаков, на которых обучалась модель
    import joblib
    feature_columns = joblib.load(FEATURES_LIST_FILE)
    print(f"Загружено {len(feature_columns)} признаков для модели")

    # Определяем период тестирования
    test_start = datetime.strptime(TEST_START_DATE, '%Y-%m-%d')
    test_end = test_start + timedelta(days=TEST_DAYS)
    print(f"Тестирование с {test_start} по {test_end}")

    # Фильтруем данные сентября 2025
    test_data = df_full[(df_full['datetime'] >= test_start) & (df_full['datetime'] < test_end)]
    print(f"Найдено {len(test_data)} часовых записей для тестирования")

    if len(test_data) == 0:
        print("Ошибка: Нет данных для тестового периода!")
        return

    # Подготавливаем X и y для теста
    # Берем ТОЛЬКО те колонки, на которых обучалась модель
    X_test = test_data[feature_columns]
    y_test = test_data['consumption']

    # Проверяем, что все колонки присутствуют
    missing_cols = [col for col in feature_columns if col not in X_test.columns]
    if missing_cols:
        print(f"Критическая ошибка: Отсутствуют колонки: {missing_cols}")
        return

    # Делаем предсказание
    print("Начинаем предсказания...")
    predictions = model.predict(X_test)

    # Оценка качества
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    mean_actual = np.mean(y_test)

    print(f"\n===ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    print(f"Количество предсказаний: {len(predictions)}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Среднее потребление: {mean_actual:.1f} МВт")
    print(f"Точность (±2.5% от среднего): {mae < (mean_actual * 0.025)}")

    # Визуализация
    prediction_times = test_data['datetime'].tolist()

    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(prediction_times, y_test, 'b-', label='Реальное', linewidth=2, marker='o', markersize=4)
    plt.plot(prediction_times, predictions, 'r--', label='Предсказанное', linewidth=2, marker='x', markersize=4)
    plt.title(f'Прогноз потребления на {TEST_DAYS} дня, начиная с {TEST_START_DATE}')
    plt.ylabel('Потребление (МВт)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    errors = np.abs(y_test.values - predictions)
    plt.plot(prediction_times, errors, 'g-', alpha=0.7, linewidth=2, marker='s', markersize=3)
    plt.axhline(y=mean_actual * 0.025, color='r', linestyle='--', label='Порог ±2.5%')
    plt.title(f'Абсолютные ошибки (MAE = {mae:.3f})')
    plt.ylabel('Ошибка (МВт)')
    plt.xlabel('Дата и время')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('simple_september_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Сохранение детальных результатов
    results_df = pd.DataFrame({
        'datetime': prediction_times,
        'actual_consumption': y_test.values,
        'predicted_consumption': predictions,
        'absolute_error': errors
    })
    results_df.to_csv('simple_september_test_detailed.csv', index=False)
    print("Детальные результаты сохранены в simple_september_test_detailed.csv")

    return results_df

if __name__ == "__main__":
    results = simple_september_test()
    print("\nТестирование завершено успешно!")
