import pandas as pd
import os

def run_eda_preprocess():
    # Путь к папке с датасетами
    base_path = os.path.join('8', 'data', 'raw')
    files = [
        'car_data.csv',
        'CAR_DETAILS_FROM_CAR_DEKHO.csv',
        'Car_details_v3.csv'
    ]
    
    dataframes = []

    print("--- 1. Загрузка и приведение к единому стандарту ---")
    
    # Словарь для объединения одинаковых по смыслу столбцов
    rename_map = {
        'car_name': 'name',
        'fuel_type': 'fuel',
        'kms_driven': 'km_driven',
        'selling_price': 'selling_price'
    }

    for file_name in files:
        path = os.path.join(base_path, file_name)
        if not os.path.exists(path):
            print(f"Файл {file_name} не найден!")
            continue
            
        df = pd.read_csv(path)
        # Приводим названия к нижнему регистру
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns=rename_map)
        
        # Переводим цены на автомобили в одинаковый формат
        if file_name == 'car_data.csv':
            df['selling_price'] = (df['selling_price'] * 100000).astype(int)
            if 'present_price' in df.columns:
                df['present_price'] = (df['present_price'] * 100000).astype(int)
            print(f"✅ {file_name}: цены переведены из млн в целые числа")
        else:
            # В других датасетах всё явно приводим к int
            df['selling_price'] = df['selling_price'].astype(int)
        
        dataframes.append(df)

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True, sort=False)
    
    # Очистка мусорных строк (mileage, engine, max_power)
    print("\n--- 2. Очистка технических характеристик ---")
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if col in combined_df.columns:
            # Извлекаем числовую часть (до пробела)
            combined_df[col] = combined_df[col].astype(str).str.split(' ').str[0]
            # Конвертируем в числа, некорректные записи станут NaN
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    combined_df = combined_df.fillna(0)
    
    # Проверка ключевых столбцов, что они целые числа
    for col in ['selling_price', 'km_driven', 'year', 'engine', 'seats']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(int)

    # Сохранение
    output_dir = os.path.join('8', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_data.csv')
    
    combined_df.to_csv(output_path, index=False)

    print("\n--- Итоговый объединенный датасет ---")
    print(f"Размер таблицы: {combined_df.shape[0]} строк, {combined_df.shape[1]} колонок")
    print("\nСтатистика по ценам (selling_price):")
    print(combined_df['selling_price'].describe().apply(lambda x: format(x, 'f')))
    print("\nСтолбцы в итоговом файле:", combined_df.columns.tolist())
    print(f"\n✅ Файл успешно сохранен: {output_path}")

if __name__ == "__main__":
    run_eda_preprocess()
