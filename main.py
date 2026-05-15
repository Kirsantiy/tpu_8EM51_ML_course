import pandas as pd
import os

def load_car_data():
    base_path = os.path.join('8', 'data', 'raw')
    
    files = [
        'car_data.csv',
        'CAR_DETAILS_FROM_CAR_DEKHO.csv',
        'Car_details_v3.csv'
    ]
    
    datasets = {}

    print("--- Загрузка данных из DVC ---")
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        
        if os.path.exists(file_path):
            datasets[file_name] = pd.read_csv(file_path)
            print(f"✅ {file_name}: загружен ({len(datasets[file_name])} строк)")
        else:
            print(f"❌ {file_name}: файл не найден! Проверьте 'dvc pull'")
            
    return datasets

if __name__ == "__main__":
    data = load_car_data()
    
    if 'car_data.csv' in data:
        df = data['car_data.csv']
        
        print("\n--- Анализ первого датасета ---")
        print(df.info())
        print(df.head())
