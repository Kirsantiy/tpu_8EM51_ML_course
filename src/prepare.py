import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def run_eda_preprocess():
    # CSV файлы лежат в корне проекта
    files = [
        'car_data.csv',
        'CAR_DETAILS_FROM_CAR_DEKHO.csv',
        'Car_details_v3.csv'
    ]
    
    dataframes = []

    print("--- 1. Загрузка и приведение к единому стандарту ---")
    
    rename_map = {
        'car_name': 'name',
        'fuel_type': 'fuel',
        'kms_driven': 'km_driven',
        'selling_price': 'selling_price'
    }

    for file_name in files:
        if not os.path.exists(file_name):
            print(f"Файл {file_name} не найден!")
            continue
            
        df = pd.read_csv(file_name)
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns=rename_map)
        
        if file_name == 'car_data.csv':
            df['selling_price'] = (df['selling_price'] * 100000).astype(int)
            if 'present_price' in df.columns:
                df['present_price'] = (df['present_price'] * 100000).astype(int)
            print(f"✅ {file_name}: цены переведены из млн в целые числа")
        else:
            df['selling_price'] = df['selling_price'].astype(int)
        
        dataframes.append(df)

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True, sort=False)
    
    # Очистка технических характеристик
    print("\n--- 2. Очистка технических характеристик ---")
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).str.split(' ').str[0]
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    combined_df = combined_df.fillna(0)
    
    for col in ['selling_price', 'km_driven', 'year', 'engine', 'seats']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(int)

    # ─────────────────────────────────────────────
    # 2б. Конвертация признака owner
    # ─────────────────────────────────────────────
    print("--- 2б. Конвертация признака owner ---")
    if "owner" in combined_df.columns:
        owner_str_map = {
            "First Owner":          1,
            "Second Owner":         2,
            "Third Owner":          3,
            "Fourth & Above Owner": 4,
            "Test Drive Car":       1
        }

        def convert_owner(val):
            if isinstance(val, str):
                # Формат: "First Owner", "Second Owner"...
                return owner_str_map.get(val.strip(), 1)
            else:
                # Формат: 0, 1, 2... → добавляем +1
                return int(val) + 1

        combined_df["owner"] = combined_df["owner"].apply(convert_owner)
        print(f"✅ owner сконвертирован. Уникальные значения: {sorted(combined_df['owner'].unique())}")
    else:
        print("⚠️  Столбец owner не найден ни в одном датасете")

    # ─────────────────────────────────────────────
    # 3. Отбор нужных столбцов
    # ─────────────────────────────────────────────
    print("\n--- 3. Отбор нужных столбцов ---")
    keep_cols = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price']
    keep_cols = [col for col in keep_cols if col in combined_df.columns]
    combined_df = combined_df[keep_cols]
    print(f"✅ Оставлены столбцы: {keep_cols}")

    # ─────────────────────────────────────────────
    # 4а. Выборки для CatBoost (без OneHot — категории остаются строками)
    # ─────────────────────────────────────────────
    print("--- 4а. Подготовка выборок для CatBoost (без OneHot) ---")
    cb_df = combined_df.copy()

    # Нормализуем только числовые столбцы
    cb_numeric_cols = ["year", "km_driven", "owner"]
    cb_scaler = MinMaxScaler(feature_range=(0, 1))
    cb_df[cb_numeric_cols] = cb_scaler.fit_transform(cb_df[cb_numeric_cols])

    # selling_price тоже нормализуем (как в основной ветке)
    sp_scaler = MinMaxScaler(feature_range=(0, 1))
    cb_df[["selling_price"]] = sp_scaler.fit_transform(cb_df[["selling_price"]])

    # Разбивка с тем же random_state
    cb_df = cb_df.sample(frac=1, random_state=42).reset_index(drop=True)
    cb_train, cb_temp = train_test_split(cb_df, test_size=0.30, random_state=42)
    cb_val,   cb_test = train_test_split(cb_temp, test_size=0.33, random_state=42)

    cat_feature_cols = ["fuel", "seller_type", "transmission"]
    cb_feature_cols  = [col for col in cb_df.columns if col != "selling_price"]

    os.makedirs("data", exist_ok=True)
    for split_name, split_df in [("train", cb_train), ("val", cb_val), ("test", cb_test)]:
        X = split_df[cb_feature_cols]
        y = split_df[["selling_price"]]
        X.to_csv(f"data/X_{split_name}_cb.csv", index=False)
        y.to_csv(f"data/y_{split_name}_cb.csv", index=False)
        print(f"✅ data/X_{split_name}_cb.csv  |  data/y_{split_name}_cb.csv")

    print(f"Категориальные признаки для CatBoost: {cat_feature_cols}")

    # ─────────────────────────────────────────────
    # 4. OneHotEncoding категориальных столбцов
    # ─────────────────────────────────────────────
    print("\n--- 4. OneHotEncoding категориальных признаков ---")
    cat_cols = ['fuel', 'seller_type', 'transmission']
    cat_cols = [col for col in cat_cols if col in combined_df.columns]
    combined_df = pd.get_dummies(combined_df, columns=cat_cols)
    combined_df = combined_df.astype({col: int for col in combined_df.select_dtypes(include='bool').columns})
    ohe_cols = [col for col in combined_df.columns if any(col.startswith(c + '_') for c in cat_cols)]
    print(f"✅ Созданы OneHot-столбцы: {ohe_cols}")

    # ─────────────────────────────────────────────
    # 5. Нормализация числовых признаков (кроме selling_price)
    # ─────────────────────────────────────────────
    print("\n--- 5. Нормализация числовых признаков ---")
    #all_numeric = combined_df.select_dtypes(include=['float64', 'int64']).columns
    all_numeric = combined_df.select_dtypes(include='number').columns
    features_to_scale = list(all_numeric)
    #[col for col in all_numeric if col != 'selling_price']

    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_df[features_to_scale] = scaler.fit_transform(combined_df[features_to_scale])

    print(f"Нормализованы столбцы: {features_to_scale}")
    print(f"Столбец 'selling_price' остался в оригинале: {combined_df['selling_price'].iloc[0]}")

    # ─────────────────────────────────────────────
    # 6. Разбивка на train / val / test
    # ─────────────────────────────────────────────
    print("\n--- 6. Разбивка на train/val/test ---")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_data, temp_data = train_test_split(combined_df, test_size=0.30, random_state=42)
    val_data, test_data   = train_test_split(temp_data,   test_size=0.33, random_state=42)

    print(f"Размер тренировочной выборки: {train_data.shape}")
    print(f"Размер валидационной выборки: {val_data.shape}")
    print(f"Размер тестовой выборки:      {test_data.shape}")

    # ─────────────────────────────────────────────
    # 7. Сохранение в data/
    # ─────────────────────────────────────────────
    print("\n--- 7. Сохранение файлов в data/ ---")
    os.makedirs('data', exist_ok=True)

    target_col   = 'selling_price'
    feature_cols = [col for col in combined_df.columns if col != target_col]

    for split_name, split_df in [('train', train_data), ('val', val_data), ('test', test_data)]:
        X = split_df[feature_cols]
        y = split_df[[target_col]]
        X.to_csv(f'data/X_{split_name}.csv', index=False)
        y.to_csv(f'data/y_{split_name}.csv', index=False)
        print(f"✅ data/X_{split_name}.csv  |  data/y_{split_name}.csv")

    print(f"\nИтоговые столбцы в X: {feature_cols}")
    print(f"Столбец в y: {target_col}")
    print(f"\n✅ Готово! Всего строк в датасете: {combined_df.shape[0]}")

if __name__ == "__main__":
    run_eda_preprocess()
