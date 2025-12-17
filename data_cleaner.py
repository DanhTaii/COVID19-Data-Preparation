import pandas as pd
import numpy as np

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
RAW_DATA_PATH = "data/raw-owid-covid-data.csv"

def preprocessing():
    df = pd.read_csv(RAW_DATA_PATH)

    # Chọn ra những cột trọng tâm
    cols_to_keep = [
        'iso_code', 'continent', 'location', 'date',
        'total_cases', 'total_deaths', 'new_cases', 'new_deaths',
        'population', 'people_vaccinated', 'people_fully_vaccinated'
    ]

    # Copy tạo bản sao
    df_clean = df[cols_to_keep].copy()

    # Chuyển đổi dữ liệu ngày
    df_clean['date'] = pd.to_datetime(df_clean['date'])

    df_clean = df_clean.sort_values(['location', 'date'])

    # Xử lý dữ liệu bị thiếu
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby('location')[numeric_cols].ffill()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Khử nhiễu theo chu kỳ
    # Tính cột mới cho cases
    df_clean['new_cases_smoothed'] = df_clean.groupby('location')['new_cases'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # Tính cột mới cho deaths
    df_clean['new_death_smoothed'] = df_clean.groupby('location')['new_deaths'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df_clean.to_parquet("data/covid_cleaned.parquet", engine="pyarrow", index=False)
    # df.to_parquet("data.parquet", engine="pyarrow", index=False)

preprocessing()