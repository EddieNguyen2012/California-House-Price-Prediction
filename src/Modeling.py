import pandas as pd

def load_data():
    df = pd.read_parquet('../Data/clean_data.parquet')
    return df