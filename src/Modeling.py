import pandas as pd
from src.Preprocessing.main import *

def load_data():
    df = pd.read_parquet('../Data/clean_data.parquet')
    return df

