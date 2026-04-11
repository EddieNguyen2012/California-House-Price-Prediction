"""
Feature_Engineering.py

Contains all the functions for feature engineering. A sort of scrapyard of different methods

Ordered by:
    - Helper functions
    - In-use functions
    - Work-in-progress functions
    - Temporarily unused functions

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import path_finder
"""================================ HELPER ==========================="""

# I think everything here is Eddie's code? Please verify or correct if not.
# Eddie: Yes it is :)

# Functions working with stacked data are for columns with list values (i.e. Flooring)
def destack(x):
    if pd.isna(x) or x is None:
        return []

    items = str(x).split(',')

    cleaned = [i.strip() for i in items if i.strip().lower() not in ['nan', 'none', 'null', '']]

    return cleaned


def stacked_data_encode(df, feature):
    # 1. Preprocess into lists of strings
    extracted = df[feature].apply(destack)

    # 2. Initialize and Fit the MultiLabelBinarizer
    mlb = path_finder.get_imputer_artifacts(feature, 'encoder')

    if mlb is None:
        mlb = MultiLabelBinarizer()
        mlb.fit(extracted)
        path_finder.create_artifacts(mlb, feature, 'encoder')

    print(f'Encoding {feature}...')
    encoded_array = mlb.transform(extracted)

    column_names = [f"{feature}_{cls}" for cls in mlb.classes_]
    encoded_df = pd.DataFrame(encoded_array, columns=column_names)

    print(f'Encoded {feature} with MultiLabelBinarizer.')
    return encoded_df

# Turns month into a feature
def sin_cyclical_encoding(x):
    return np.sin(2 * np.pi * (x.month/12.0))

def cos_cyclical_encoding(x):
    return np.cos(2 * np.pi * (x.month/12.0))

# Parse PostalCode column into 5-digit integer format - want to encode categorically in future.
def zipcode_parse(org):
    org = str(org)
    if org is not None:
        if len(org) >= 5:
            return int(org[:5])
        else:
            return 0
    else:
        return 0

def school_encode(x):
    if pd.isna(x):
        return False
    else:
        return True

def lag_price_by_zip(df, unit=1):
    postal_code_agg = df.groupby(['PostalCode', 'ReadDate'])['ClosePrice'].mean()
    # original_agg.sort_values(['PostalCode', 'ReadDate'], inplace=True, ascending=False)
    for zipcode in df.PostalCode.unique():
        first = postal_code_agg[zipcode].iloc[0]
        postal_code_agg[zipcode] = postal_code_agg[zipcode].shift(unit, fill_value=first)

    postal_code_agg = postal_code_agg.reset_index()
    postal_code_agg.rename(columns={"ClosePrice": "lastPriceByZip"}, inplace=True)
    return postal_code_agg
"""================================ IN USE ==========================="""

def baseline_feature_engineer(df: pd.DataFrame):
    """
    Engineers all features for baseline testing (linear regression)
    """
    df = df.copy()

    df['PostalCode'] = df['PostalCode'].apply(zipcode_parse)
    df = df[df['PostalCode'].astype(str).str.startswith('9')] # dropping zip codes outside of California (~1 in 10000 roughly)
    last_price_by_zip = lag_price_by_zip(df)
    df = pd.merge(df, last_price_by_zip, how='left', on=['PostalCode', 'ReadDate'])

    print('Finished parsing PostalCode.')

    # CloseDate - 
    df['sin_closed_date'] = df['CloseDate'].apply(sin_cyclical_encoding)
    df['cos_closed_date'] = df['CloseDate'].apply(cos_cyclical_encoding)
    df.drop('ReadDate', axis=1, inplace=True)
    print('Finished cyclical encoding CloseDate.')

    # DaysOnMarket - Changing negaive values to positive ones (~1 in 10000 entries)
    df['DaysOnMarket'] = np.abs(df['DaysOnMarket']) 
    print('Finished transforming abs(DaysOnMarket).')

    df['log_price'] = df['ClosePrice'].apply(lambda x: np.log(x))  # Transform close price by logging
    print('Finished transforming log(ClosePrice).')
    df.drop('ClosePrice', axis=1, inplace=True)


    df['log_lastPriceByZip'] = df['lastPriceByZip'].apply(lambda x: np.log(x))  # Transform close price by logging
    print('Finished transforming log(lastPriceByZip).')
    df.drop('lastPriceByZip', axis=1, inplace=True)


    df['HighSchoolDistrict'] = df['HighSchoolDistrict'].apply(school_encode)
    for feature in ['Flooring', 'Levels']:
        mapped = stacked_data_encode(df, feature)
        df[mapped.columns] = mapped
        df.drop([feature], axis=1, inplace=True)
        print(f'Finished de-stacking and encoding {feature}.')
    
    return df

"""================================ WORK IP ==========================="""

"""================================ UNUSED ==========================="""