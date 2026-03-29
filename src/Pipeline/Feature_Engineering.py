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

"""================================ HELPER ==========================="""

# I think everything here is Eddie's code? Please verify or correct if not.
# Eddie: Yes it is :)

# Functions working with stacked data are for columns with list values (i.e. Flooring)
def destack(x):
    if x is None:
        return None
    else:
        return x.split(',')

def extract_stacked_data(df, feature):
    extracted = df[feature].apply(destack)
    unique = set()
    for entry in extracted:
        if entry is not None:
            for floor_type in entry:
                unique.add(floor_type)
    return list(unique)

def stacked_data_encode(df, feature):
    unique_ordered = extract_stacked_data(df, feature)

    def mapping(x):
        if x is None:
            return np.zeros(len(unique_ordered))
        else:
            mapper = np.zeros(len(unique_ordered))
            for category in x:
                if category in unique_ordered:
                    mapper[unique_ordered.index(category)] = 1

            return mapper

    extracted = df[feature].apply(destack)
    mapped = extracted.apply(mapping)
    unique_ordered = [feature + '_' + x for x in unique_ordered]
    mapped = pd.DataFrame(np.vstack(mapped), columns=unique_ordered)

    return mapped, unique_ordered

# Turns month into a feature
def cyclical_encoding(x):
    return np.sin(2 * np.pi * (x.month/12.0))

# Parse PostalCode column into 5-digit integer format - want to encode categorically in future.
def zipcode_parse(org: str):
    if org is not None:
        if len(org) >= 5:
            return int(org[:5])
        else:
            return 0
    else:
        return 0

"""================================ IN USE ==========================="""

def baseline_feature_engineer(df: pd.DataFrame):
    """
    Engineers all features for baseline testing (linear regression)
    """
    df = df.copy()

    df['PostalCode'] = df['PostalCode'].apply(zipcode_parse)
    df = df[df['PostalCode'].astype(str).str.startswith('9')] # dropping zip codes outside of California (~1 in 10000 roughly)

    # CloseDate - 
    df['sin_closed_date'] = df['CloseDate'].apply(cyclical_encoding)

    # DaysOnMarket - Changing negaive values to positive ones (~1 in 10000 entries)
    df['DaysOnMarket'] = np.abs(df['DaysOnMarket']) 

    df['log_price'] = df['ClosePrice'].apply(lambda x: np.log(x))  # Transform close price by logging
    df.drop('ClosePrice', axis=1, inplace=True)

    for feature in ['Flooring', 'Levels']:
        mapped, types = stacked_data_encode(df, feature)
        df[types] = mapped
        df.drop([feature], axis=1, inplace=True)
    
    return df


"""================================ WORK IP ==========================="""

"""================================ UNUSED ==========================="""