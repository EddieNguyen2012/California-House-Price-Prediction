import pandas as pd
import src.path_finder as pathfinder
from src.Ingestion import DataIngestion
from src.Pipeline.Data_Cleaning import baseline_impute_normalize
from src.Pipeline.Feature_Engineering import baseline_feature_engineer
from sklearn.model_selection import train_test_split
import os

##### List of usable columns from Jenny's EDA. Change this to get more columns
columns_to_use = [

# --------- Other/future Use Cases. Drop when modeling for now -----------
# 
# --- Agent & Office Info --- (Potential to impute/combine into boolean "Has X" variables)
# 'BuyerAgentAOR', 'BuyerAgentFirstName', 'BuyerAgentLastName', 'BuyerAgentMlsId',
# 'BuyerOfficeAOR', 'BuyerOfficeName', 'ListAgentAOR', 'ListOfficeName',

# ---------- Columns in use ----------------------------------------------
# --- Location & Address ---
## Eddie removed due to overlap info 'MLSAreaMajor', 'StateOrProvince', 'HighSchoolDistrict', 'CountyOrParish', 'City', 'StreetNumberNumeric',
'Latitude', 'Longitude',
'PostalCode',

# --- Property Specs ---
'AttachedGarageYN', 'BathroomsTotalInteger', 'BedroomsTotal', 'FireplaceYN',
'Flooring', 'GarageSpaces', 'Levels', 'LivingArea', 'MainLevelBedrooms',
'NewConstructionYN', 'ParkingTotal', 'PoolPrivateYN', 'Stories', 'ViewYN', 'YearBuilt',

# --- Lot Information ---
'LotSizeAcres', 'LotSizeArea', 'LotSizeSquareFeet',

# --- Transaction & Status ---
'AssociationFee', 'CloseDate', 'ClosePrice', 'DaysOnMarket',
    # 'MlsStatus',
]
#######

# Median imputation -> columns including LivingArea, LotSizeAcres, LotSizeArea, LotSizeSquareFeet, BathroomsTotalInteger, Stories, and GarageSpaces.
# Reasons: The median LivingArea of 1,810 square feet represents a typical suburban single-family home; The median lot size of approximately 7,200 square feet is consistent with standard residential parcels in Southern California; Similarly, the median values of 2 bathrooms, 1 story, and 2 garage spaces reflect common housing configurations in California suburban areas. So median imputation seems reasonable for these columns.


#### Huiyu's Feature Handling Plan

# Zip code parsing -> columns including PostalCode.
# Reasons: PostalCode is a geographic identifier and should be treated as categorical.

# Binary encoding -> columns including AttachedGarageYN, FireplaceYN, NewConstructionYN, PoolPrivateYN, and ViewYN.
# Reasons: These variables indicate the presence or absence of specific property attributes. Convert to 0/1.

# One-Hot Encoding -> columns including PropertyType, PropertySubType, Levels, MlsStatus, and StateOrProvince.
# Reasons: These variables are nominal categories without inherent ordering.

# Categorical -> columns including City, CountyOrParish, and MLSAreaMajor.
# Reasons: These variables capture geographic variation in housing markets. 
# If the number of unique categories is moderate, one-hot encoding can be applied. 
# If cardinality is high, grouping or alternative encoding methods may be considered to control feature dimensionality.

# Date extraction -> columns including CloseDate.
# Reasons: Raw date values are not directly meaningful for modeling. 
# We extracting components such as year and month to capture seasonality and temporal market trends in property transactions.
# Eddie's note: month is encoded using cyclical engineering using sin and cos function to simulate the cyclical nature of the data

# Numeric features keep as it is as continuous variables -> columns including Latitude, Longitude, YearBuilt, BedroomsTotal, MainLevelBedrooms, ParkingTotal, AssociationFee, DaysOnMarket, and StreetNumberNumeric.
# Reasons: They are inherently numerical. 

# Target variable -> ClosePrice.
# Reasons: used as the prediction target.

####


# Get data withing restriction.
# Params: columns = required columns
def get_unprocessed_data(accessor: DataIngestion=DataIngestion(data_path=pathfinder.CSV_DIR), columns=None, aggregations = None):

    if columns is None:
        columns = columns_to_use
    if aggregations is not None:
        columns = columns.append(aggregations)

    # Optimized query from Johnny's EDA
    df = accessor.query(
        f"""
        SELECT {', '.join(columns)}
        FROM Property
        WHERE PropertyType = 'Residential'
          AND PropertySubType = 'SingleFamilyResidence'
          AND ClosePrice > 0
          AND LivingArea > 0
          AND Latitude IS NOT NULL
          AND Longitude IS NOT NULL
          AND Latitude BETWEEN 32 AND 43
          AND Longitude BETWEEN -125 AND -113
        """
    )
    return df

# To be used for trimming test set ClosePrice outliers
def trimming_quantiles(X, y, quantile=0.05):
    if not (0 <= quantile < 0.5):
        raise ValueError("quantile must be in [0, 0.5)")

    lo, hi = y.quantile([quantile, 1 - quantile])
    mask = (y >= lo) & (y <= hi)
    return X.loc[mask], y.loc[mask]

# Splits into training and test set, where test set is random 20% of observations. Includes trimming.
def train_test_split_with_trimming(df, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=random_state)
    X_test, y_test = trimming_quantiles(X_test, y_test)
    return X_train, X_test, y_train, y_test

# Johnny: Splits into training and test set, where test set is most recent month of data. Includes trimming.
def train_test_recent_month(df, y, date_col="CloseDate"):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Find most recent month
    latest_month = df[date_col].dt.to_period("M").max()

    # Mask for test set
    test_mask = df[date_col].dt.to_period("M") == latest_month

    # Split
    X_train = df.loc[~test_mask]
    X_test = df.loc[test_mask]
    y_train = y.loc[~test_mask]
    y_test = y.loc[test_mask]

    # Apply trimming
    X_test, y_test = trimming_quantiles(X_test, y_test)

    return X_train, X_test, y_train, y_test

#  Eddie's code continues from here.
def store_data_in_parquet(df: pd.DataFrame, path=None):
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_parquet(os.path.join(path, '/clean_data.parquet'))

def store_data_in_csv(df: pd.DataFrame, path=None):
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, 'clean_data.csv'))

# Johnny made changes
# For baseline testing, call function without arguments
# To do your own imputation/normalization/feature engineering, change use_for.
def get_preprocessed_data(path=pathfinder.CSV_DIR, output_as: str = "standard_split", use_for: str = "baseline"):
    '''
    Pipeline function.

    :param path: path to raw csv files. (by default it is ../IDX_data)
    :param output_as: Choose between several options:
        "csv": saves processed csv to path
        "random_split": (4 DataFrames) returns x_train, x_test, y_train, y_test (test set = random 20% of observations)
        "standard_split": (4 DataFrames) returns x_train, x_test, y_train, y_test (test set = most recent month of observations)
        None of the above: returns full 1 DataFrame
    :param use_for: choose between "baseline" or ...
    '''

    ## Read required data
    df = get_unprocessed_data(columns=columns_to_use)

    if use_for == "baseline":
        df = baseline_feature_engineer(df)
        df = baseline_impute_normalize(df)


    if output_as == "csv":
        store_data_in_csv(df, path=pathfinder.OUTPUT_DIR)
    if output_as == "random_split":
        y = df['log_price']
        df.drop('log_price', axis=1, inplace=True) 
        return train_test_split_with_trimming(df=df, y=y, test_size=0.2, random_state=42)
    if output_as == "standard_split":
        y = df['log_price']
        df.drop('log_price', axis=1, inplace=True)
        x_train, x_test, y_train, y_test = train_test_recent_month(df,y)
        return x_train, x_test, y_train, y_test
    else:
        return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_preprocessed_data(output_as='random_split')





