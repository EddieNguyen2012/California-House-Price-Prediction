import pandas as pd
import numpy as np
import src.path_finder as pathfinder
from src.Ingestion import DataIngestion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import os

##### List of usable columns from Jenny's EDA. Change this to get more columns
columns_to_use = [
# --- Agent & Office Info ---
# 'BuyerAgentAOR', 'BuyerAgentFirstName', 'BuyerAgentLastName', 'BuyerAgentMlsId',
# 'BuyerOfficeAOR', 'BuyerOfficeName', 'ListAgentAOR', 'ListOfficeName',

# --- Location & Address ---
## Eddie removed due to overlap info 'MLSAreaMajor', 'StateOrProvince', 'HighSchoolDistrict', 'CountyOrParish', 'City',
'Latitude', 'Longitude',
'PostalCode', 'StreetNumberNumeric',

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
def get_unprocessed_data(accessor: DataIngestion, columns=None, aggregations = None):
    if columns is None:
        columns = columns_to_use
    if aggregations is not None:
        columns = columns.append(aggregations)

    ######### Optimized query from Johnny's EDA
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

###### Eddie's Code
def cyclical_encoding(x):
    return np.sin(2 * np.pi * (x.month/12.0))

def zipcode_parse(org: str):
    if org is not None:
        if len(org) >= 5:
            return org[:5]
        else:
            return 0
    else:
        return 0

def trimming_quantiles(X, y, quantile=0.05):
    if not (0 <= quantile < 0.5):
        raise ValueError("quantile must be in [0, 0.5)")

    lo, hi = y.quantile([quantile, 1 - quantile])
    mask = (y >= lo) & (y <= hi)
    return X.loc[mask], y.loc[mask]


def train_test_split_with_trimming(df, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=random_state)
    X_train, y_train = trimming_quantiles(X_train, y_train)
    X_test, y_test = trimming_quantiles(X_test, y_test)
    return X_train, X_test, y_train, y_test

def store_data_in_parquet(df: pd.DataFrame, path=None):
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_parquet(os.path.join(path, '/clean_data.parquet'))

def store_data_in_csv(df: pd.DataFrame, path=None):
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, 'clean_data.csv'))


#### Huiyu: Helper function for imputation
def build_imputer(strategy: str = "median"):
    """
    options for strategy:
    - "median"
    - "mean"
    - "most_frequent"
    - "constant"
    """
    from sklearn.impute import SimpleImputer
    return SimpleImputer(strategy=strategy)


#### Huiyu: Helper function to get categorical feature indices by column names
def get_cat_feature_indices(X: pd.DataFrame, cat_cols: list[str]):
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]


#### Huiyu: ColumnTransformer preprocessor
# def build_sklearn_preprocessor(
#     X: pd.DataFrame,
#     categorical_cols: list[str],
#     numeric_cols = None,
#     scale_numeric: bool = True,
# ):
#     """
#     This is a preprocessing ColumnTransformer:
#     - numeric: median impute (+ optional scaling)
#     - categorical: most_frequent impute + one-hot
#     """
#     from sklearn.compose import ColumnTransformer
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import OneHotEncoder, StandardScaler

#     if numeric_cols is None:
#         numeric_cols = [c for c in X.columns if c not in categorical_cols]

#     num_imputer = build_imputer("median")
#     cat_imputer = build_imputer("most_frequent")

#     num_steps = [("imputer", num_imputer)]
#     if scale_numeric:
#         num_steps.append(("scaler", StandardScaler()))
#     num_pipe = Pipeline(steps=num_steps)

#     cat_pipe = Pipeline(steps=[
#         ("imputer", cat_imputer),
#         ("onehot", OneHotEncoder(handle_unknown="ignore")),
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", num_pipe, numeric_cols),
#             ("cat", cat_pipe, categorical_cols),
#         ],
#         remainder="drop",
#         verbose_feature_names_out=False,
#     )
#     return preprocessor

def normalize(series: pd.Series, technique):
    if technique == "minmax":
        normalizer = MinMaxScaler()
        return normalizer.fit_transform(series)
    elif technique == "standard":
        normalizer = StandardScaler()
        return normalizer.fit_transform(series)
    else:
        raise ValueError("Unknown normalization type")

def flooring_encode(df):
    unique_ordered = ['Bamboo', 'Brick', 'Carpet', 'Concrete', 'Laminate', 'Stone', 'Tile', 'Vinyl', 'Wood']

    def map_floor_type(x):
        if x is None:
            return np.zeros(len(unique_ordered))
        else:
            mapper = np.zeros(len(unique_ordered))
            for floor_type in x:
                if floor_type in unique_ordered:
                    mapper[unique_ordered.index(floor_type)] = 1

            return mapper

    def extract_floor_type(x):
        if x is None:
            return None
        else:
            return x.split(',')

    extracted = df['Flooring'].apply(extract_floor_type)
    mapped = extracted.apply(map_floor_type)
    unique_ordered = ["Floor_" + x for x in unique_ordered]
    mapped = pd.DataFrame(np.vstack(mapped), columns=unique_ordered)

    return mapped, unique_ordered

def bool_encode(x):
    if x is None or type(x) is not bool:
        return 0
    if x:
        return 1
    else:
        return 0


def impute(df: pd.DataFrame, target_col: str | list, technique: str, knn_n_neighbors: int = 5, knn_weights: str = 'distance'):
    '''
    Optimizing Jenny's imputation function
    Combine all imputation techniques into one function
    :param df: the input dataframe
    :param target_col: column(s) to impute
    :param technique: 'mean', 'median', 'knn'
    :param knn_n_neighbors: if technique is 'knn', this indicates n_neighbors
    :param knn_weights: 'uniform', 'distance'
    :return:
    '''
    if technique == "median":
        imputer = SimpleImputer(strategy="median")
        return imputer.fit_transform(df[target_col])
    elif technique == "mean":
        imputer = SimpleImputer(strategy="mean")
        return imputer.fit_transform(df[target_col])
    elif technique == "knn":
        imputer = KNNImputer(n_neighbors=knn_n_neighbors, weights=knn_weights)
        return imputer.fit_transform(df[target_col])
    return None

## Improved KNN imputation function
## Idea: make a baseline dataset with low missingness features (geo, lot size,...) (NOT close price to prevent leakage)
## then fit to sklearn.KNN_Imputer and impute columns in the categories Property Specs from the columns_to_use global var.
def knn_imputation():
    pass




#######

####### Jenny (median imputation)
# def compute_medians(df: pd.DataFrame) -> pd.Series:
#     cols_impute = [
#         "LivingArea",
#         "LotSizeAcres",
#         "LotSizeArea",
#         "LotSizeSquareFeet",
#         "BathroomsTotalInteger",
#         "Stories",
#         "GarageSpaces",
#     ]
#
#     cols_impute = [col for col in cols_impute if col in df.columns]
#
#     return df[cols_impute].median()
#
# def impute_with_medians(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
#     df = df.copy()
#     df[medians.index] = df[medians.index].fillna(medians)
#     return df

#######

##### Eddie's code (cont.)
def get_preprocessed_data(path=pathfinder.CSV_DIR, split: bool = False, to_file: bool = False):
    '''
    Pipeline function. Johnny feel free to edit this to integrate your functions into the pipeline.

    :param path: path to raw csv files. (by default it is ../IDX_data
    :param split: if true, return 4 entities X_train, X_test, y_train, y_test, in this order. Else, return the whole dataset. False by default.
    :param to_file: if true, create a csv of the preprocessed df to ../Output
    :return: 1 or multiple pd.DataFrame
    '''
    accessor = DataIngestion(data_path=path)

    ## Read required data
    df = get_unprocessed_data(accessor= accessor, columns=columns_to_use)
    # print(df.columns)
    df['PostalCode'] = df['PostalCode'].apply(zipcode_parse)

    df['engineered_closed_date'] = df['CloseDate'].apply(cyclical_encoding)
    df.drop('CloseDate', axis=1, inplace=True)

    # medians = compute_medians(df)
    # df = impute_with_medians(df, medians)

    df['log_price'] = df['ClosePrice'].apply(lambda x: np.log(x))  # Transform close price by logging
    cols_median_impute = [
        "LivingArea",
        "LotSizeAcres",
        "LotSizeArea",
        "LotSizeSquareFeet",
        "BathroomsTotalInteger",
        "Stories",
        "GarageSpaces",
    ]
    df[cols_median_impute] = impute(df, target_col=cols_median_impute, technique="median")
    flooring_mapped, flooring_types = flooring_encode(df)
    df[flooring_types] = flooring_mapped
    df.drop(['Flooring'], axis=1, inplace=True)
    # boolean_cols = ['AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN', 'PoolPrivateYN', 'ViewYN']
    # for col in boolean_cols:
    #     df[col] = df[col].apply(bool_encode)


    ### Engineered features
    # PostalCode -> zipcode_parse()
    # CloseDate -> cyclical_encoding()
    # ClosePrice -> zipcode_parse()
    # AttachedGarageYN, FireplaceYN, NewConstructionYN, PoolPrivateYN, and ViewYN -> bool_encode()
    ### Imputed features:
    ## Median:
    # "LivingArea",
    # "LotSizeAcres",
    # "LotSizeArea",
    # "LotSizeSquareFeet",
    # "BathroomsTotalInteger",
    # "Stories",
    # "GarageSpaces",

    ## Output for training
    # store_data_in_parquet(df, path=pathfinder.ARTIFACTS_DIR)

    ########### To Johnny: I have not implemented the normalization steps because we don't know the right technique. If
    ########### possible, we can split up the continuous numerical features to identify it.
    if to_file:
        store_data_in_csv(df, path=pathfinder.OUTPUT_DIR)
    if split:
        y = df['log_price']
        df.drop('log_price', axis=1, inplace=True)
        return train_test_split_with_trimming(df=df, y=y, test_size=0.2, random_state=42)
    else:
        return df








