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
'City', 'CountyOrParish', 'HighSchoolDistrict', 'Latitude', 'Longitude',
'MLSAreaMajor', 'PostalCode', 'StateOrProvince', 'StreetNumberNumeric',

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

def normalize(series: pd.Series, technique):
    if technique == "minmax":
        normalizer = MinMaxScaler()
        return normalizer.fit_transform(series)
    elif technique == "standard":
        normalizer = StandardScaler()
        return normalizer.fit_transform(series)
    else:
        raise ValueError("Unknown normalization type")


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

    ### Engineered features
    # PostalCode -> zipcode_parse()
    # CloseDate -> cyclical_encoding()
    # ClosePrice -> zipcode_parse()

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








