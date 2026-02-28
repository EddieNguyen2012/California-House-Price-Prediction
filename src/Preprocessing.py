import pandas as pd
import numpy as np
import src.path_finder as pathfinder
from src.Ingestion import DataIngestion
from sklearn.model_selection import train_test_split
import os
##### List of usable columns from Jenny's EDA
columns_to_use = [
# --- Agent & Office Info ---
'BuyerAgentAOR', 'BuyerAgentFirstName', 'BuyerAgentLastName', 'BuyerAgentMlsId',
'BuyerOfficeAOR', 'BuyerOfficeName', 'ListAgentAOR', 'ListOfficeName',

# --- Location & Address ---
'City', 'CountyOrParish', 'HighSchoolDistrict', 'Latitude', 'Longitude',
'MLSAreaMajor', 'PostalCode', 'StateOrProvince', 'StreetNumberNumeric',
'UnparsedAddress',

# --- Property Specs ---
'AttachedGarageYN', 'BathroomsTotalInteger', 'BedroomsTotal', 'FireplaceYN',
'Flooring', 'GarageSpaces', 'Levels', 'LivingArea', 'MainLevelBedrooms',
'NewConstructionYN', 'ParkingTotal', 'PoolPrivateYN', 'PropertySubType',
'PropertyType', 'Stories', 'ViewYN', 'YearBuilt',

# --- Lot Information ---
'LotSizeAcres', 'LotSizeArea', 'LotSizeSquareFeet',

# --- Transaction & Status ---
'AssociationFee', 'CloseDate', 'ClosePrice', 'DaysOnMarket', 'MlsStatus',
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


### Huiyu: ColumnTransformer preprocessor
def build_sklearn_preprocessor(
    X: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols = None,
    scale_numeric: bool = True,
):
    """
    This is a preprocessing ColumnTransformer:
    - numeric: median impute (+ optional scaling)
    - categorical: most_frequent impute + one-hot
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if numeric_cols is None:
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

    num_imputer = build_imputer("median")
    cat_imputer = build_imputer("most_frequent")

    num_steps = [("imputer", num_imputer)]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


####### Jenny (median imputation)
def compute_medians(df: pd.DataFrame) -> pd.Series:
    cols_impute = [
        "LivingArea",
        "LotSizeAcres",
        "LotSizeArea",
        "LotSizeSquareFeet",
        "BathroomsTotalInteger",
        "Stories",
        "GarageSpaces",
    ]
    
    cols_impute = [col for col in cols_impute if col in df.columns]
    
    return df[cols_impute].median()

def impute_with_medians(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df[medians.index] = df[medians.index].fillna(medians)
    return df
#######

if __name__ == "__main__":
    accessor = DataIngestion(pathfinder.CSV_DIR)

    ## Read required data
    df = get_unprocessed_data(accessor= accessor)
    # print(df.columns)
    df['PostalCode'] = df['PostalCode'].apply(zipcode_parse)

    df['engineered_closed_date'] = df['CloseDate'].apply(cyclical_encoding)
    df.drop('CloseDate', axis=1, inplace=True)

    medians = compute_medians(df)
    df = impute_with_medians(df, medians)

    df['log_price'] = df['ClosePrice'].apply(lambda x: np.log(x))  # Transform close price by logging


    ## Output for training
    # store_data_in_parquet(df, path=pathfinder.ARTIFACTS_DIR)
    store_data_in_csv(df, path=pathfinder.OUTPUT_DIR)







