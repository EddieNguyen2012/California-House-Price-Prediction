"""
Data_Cleaning.py

Contains all the functions for imputing and normalizing data. A sort of scrapyard of different methods

Ordered by:
    - Helper functions
    - In-use functions
    - Work-in-progress functions
    - Temporarily unused functions

"""
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import src.path_finder as pathfinder
import joblib
"""============================== HELPER ================================"""

def normalize(df: pd.DataFrame, target_col: str | list, technique: str):
    """
    Johnny's general-purpose function for normalizing column(s) of numeric data.
    Can choose between different techniques and add more if desired.

    Available techniques: 
    - minmax
    - standard
    - robust

    Returns normalized column. Example of intended usage: 
    df[columns] = normalize(df, columns, technique) 
    """
    t = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }
    try:
        normalizer = t[technique]
    except: 
        raise ValueError("Unknown normalization type")
    return normalizer.fit_transform(df[target_col])

def impute(df: pd.DataFrame, target_col: list, technique: str, fill_value=0):
    '''
    Building on Jenny's and Huiyu's work, Johnny's general-purpose function for imputing column(s)
    Can choose between different techniques and add more if desired.

    Available techniques:
    - (numeric) 'mean', 'median', 'most_frequent', 'knn' 
    - (boolean works with column of any data type) 'boolean' 
    - (categorical) 'most_frequent' 

    If technique is "knn", optional paramaters are 
    - knn_n_neighbors: if technique is 'knn', this indicates n_neighbors
    - knn_weights: select from 'uniform', 'distance'
    
    Returns imputed column. Example of intended usage:
    df[columns] = impute(df, columns, technique, ...)
    '''

    imputer = None

    simple_metrics = ["median","mean","most_frequent","constant"]
    if technique in simple_metrics:
        if technique == "constant":
            imputer = SimpleImputer(strategy=technique, fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=technique)
    elif technique == "boolean":
        return bool_encode(df, target_col)
    else:
        return None
    
    return imputer.fit_transform(df[target_col])

def bool_encode(df: pd.DataFrame, target_col: str | list):
    '''
    Johnny's improved bool_encode function (to handle errors in data)
    Converts "TRUE" to 1 and "FALSE"/missing/invalid types to 0.
    Edited to match input/output of impute()
    '''
    mapping = {"TRUE": True, "FALSE": False}
    
    if isinstance(target_col, str):
        return (
            df[target_col]
            .map(mapping)
            .astype("boolean")
            .fillna(False)
            .astype(int)
        )
    else:
        return (
            df[target_col]
            .apply(lambda col: (
                col.map(mapping)
                   .astype("boolean")
                   .fillna(False)
                   .astype(int)
            ))
        )
    
"""============================== IN USE ================================"""

def baseline_impute_normalize(df: pd.DataFrame):
    """
    Imputes and normalizes all data for baseline testing (linear regression).

    Numerics are imputed by 'median' and normalized with 'robust'
    Booleans are imputed by 'boolean'
    Categoricals are imputed by 'most_frequent'
    """
    df = df.copy()
    base_features = [
        'Latitude',
        'Longitude',
        'BedroomsTotal',
        'LivingArea',
        'sin_closed_date',
        'cos_closed_date',
        'DaysOnMarket'
    ]
    random_forest_impute_targets = ['YearBuilt', 'BedroomsTotal', 'BathroomsTotalInteger']
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = list(set(numeric_cols) - set(random_forest_impute_targets) - {'AssociationFee'})

    # Use random forest imputer
    df = random_forest_imputer(df, base_features, random_forest_impute_targets, strat='classifier')

    # Set all NA of HOA fees to 0
    df['AssociationFee'] = impute(df, target_col=['AssociationFee'], technique="constant", fill_value=0)
    # Median imputation of numeric columns
    df[numeric_cols] = impute(df, target_col=numeric_cols, technique="median")

    # RobustScaler to normalize numeric columns
    # df[numeric_cols] = normalize(df, target_col=numeric_cols, technique="robust")

    # Turns columns to boolean (also imputes missing/invalid values as False)
    boolean_cols = ['AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN', 'PoolPrivateYN', 'ViewYN']
    df[boolean_cols] = impute(df, target_col=boolean_cols, technique="constant", fill_value=False)


    return df

def random_forest_imputer(df: pd.DataFrame, features, y, strat='classifier'):
    df = df.copy()
    for ind_y in y:
        model = get_imputer_artifacts(ind_y)
        if model is None:
            model = create_imputer_artifacts(df, features, ind_y, strat=strat)

        # Predict on the same 5 features from the main dataframe
        prediction = model.predict(df[features].values)

        # Efficiently fill NaNs without a manual loop
        df[ind_y] = df[ind_y].fillna(pd.Series(prediction, index=df.index))

    return df

def get_imputer_artifacts(name):
    artifact_path = pathfinder.ARTIFACTS_DIR / f'{name}_imputer_model.joblib'
    if os.path.exists(artifact_path):
        return joblib.load(artifact_path)
    return None

def create_imputer_artifacts(df: pd.DataFrame, features, y, strat='classifier'):
    df_train = df[df[y].notna()].copy()

    # Extract target and training data
    target = df_train[y].values
    train = df_train[features].values  # USES ORIGINAL 5 FEATURES

    if strat == 'classifier':
        model = RandomForestClassifier()
    elif strat == 'regressor':
        model = RandomForestRegressor()
    else:
        raise Exception(f"Unknown imputation technique: {strat}")

    # Train on 5 features
    model.fit(train, target)
    joblib.dump(model, filename=pathfinder.ARTIFACTS_DIR / f'{y}_imputer_model.joblib', compress=3)
    return model

"""============================= WORK IN PROGRESS ==================================="""

"""============================ TEMPORARILY UNUSED ==================================="""

# Huiyu: Helper function for imputation
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


# Huiyu: Helper function to get categorical feature indices by column names
def get_cat_feature_indices(X: pd.DataFrame, cat_cols: list[str]):
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

# Huiyu: ColumnTransformer preprocessor

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


