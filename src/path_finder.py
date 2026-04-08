from pathlib import Path
import os
import joblib

# src/utils/path)_finder.py or inside your script
REPO_ROOT = Path(__file__).resolve().parents[1]   # adjust depth if needed
ARTIFACTS_DIR = REPO_ROOT / os.path.join("Data", "Artifacts")
DATABASE_DIR = REPO_ROOT / os.path.join("Data", "Warehouse")
OUTPUT_DIR = REPO_ROOT / "Output"
DATA_DIR = REPO_ROOT / "Data"
CSV_DIR = REPO_ROOT / "IDX_data"


def get_imputer_artifacts(name, type):
    """
    This module provides a function to retrieve pre-trained machine learning models
    from a designated artifacts directory.

    Parameters:
    - name (str): The name of the model or artifact.
    - type (str): The type of model or artifact ('imputer', 'classifier', 'regressor', 'encoder').

    Returns:
    - The loaded model artifact if found, otherwise None.
    """
    artifact_path = ARTIFACTS_DIR / f'{name}_{type}_model.joblib'
    if os.path.exists(artifact_path):
        print(f'Found pre-trained {type} model for {name}. Returning artifact...')
        return joblib.load(artifact_path)
    print(f'Not found pre-trained {type} model for {name}.')
    return None


def create_artifacts(model, name, type):
    """
    This function creates and saves a model artifact to a specified directory.

    Args:
        model: The model object to be saved.
        name: The name of the model.
        type: The type of the model.
    """
    joblib.dump(model, filename=ARTIFACTS_DIR/f'{name}_{type}_model.joblib', compress=3)
    print(f'Created and saved {type} model for {name} at {ARTIFACTS_DIR/f"{name}_{type}_model.joblib"}')
    return model