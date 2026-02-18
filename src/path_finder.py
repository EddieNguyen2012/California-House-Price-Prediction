from pathlib import Path
import os
# src/utils/path)_finder.py or inside your script
REPO_ROOT = Path(__file__).resolve().parents[1]   # adjust depth if needed
ARTIFACTS_DIR = REPO_ROOT / os.path.join("Data", "Artifacts")
DATABASE_DIR = REPO_ROOT / os.path.join("Data", "Warehouse")
OUTPUT_DIR = REPO_ROOT / "Output"
DATA_DIR = REPO_ROOT / "Data"
CSV_DIR = REPO_ROOT / "IDX_data"


