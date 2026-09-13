"""
Configuration module for Liver Cirrhosis Survival Prediction.

Centralizes paths, feature schemas, alias mappings, hyperparameter spaces,
and clinical research safety disclaimers.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "cirrhosis.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for p in [DATA_DIR / "raw", PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Medical & Research Safety Disclaimer
MEDICAL_DISCLAIMER = (
    "This application is for research and educational purposes only. "
    "It is not a medical device, does not provide medical advice, and must not be "
    "used for diagnosis, treatment, prognosis, or clinical decision-making."
)

# Canonical Column Names
TARGET_COLUMN = "Status"

NUMERIC_FEATURES = [
    "Age",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Triglycerides",
    "Platelets",
    "Prothrombin",
    "Stage",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema",
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that should not be used as clinical prediction biomarkers
# (e.g. follow-up duration 'time' / 'N_Days' would represent direct survival label leakage)
METADATA_COLUMNS = ["id", "rownames", "ID", "time", "N_Days", "trt", "Drug"]

# Canonical Target Classes
TARGET_CLASSES = {
    0: "Death",
    1: "Censored",
    2: "Transplant",
}

TARGET_ENCODING_REVERSE = {v: k for k, v in TARGET_CLASSES.items()}

# Common aliases across Mayo Clinic PBC / Kaggle / R datasets
ALIAS_MAP: Dict[str, str] = {
    # Lowercase R survival::pbc aliases
    "age": "Age",
    "sex": "Sex",
    "ascites": "Ascites",
    "hepato": "Hepatomegaly",
    "spiders": "Spiders",
    "edema": "Edema",
    "bili": "Bilirubin",
    "chol": "Cholesterol",
    "albumin": "Albumin",
    "copper": "Copper",
    "alk.phos": "Alk_Phos",
    "alk_phos": "Alk_Phos",
    "ast": "SGOT",
    "sgot": "SGOT",
    "trig": "Triglycerides",
    "platelet": "Platelets",
    "platelets": "Platelets",
    "protime": "Prothrombin",
    "stage": "Stage",
    "status": "Status",
    # Alternative naming conventions
    "hepatomegaly": "Hepatomegaly",
    "bilirubin": "Bilirubin",
    "cholesterol": "Cholesterol",
    "triglycerides": "Triglycerides",
    "prothrombin": "Prothrombin",
    "tryglicerides": "Triglycerides",
}

# Target status value mapping
# In R survival::pbc: 0 = censored, 1 = transplant, 2 = dead
# In Kaggle: 'D' = Dead, 'C' = Censored, 'CL' = Censored due to liver transplant
STATUS_VALUE_MAPPINGS: Dict[Any, int] = {
    # R dataset coding
    2: 0,  # Dead -> 0
    0: 1,  # Censored -> 1
    1: 2,  # Transplant -> 2
    # Float equivalents
    2.0: 0,
    0.0: 1,
    1.0: 2,
    # String equivalents
    "2": 0,
    "0": 1,
    "1": 2,
    "D": 0,
    "Death": 0,
    "dead": 0,
    "C": 1,
    "Censored": 1,
    "censored": 1,
    "CL": 2,
    "Transplant": 2,
    "transplant": 2,
}

# Physiological / Clinical Boundary Rules for Input Validation
PHYSIOLOGICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "Age": (1.0, 120.0),            # Years
    "Bilirubin": (0.0, 100.0),       # mg/dL (Normal ~0.1 - 1.2)
    "Cholesterol": (10.0, 3000.0),   # mg/dL
    "Albumin": (0.5, 10.0),          # g/dL (Normal ~3.4 - 5.4)
    "Copper": (0.0, 2500.0),         # ug/day
    "Alk_Phos": (0.0, 30000.0),      # U/L
    "SGOT": (0.0, 2000.0),           # U/mL (AST)
    "Triglycerides": (10.0, 2500.0), # mg/dL
    "Platelets": (1.0, 2000.0),      # x1000/uL
    "Prothrombin": (4.0, 50.0),      # Seconds
    "Stage": (1.0, 4.0),             # Histologic stage 1-4
}

VALID_CATEGORICAL_VALUES: Dict[str, List[str]] = {
    "Sex": ["F", "M", "f", "m"],
    "Ascites": ["0", "1", "0.0", "1.0", "N", "Y", "No", "Yes", 0, 1],
    "Hepatomegaly": ["0", "1", "0.0", "1.0", "N", "Y", "No", "Yes", 0, 1],
    "Spiders": ["0", "1", "0.0", "1.0", "N", "Y", "No", "Yes", 0, 1],
    "Edema": ["0", "0.5", "1", "0.0", "1.0", "N", "S", "Y", 0, 0.5, 1],
}

# Split & CV parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
PRIMARY_METRIC = "macro_f1"

# Remote dataset source fallback
REMOTE_DATASET_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv"
