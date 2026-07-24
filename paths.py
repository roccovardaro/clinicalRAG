from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMG_DIR = DATASET_DIR/"images"
TXT_DIR = DATASET_DIR/"txt"

CSV_PATH = DATASET_DIR/"output.csv"

MODEL_DIR = BASE_DIR / "models"
BIOMEDCLIP_DIR = MODEL_DIR / "biomedCLIP"
MEDCPTCROSSENCODER_DIR = MODEL_DIR / "medcptcrossencoder"
