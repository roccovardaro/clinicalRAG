import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from paths import MEDCPTCROSSENCODER_DIR
import os
DEVICE = "mps" if torch.mps.is_available() else "cpu"


def download_and_save_model_locally():
    os.makedirs(MEDCPTCROSSENCODER_DIR, exist_ok=True)
    # Download the model and config files
    hf_hub_download(
        repo_id="ncbi/MedCPT-Cross-Encoder",
        filename="pytorch_model.bin",
        local_dir=MEDCPTCROSSENCODER_DIR
    )
    hf_hub_download(
        repo_id="ncbi/MedCPT-Cross-Encoder",
        filename="config.json",
        local_dir=MEDCPTCROSSENCODER_DIR
    )

def load_model():
    return load_local_model(MEDCPTCROSSENCODER_DIR)


def load_local_model(model_path: str):
    """
    Carica tokenizer e modello da una directory locale.

    Args:
        model_path (str): percorso alla cartella contenente i file del modello

    Returns:
        tokenizer, model
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )

    return model, tokenizer