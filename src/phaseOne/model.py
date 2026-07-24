import json

import torch
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from paths import BIOMEDCLIP_DIR
import os

DEVICE = "mps" if torch.mps.is_available() else "cpu"



def download_and_save_model_locally():
    os.makedirs(BIOMEDCLIP_DIR, exist_ok=True)
    # Download the model and config files
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_pytorch_model.bin",
        local_dir=BIOMEDCLIP_DIR
    )
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_config.json",
        local_dir=BIOMEDCLIP_DIR
    )

def load_model():
    return load_model_locally(BIOMEDCLIP_DIR)

def load_model_locally(model_dir:str):

    model_name = "biomedclip_local"

    with open(f'''{model_dir}/open_clip_config.json''', "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)

    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=f'''{model_dir}/open_clip_pytorch_model.bin''',
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    return model, preprocess, tokenizer


