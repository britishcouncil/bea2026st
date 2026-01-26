#download.py
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from utils import is_model_downloaded

MODELS_DIR = Path("models/")
HF_USER = "lucyskidmore"

def download_models(models_to_run):
    """
    Download models from Hugging Face Hub to local folder.

    Args:
        models_to_run (list[str]): List of model names to download.
    """

    for model_name in models_to_run:
        
        dest_folder = MODELS_DIR / model_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        if is_model_downloaded(dest_folder):
            logging.info(f"{model_name} already downloaded, skipping.")
            continue

        logging.info(f"Downloading {model_name}...")    
        
        # Download all files from the HF repo into the local folder
        snapshot_download(
            repo_id=f"{HF_USER}/{model_name}",
            local_dir=str(dest_folder),
            local_dir_use_symlinks=False
        )
        logging.info(f"Saved {model_name} to {dest_folder}\n")