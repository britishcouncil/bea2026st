# utils.py
import random
import gc
import logging
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
from transformers import logging as hf_logging

def configure_logging(verbose=False):
    """
    Configure logging for the pipeline.

    Args:
        verbose (bool): If True, set log level to DEBUG; else INFO.
    """
    # Determine log file path 
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"run_{timestamp}.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    # Add handlers
    for handler in (ch, fh):
        root.addHandler(handler)

    # Hugging Face logging
    if verbose:
        hf_logging.set_verbosity_debug()
    else:
        hf_logging.set_verbosity_warning()

    hf_logger = hf_logging.get_logger("transformers")
    hf_logger.propagate=False
    hf_logger.handlers.clear()
    for handler in (ch, fh):
        hf_logger.addHandler(handler)
        

def is_model_downloaded(model_folder):
    """
    Check if the Hugging Face model is downloaded.
    
    Args:
        model_folder (Path): Local path to model folder.
    
    Returns:
        bool: True if all required files exist, False otherwise.
    """

    REQUIRED_FILES = [
        "training_args.bin", 
        "tokenizer.json", 
        "special_tokens_map.json", 
        "tokenizer_config.json",
        "model.safetensors",
        "config.json",
    ]
    
    if not model_folder.exists():
        return False
    
    missing_files = [f for f in REQUIRED_FILES if not (model_folder / f).exists()]
    if missing_files:
        return False
    
    return True



def load_model_params(model_params_path, models_to_run):
    """
    Load the model parameters CSV and yield each row as a dict.

    Args:
        model_params_path (Path or str): Path to the CSV file containing model parameters.
        models_to_run (List[str]): List of model names to include.
    """
    with Path(model_params_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get("model_name")
            if model_name not in models_to_run:
                continue
            yield row


def load_data_paths(data_dir, l1, mode, dataset_split=None):
    """
    Loads CSV file paths for Hugging Face `load_dataset`.

    Returns a dictionary with keys 'train', 'validation', and 'test',
    mapping to lists of CSV paths. Keys with no files are omitted.

    Args:
        data_dir (Path): Base directory containing the data splits.
        l1 (str): Language code (e.g., 'es') or 'xx' to include all languages during finetune.
        mode (str): One of 'finetune', 'predict', or 'evaluate'.
        dataset_split (str or None): For predict/evaluate, which data splits to load:
            'dev', 'test', or 'both'. Ignored for finetune.

    Returns:
        dict: Mapping HF split names to lists of CSV paths, e.g.,
            {'train': ['train/es/file1.csv'], 'validation': ['dev/es/file1.csv']}
    """

    # for finetuning 'xx' means combine all L1s
    # L1s are separate for predict and evaluate
    l1_list = ["es", "de", "cn"] if mode == "finetune" and l1 == "xx" else [l1]

    data_files = {}

    # Mapping from KVL dataset split names to HF split names
    kvl_splits_to_hf = {
        "train": "train",
        "dev": "validation",
        "test": "test"
    }

    if mode == "finetune":
        train_files = []
        val_files = []

        for l1 in l1_list:
            train_folder=data_dir / "train" / l1
            val_folder=data_dir / "dev" / l1

            train_files.extend(str(f) for f in train_folder.glob("*.csv"))
            val_files.extend(str(f) for f in val_folder.glob("*.csv"))

        if train_files:
            data_files["train"] = train_files
        if val_files:
            data_files["validation"] = val_files

    elif mode in {"predict", "evaluate"}:
        
        splits_to_load = ["dev", "test"] if dataset_split == "both" else [dataset_split]

        for folder_name in splits_to_load:
            files = []
            for l1 in l1_list:
                folder = data_dir / folder_name / l1
                files.extend(str(f) for f in folder.glob("*.csv"))

            hf_key = kvl_splits_to_hf[folder_name]
            if files:
                data_files[hf_key] = files

    return data_files


def merge_cols(batch, cols_to_merge, sep_token):
    """
    Merge specified item text components into a single input string per example.

    Designed for use with Hugging Face `Dataset.map` in batched mode.

    Args:
        batch (dict): Batched dataset examples.
        cols_to_merge (list[str]): Ordered column names to concatenate.
        sep_token (str): Separator string between components.

    Returns:
        dict: Dictionary with key `"input_text"` containing merged text strings.
    """
    rows=zip(*(batch[col] for col in cols_to_merge))
    return {
        "input_text": [
            sep_token.join(str(x).strip() for x in row)
            for row in rows
        ]
    }


def preprocess_dataset(ds_dict, cols_to_merge, sep_token):
    """
    Preprocess a Hugging Face DatasetDict by:
        1. Merging specified text columns into a single 'input_text' column.
        2. Renaming the target column 'GLMM_score' to 'label'.
        3. Removing all other columns except 'input_text' and 'label'.

    Args:
        ds_dict (datasets.DatasetDict): Input DatasetDict with one or more splits.
        cols_to_merge (list[str]): List of text columns to concatenate into 'input_text'.
        sep_token (str): Separator string to insert between merged columns.

    Returns:
        datasets.DatasetDict: Preprocessed DatasetDict where each split contains only
            the columns 'input_text' and 'label', ready for tokenization.
    """
    
    # Compute columns to remove
    first_split = next(iter(ds_dict.values())) # get first key in ds_dict
    all_columns = first_split.column_names
    cols_to_keep = ["input_text", "label"]
    cols_to_remove = [c for c in all_columns if c not in cols_to_keep and c != 'GLMM_score']
    
    # Format input text, rename target label and remove extra columns
    ds_dict = ds_dict.map(
        merge_cols,
        batched=True,
        fn_kwargs={"cols_to_merge": cols_to_merge, "sep_token": sep_token},
        desc="Formatting input text"
    ).rename_column("GLMM_score", "label").remove_columns(cols_to_remove)
    
    return ds_dict


def compute_metrics(eval_pred):
    """
    Compute regression performance metrics for Hugging Face Trainer evaluation.

    Args:
        eval_pred (tuple): Tuple of (predictions, labels).

    Returns:
        dict: Dictionary containing RMSE and Pearson correlation.
    """
    predictions, labels=eval_pred
    predictions = predictions.flatten() # shape (num_examples,)
    rmse = root_mean_squared_error(labels, predictions)
    p_corr, _ = pearsonr(predictions, labels)
    
    return {"rmse": rmse, "pearson": p_corr}


def print_evaluation_results(eval_results_df, decimals=3):
    """
    Print evaluation results for both tracks.

    Args:
        eval_results_df (pd.DataFrame): DataFrame with columns 'model', 'track', 'L1', and metric columns.
        decimals (int): Number of decimal places to round metrics.
    """

    # Ensure L1 is ordered as es, de, cn
    L1_order = ["es", "de", "cn"]
    eval_results_df["L1"] = pd.Categorical(eval_results_df["L1"], categories=L1_order, ordered=True)

    tables = [
        ("CLOSED TRACK", eval_results_df[eval_results_df["track"] == "closed"]),
        (
            "OPEN TRACK",
            eval_results_df[(eval_results_df["track"] == "open") & (eval_results_df["L1"] != "xx")],
        ),
    ]

    output_lines = []
    cols = ["model", "L1", "rmse", "pearson"]

    for title, df in tables:
        # Sort by L1 according to our categorical order
        df=df.sort_values("L1") 

        output_lines.extend([
            "",
            title,
            "-" * len(title),
        ])

        if df.empty:
            output_lines.append("<no results>")
        else:
            output_lines.append(df[cols].round(decimals).to_string(index=False))

    output_text = "\n".join(output_lines)

    logging.info(f"Evaluation summary:\n{output_text}")



def save_predictions(save_path, item_ids, predictions):
    """
    Save predictions to a CSV file, sorted by item_id.

    Args:
        save_path (Path): Path to save the CSV file.
        item_ids (list): List of item IDs corresponding to predictions.
        predictions (list or array): List of predicted values.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a DataFrame and sort by item_id
    df = pd.DataFrame({"item_id": item_ids, "prediction": predictions})
    df = df.sort_values("item_id")

    # Save to CSV
    df.to_csv(save_path, index=False)


def cleanup_trainer_memory(*objects):
    """
    Free up memory used by PyTorch and delete given objects.

    Args:
        *objects: Any Python objects (e.g., Trainer, datasets) to delete.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    for obj in objects:
        del obj