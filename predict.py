# predict.py
import logging
from pathlib import Path
from datasets import load_dataset
import torch

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification
)
from utils import (
    load_model_params,
    load_data_paths,
    preprocess_dataset,
    save_predictions,
    cleanup_trainer_memory
)

# Fixed paths 
DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")
PRED_DIR = Path("predictions/")

def run_predict(model_params_path, models_to_run, dataset_split):
    """
    Run predictions using fine-tuned transformer models and save as CSV files.

    Args:
        model_params_path (Path): CSV file with model parameters.
        models_to_run (List[str]): List of models to load for predictions.
        dataset_split (str): Dataset split(s) to predict: 'dev', 'test', or 'both'.
    """

    # Mapping from KVL dataset split names to HF split names
    kvl_splits_to_hf = {
        "dev": "validation",
        "test": "test"
    }

    # Dataset_split to list of splits
    splits_to_run = ["dev", "test"] if dataset_split == "both" else [dataset_split]

    # Loop over models defined in the parameter CSV
    for row in load_model_params(model_params_path, models_to_run):
        model_name = row["model_name"]
        track = row["track"]
        l1 = row["L1"]
        l1_datasets = ["es", "de", "cn"] if l1 == "xx" else [l1]

        for l1 in l1_datasets:
            try:
                logging.info(
                    f"Predicting with model {model_name} on L1={l1}"
                )

                # Load dataset paths
                data_files = load_data_paths(DATA_DIR, l1, "predict", dataset_split=dataset_split)

                if not data_files:
                    logging.warning(f"No data files found for {l1} {dataset_split} set(s). Skipping this L1.")
                    continue

                hf_dataset = load_dataset("csv", data_files=data_files)

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(row["pretrained_model"], use_fast=True)
                cols_to_merge = row["component_order"].split("; ")
                sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "

                # Preprocess dataset: format input text, rename target label and remove extra columns
                preprocessed_ds = preprocess_dataset(hf_dataset, cols_to_merge, sep_token)

                # Tokenize dataset
                tokenized_ds = preprocessed_ds.map(
                    lambda x: tokenizer(x["input_text"], truncation=True),
                    batched=True,
                    desc="Tokenizing input text",
                )

                # Load the fine-tuned model
                model_path = MODELS_DIR / model_name
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.eval()

                # Initialize trainer
                data_collator = DataCollatorWithPadding(tokenizer)
                trainer = Trainer(
                    model=model,
                    data_collator=data_collator,
                    args=TrainingArguments(
                        output_dir=str(MODELS_DIR / model_name),
                        report_to="none"
                    ),
                )

                # Loop over requested dataset splits (dev / test)
                for split_name in splits_to_run:
                
                    logging.info(f"Getting {split_name} predictions...")
                
                    # Map KVL split name to Hugging Face split name
                    hf_key = kvl_splits_to_hf.get(split_name, split_name)
                
                    # Skip if this split is not available
                    if hf_key not in tokenized_ds:
                        logging.warning(f"No data found for {l1} {split_name} set. Skipping.")
                        continue
                
                    ds = tokenized_ds[hf_key]
                
                    # Run inference and flatten predictions
                    with torch.no_grad():
                        preds = trainer.predict(ds).predictions.flatten()
                
                    # Get item IDs to align with predictions
                    item_ids = hf_dataset[hf_key]["item_id"]
                
                    # Save predictions
                    save_dir = PRED_DIR / track / split_name / l1
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{model_name}_preds.csv"
                    save_predictions(save_path, item_ids, preds)
                    
                    logging.info(f"Saved predictions for {model_name} on {l1} {split_name} set: {save_path}")
                    
                # Free memory after predictions
                cleanup_trainer_memory(trainer, tokenized_ds, preprocessed_ds)

            except Exception:
                logging.exception(f"Predictions failed for {model_name} on {l1} {dataset_split} set(s)")
                raise