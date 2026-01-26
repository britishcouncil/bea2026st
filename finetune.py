# finetune.py
import logging
from datasets import load_dataset
from pathlib import Path

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    set_seed,
)

from utils import ( 
    compute_metrics, 
    load_model_params,
    load_data_paths,
    preprocess_dataset,
    cleanup_trainer_memory
)

# Fixed paths 
DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")

def run_finetune(model_params_path, models_to_run, seed):
    """
    Fine-tune pre-trained transformer models based on a CSV of parameters.

    Args:
        model_params_path (Path): CSV file with model parameters and metadata.
        models_to_run (List[str]): List of models to finetune.
        seed (int, optional): Random seed.
    """
    
    # Loop over models defined in the parameter CSV
    for row in load_model_params(model_params_path, models_to_run):
        
        model_name = row["model_name"]
        l1 = row["L1"]

        try:
            logging.info(f"Fine-tuning model: {model_name}...")

            # Load dataset paths and Hugging Face DatasetDict
            data_files = load_data_paths(DATA_DIR, l1, "finetune")
            hf_dataset = load_dataset("csv", data_files=data_files)
    
            # Load tokenizer and prepare input text formatting
            tokenizer = AutoTokenizer.from_pretrained(row["pretrained_model"], use_fast=True)
            cols_to_merge = row["component_order"].split("; ")
            sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "
    
            # Preprocess dataset: format input text, rename target label and remove extra columns
            preprocessed_ds = preprocess_dataset(hf_dataset, cols_to_merge, sep_token)
            
            # Tokenize dataset
            tokenized_ds = preprocessed_ds.map(
                lambda x: tokenizer(x["input_text"], truncation=True),
                batched=True,
                desc="Tokenizing input text"
            )

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(MODELS_DIR / model_name),
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                num_train_epochs=int(row["epochs"]),
                per_device_train_batch_size=int(row["batch_size"]),
                per_device_eval_batch_size=int(row["batch_size"]),
                learning_rate=float(row["learning_rate"]),
                weight_decay=float(row["weight_decay"]),
                warmup_ratio=float(row["warmup_ratio"]),
                load_best_model_at_end=True,
                report_to="none",
                seed=seed,
            )

            # Initialise trainer
            data_collator = DataCollatorWithPadding(tokenizer)
            trainer = Trainer(
                model_init=lambda: AutoModelForSequenceClassification.from_pretrained(
                    row["pretrained_model"], num_labels=1
                ),
                args=training_args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["validation"],
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            # Finetune model and save
            trainer.train()
            trainer.save_model(MODELS_DIR / model_name)
            logging.info(f"Model {model_name} fine-tuned and saved at {str(MODELS_DIR / model_name)}")
            
            # Free memory after training
            cleanup_trainer_memory(trainer, tokenized_ds, preprocessed_ds)
        
        except Exception:
            logging.exception(f"Failed model {model_name}")
            raise