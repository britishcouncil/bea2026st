# run_pipeline.py
import argparse
import logging
import warnings
from pathlib import Path

from download import download_models
from finetune import run_finetune
from predict import run_predict
from evaluate import run_evaluate

from utils import configure_logging

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models/")
BASELINE_MODELS = ["baseline_closed_es", "baseline_closed_de", "baseline_closed_cn", "baseline_open_xx"]

def main():
    parser = argparse.ArgumentParser(description="Run download, predict, and evaluate pipeline, with optional fine-tuning.")

    # Optional flags to run steps
    parser.add_argument("--download", action="store_true",help="Download baseline models from Hugging Face Hub")
    parser.add_argument("--finetune", action="store_true", help="Run finetuning step")
    parser.add_argument("--predict", action="store_true", help="Run prediction step")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation step")

    # Configurable options
    parser.add_argument(
        "--models_to_run",
        nargs="+",
        default=BASELINE_MODELS,
        help=f"Baseline models to use. Options: {', '.join(BASELINE_MODELS)}"
    )
    parser.add_argument(
        "--model_params_path",
        type=Path,
        default=MODELS_DIR / "model_parameters.csv",
        help="Path to model parameters CSV"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="dev",
        choices=["dev", "test", "both"],
        help="Dataset splits to use for prediction and evaluation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=10, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args=parser.parse_args()
    configure_logging(args.verbose)

    # Determine which steps to run
    steps=[]
    if args.download: steps.append("download")
    if args.finetune: steps.append("finetune")
    if args.predict: steps.append("predict")
    if args.evaluate: steps.append("evaluate")

    # Default: run all steps if no flags were provided
    if not steps:
        steps = ["download", "predict", "evaluate"]
        
    # Run download
    if "download" in steps:
        logging.info("=== Downloading models ===")
        download_models(
            models_to_run=args.models_to_run
        )    

    # Run finetuning
    if "finetune" in steps:
        logging.info("=== Running finetune ===")
        run_finetune(
            model_params_path=args.model_params_path,
            models_to_run=args.models_to_run,
            seed=args.seed
        )

    # Run predictions
    if "predict" in steps:
        logging.info("=== Running predict ===")
        run_predict(
            model_params_path=args.model_params_path,
            models_to_run=args.models_to_run,
            dataset_split=args.dataset_split,
        )

    # Run evaluation
    if "evaluate" in steps:
        logging.info("=== Running evaluate ===")
        run_evaluate(
            dataset_split=args.dataset_split,
        )

if __name__ == "__main__":
    main()