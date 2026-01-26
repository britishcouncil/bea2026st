from pathlib import Path
import pandas as pd
import logging
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

from utils import print_evaluation_results

# Fixed paths 
DATA_DIR = Path("data/")
PRED_DIR = Path("predictions/")
RESULTS_DIR = Path("results/")

def run_evaluate(dataset_split):
    """
    Evaluate prediction CSV files against the labelled datasets and save results.

    Args:
        dataset_split (str): Which subset(s) to predict: 'dev', 'test', or 'both'.
    """
    
    eval_results = []
    pred_paths = PRED_DIR.rglob("*.csv")
        
    # Dataset_split to list of splits
    splits_to_run = ["dev", "test"] if dataset_split == "both" else [dataset_split]

    # Loop over each prediction file
    for pred_path in pred_paths:
        
        try:
            # Compute relative path to infer track, split, L1, filename
            rel = pred_path.relative_to(PRED_DIR)
            try:
                track, split, l1, filename = rel.parts
            except ValueError:
                raise ValueError(
                    f"Bad prediction path: {pred_path}.\n"
                    "Predictions must be saved at predictions/<track>/<split>/<l1>/"
                )
                
            if split not in splits_to_run:
                continue
            
            # Infer model name from {model_name}_preds.csv
            stem = pred_path.stem
            if not stem.endswith("_preds"):
                logging.warning(f"Skipping file with unexpected name: {filename}")
                continue

            model_name = stem.replace("_preds", "")
    
            logging.info(f"Evaluating pred file {filename} on {l1} {split} set...")

            # Load labels
            labels_path = DATA_DIR / split / l1 / f"kvl_shared_task_{l1}_{split}.csv"
            labels_df = pd.read_csv(labels_path)

            # Load predictions
            preds_df = pd.read_csv(pred_path)

            # Ensure prediction file has required columns
            required_cols = {"item_id", "prediction"}
            missing = required_cols - set(preds_df.columns)
            if missing:
                raise ValueError(f"Missing column(s) {missing}: {pred_path}")

            # Merge labels with predictions on item_id
            merged_df = labels_df.merge(
                preds_df[["item_id", "prediction"]],
                on="item_id",
                how="left"
            )

            # Compute metrics and store results
            labels = merged_df["GLMM_score"].values
            predictions = merged_df["prediction"].values

            eval_results.append({
                "track": track,
                "split": split,
                "model": model_name,
                "L1": l1,
                "rmse": root_mean_squared_error(labels, predictions),
                "pearson": pearsonr(predictions, labels)[0],
            })

        except Exception:
            logging.exception(f"Failed to evaluate {pred_path}")
            raise
        
    if not eval_results:
        logging.warning("No prediction files found for the requested dataset split(s).")
        return  

    # Convert results to DataFrame and sort by L1
    df = pd.DataFrame(eval_results)
    L1_order = ["es", "de", "cn"]
    df["L1"] = pd.Categorical(df["L1"], categories=L1_order, ordered=True)
    df = df.sort_values(["track", "model", "L1"])

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSVs grouped by split and print summaries
    for split, split_df in df.groupby("split"):
        out_path = RESULTS_DIR / f"results_summary_{split}.csv"
        split_df.to_csv(out_path, index=False)
        print_evaluation_results(split_df)
        logging.info(f"Saved evaluation results to {out_path}")