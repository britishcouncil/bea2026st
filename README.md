# BEA 2026 Shared Task: Vocabulary Difficulty Prediction for English Learners

This repository provides the datasets, fine-tuned transformer baseline models, and evaluation scripts for the [British Council's Shared Task at BEA 2026](https://www.britishcouncil.org/data-science-and-insights/bea2026st).

---

## Quick Start (evaluate predictions)
```bash
git clone https://github.com/britishcouncil/bea2026st
cd bea2026st
conda env create -f environment.yml
conda activate baseline_env
python run_pipeline.py --evaluate
```

**Expected output:**

```text
CLOSED TRACK
------------
             model L1  rmse  pearson
baseline_closed_es es 1.357    0.748
baseline_closed_de de 1.328    0.753
baseline_closed_cn cn 1.175    0.736

OPEN TRACK
----------
           model L1  rmse  pearson
baseline_open_xx es 1.206    0.787
baseline_open_xx de 1.149    0.800
baseline_open_xx cn 1.021    0.804
```

---

## Directory Structure

```text
.
├── data/
│   ├── train/
│   │   ├── es/kvl_shared_task_es_train.csv
│   │   ├── de/kvl_shared_task_de_train.csv
│   │   ├── cn/kvl_shared_task_cn_train.csv
│   ├── dev/
│   │   ├── es/kvl_shared_task_es_dev.csv
│   │   ├── de/kvl_shared_task_de_dev.csv
│   │   ├── cn/kvl_shared_task_cn_dev.csv
│   ├── test/
│   │   ├── es/kvl_shared_task_es_test.csv
│   │   ├── de/kvl_shared_task_de_test.csv
│   │   ├── cn/kvl_shared_task_cn_test.csv
├── models/
│   ├── baseline_closed_es/
│   ├── baseline_closed_de/
│   ├── baseline_closed_cn/
│   ├── baseline_open_xx/
│   ├── model_parameters.csv
├── predictions/
│   ├── open/
│   │   ├── dev/
│   │   |   ├── es/baseline_open_xx_preds.csv
│   │   |   ├── de/baseline_open_xx_preds.csv
│   │   |   ├── cn/baseline_open_xx_preds.csv
│   ├── closed/
│   │   ├── dev/
│   │   |   ├── es/baseline_closed_es_preds.csv
│   │   |   ├── de/baseline_closed_de_preds.csv
│   │   |   ├── cn/baseline_closed_cn_preds.csv
├── results/
│   ├── results_summary_dev.csv
├── logs/
├── finetune.py           
├── predict.py            
├── evaluate.py           
├── utils.py              
├── run_pipeline.py
├── environment.yml       
└── README.md
```

* `data/` Provided datasets (CSV files), split by dataset type (`train`, `dev`, `test`) and L1 (`es`, `de`, `cn`).
* `models/` Folders for fine-tuned baseline models. There are three separate model folders for the `closed` track and one for the `open` track that is trained jointly on all L1s. Due to their size, models must be downloaded from the [Hugging Face Hub](https://huggingface.co/lucyskidmore/models) (instructions to follow).
  * `model_parameters.csv` - Metadata and hyperparameters for each baseline model.
* `predictions/` Model predictions (CSV files) organized by track type (`open`, `closed`), dataset split (`train`, `dev`), and L1 (`es`, `de`, `cn`).
* `results/results_summary_dev.csv` Evaluation outputs (CSV files) reporting RMSE and Pearson correlation for all models and L1s, evaluated on the `dev` sets.
* `logs/` Folder for timestamped log files.
* Scripts:
  * `download.py` Downloads HF transformer models.
  * `finetune.py` Fine-tunes models.
  * `predict.py` Generates predictions.
  * `evaluate.py` Evaluates predictions against dataset `GLMM_score` labels.
  * `run_pipeline.py` Runs the full pipeline (download → predict → evaluate), with optional fine-tuning.
  * `utils.py` Shared helper functions.

<small>**Note:** The baseline models were fine-tuned following [Skidmore et al. (2025)](https://doi.org/10.18653/v1/2025.bea-1.12), but **without target variable scaling** and **using Pearson correlation** for evaluation. As a result, the reported results are similar but not identical to the original study. </small>

---

## Data

The training, development and test datasets are provided as a set of CSV files, one for each L1, which include the following data columns:

* `item_id` An ID number from 1 to 6,768. Items with the same item_id across different L1 files are parallel (i.e., refer to the same English target word).
* `L1` The L1 of the prompt (`es` for Spanish, `de` for German, or `cn` for Mandarin).
* `en_target_word` The English target word.
* `en_target_pos` The part of speech of the English target word.
* `en_target_clue` A partial-spelling clue of the English target word.
* `L1_source_word` The corresponding L1 source word(s).
* `L1_context` The L1 contextualising prompt.
* `GLMM_score` The GLMM difficulty estimate for the vocabulary test item, as reported by [Schmitt et al. (2024)](https://www.britishcouncil.org/research-insight/knowledge-based-vocabulary-lists), where a **lower score indicates a more difficult word**. This is the target value that will be predicted. **Note:** The `GLMM_score` is not provided for the test datasets. 
---

## Model Parameters File

The parameters file `model_parameters.csv` contains metadata and hyperparameters for each of the baseline models. It is used both for fine-tuning and for generating predictions.

**Metadata Columns**

* `model_name`: Name of the model. This is used both for the folder where the fine-tuned model is saved and for the corresponding prediction and result files.
* `track`: Either `open` or `closed`.  
* `pretrained_model`: Hugging Face model to load for fine-tuning.  
* `L1`: Language code. Can be `es`, `de`, `cn`, or `xx` for models trained on all L1s jointly.  
* `component_order`: Semicolon-separated list of the dataset column names (corresponding to components of the vocabulary test item text) that should be concatenated to form the model input text.   

**Model Hyperparameters**

* `batch_size`
* `learning_rate`
* `weight_decay`
* `warmup_ratio`
* `epochs`

---

## Installation & Setup

This project uses the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library for vocabulary difficulty prediction. 

We recommend using **Conda** to manage dependencies and ensure GPU compatibility.

1. **Clone the repository** to your local machine:

```bash
git clone https://github.com/britishcouncil/bea2026st
cd bea2026st
```

2. **Create and activate the Conda environment** using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate baseline_env
```

**Note:** The baseline models are **not included** in the GitHub repo. They can be downloaded directly from Hugging Face ([see baseline models](https://huggingface.co/lucyskidmore/models)) or when you run the pipeline (see instructions below).

---

## Run the full pipeline

```bash
python run_pipeline.py
```

By default, this runs the `--download`, `--predict` `--evaluate` steps sequentially for all baseline models.

1. **Download:** The fine-tuned baseline models are downloaded from the Hugging Face Hub and saved in the `models` directory.

2. **Predict**: The models are loaded and used to predict on the specified `--dataset_split` (default: `dev`). Predictions are saved to:
`predictions/{track}/{dataset_split}/{L1}/{model_name}_preds.csv`.

3. **Evaluate**: Predictions are compared to the dataset labels. Results are saved as a CSV to:
`results/results_summary_{dataset_split}.csv` and also printed to the console. **Note:** Evaluate cannot be run on the test dataset as the `GLMM_score` is not provided.

**Run individual steps:**

You can run one or more steps separately using the flags: `--download` `--finetune`, `--predict`, `--evaluate`. For example, to fine-tune the models from scratch and make new predictions:

```bash
python run_pipeline.py --finetune --predict
```

**Run a subset of baseline models:**

Use `--models_to_run` to select specific baseline models. This affects the `--download`, `--finetune`, and `--predict` steps. For example:

```bash
python run_pipeline.py --models_to_run baseline_closed_cn baseline_closed_es
```

**Note:** Running `--predict` requires the corresponding fine-tuned model to be available locally. If the model is not already in `models/`, you can:  

1. Download it manually from the Hugging Face hub ([see models](https://huggingface.co/lucyskidmore/models)).  
2. Use `--download --models_to_run <model_name>` to fetch it.  
3. Fine-tune the model from scratch with `--finetune --models_to_run <model_name>`.  
4. Or simply run the full pipeline via `run_pipeline.py --models_to_run <model_name>`.
 

**All optional arguments:**

* `--models_to_run` Baseline models to include in pipeline  (`baseline_closed_es`, `baseline_closed_de`, `baseline_closed_cn` or `baseline_open_xx`; default: all models).
* `--model_params_path` Path to model parameters CSV (default: `models/model_parameters.csv`).
* `--dataset_split` Dataset split for prediction/evaluation (`dev`, `test`, or `both`; default: `dev`).
* `--seed` Random seed for reproducibility (default: `10`).
* `--verbose`: Enable verbose logging for debugging; prints DEBUG messages in addition to INFO (default: `False`).

---

## Evaluate baseline model predictions

As baseline model predictions are already provided in the folders, you can skip directly to evaluation:

```bash
python run_pipeline.py --evaluate
```

**Expected results for the provided baseline models:**

```text
CLOSED TRACK
------------
             model L1  rmse  pearson
baseline_closed_es es 1.357    0.748
baseline_closed_de de 1.328    0.753
baseline_closed_cn cn 1.175    0.736

OPEN TRACK
----------
           model L1  rmse  pearson
baseline_open_xx es 1.206    0.787
baseline_open_xx de 1.149    0.800
baseline_open_xx cn 1.021    0.804
```
---

## Evaluate additional prediction files

`run_pipeline.py --evaluate` automatically evaluates **all prediction CSVs** in the `predictions/` folder. To evaluate predictions for any additional models you develop, simply place their CSV files in the appropriate folder structure before running:

```bash
predictions/{track}/{dataset_split}/{L1}/{your_model_name}_preds.csv
```

**Requirements for prediction files:**

* Must follow naming convention `{your_model_name}_preds.csv` 
* Must include the columns: `item_id` and `prediction`.
