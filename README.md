# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions from anonymized transaction features. The project includes exploratory analysis, preprocessing utilities, model training, threshold tuning, saved model artifacts, and validation metrics suitable for highly imbalanced fraud data.

## Overview

Credit card fraud detection is an imbalanced binary classification problem: most transactions are legitimate, while fraud cases are rare and costly to miss. This project trains a soft-voting ensemble on the standard credit card fraud feature format:

- `Time`
- PCA-anonymized features `V1` through `V28`
- `Amount`
- Target label `Class`, where `0` means legitimate and `1` means fraud

The current training pipeline combines:

- Logistic Regression with balanced class weights
- Random Forest with balanced class weights
- Multi-Layer Perceptron classifier
- Optional resampling with SMOTE or random undersampling
- Threshold search to maximize validation F1-score
- PR-AUC evaluation for imbalanced classification

## Repository Structure

```text
Credit_Card_Fraud_Detection/
|-- data/
|   |-- train.csv
|   |-- val.csv
|   |-- test.csv
|   `-- trainval.csv
|-- models/
|   `-- fraud_model.pkl
|-- notebooks/
|   `-- Credit_Card_eda.ipynb
|-- results/
|   |-- best_threshold.txt
|   `-- evaluation_results.json
|-- src/
|   |-- __init__.py
|   |-- config.json
|   |-- credit_fraud_train.py
|   |-- credit_fraud_utils_data.py
|   `-- credit_fraud_utils_eval.py
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Dataset Summary

The included CSV files follow the same schema and contain 31 columns: 30 input features plus the target label.

| File | Rows | Legitimate | Fraud |
| --- | ---: | ---: | ---: |
| `data/train.csv` | 170,884 | 170,579 | 305 |
| `data/val.csv` | 56,960 | 56,870 | 90 |
| `data/test.csv` | 56,960 | 56,863 | 97 |
| `data/trainval.csv` | 56,960 | 56,870 | 90 |

The class imbalance is severe, so accuracy alone is not a reliable metric. The project focuses on F1-score and PR-AUC.

## Installation

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- imbalanced-learn

## Training

The main training entrypoint is:

```bash
src/credit_fraud_train.py
```

The current script reads `config.json` from the current working directory. A configuration file is provided at `src/config.json`, so run from the project root with:

```bash
cp src/config.json config.json
python src/credit_fraud_train.py
```

The provided configuration trains three Random Forest variants inside the voting ensemble:

```json
{
  "train_path": "data/train.csv",
  "val_path": "data/val.csv",
  "results_dir": "results/experiments",
  "sampling": "smote",
  "hyperparameters": [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 15}
  ]
}
```

Each experiment saves:

- Metrics JSON under `results/experiments/`
- A pickled model artifact under `models/`

## Pipeline Details

### Data Loading

`src/credit_fraud_utils_data.py` loads CSV files with pandas and expects a `Class` column in every supervised dataset.

### Preprocessing

The preprocessing function:

1. Splits features from the `Class` label.
2. Applies `StandardScaler` to all feature columns.
3. Optionally applies a sampling strategy on the training set.

Supported sampling modes:

- `smote`
- `undersampling`
- no sampling, by leaving the sampling method empty

Validation data is scaled but not resampled.

### Model

`src/credit_fraud_train.py` trains three estimators:

- `LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")`
- `RandomForestClassifier(class_weight="balanced")`
- `MLPClassifier(hidden_layer_sizes=(16, 32), early_stopping=True)`

These are combined with:

```python
VotingClassifier(voting="soft")
```

Soft voting uses predicted probabilities from the base models, which also enables threshold tuning.

### Evaluation

`src/credit_fraud_utils_eval.py` computes:

- F1-score
- PR-AUC through `average_precision_score`
- Best threshold by scanning thresholds from `0.00` to `0.99`

## Current Results

The checked-in validation results are:

| Metric | Value |
| --- | ---: |
| F1-score at default model threshold | 0.8045 |
| PR-AUC | 0.8050 |
| Best threshold | 0.46 |
| Best F1-score | 0.8177 |

These values come from:

- `results/evaluation_results.json`
- `results/best_threshold.txt`

## Inference Example

The existing `models/fraud_model.pkl` artifact contains a dictionary with:

- `model`
- `scaler`
- `threshold`

Example usage:

```python
import pickle
import pandas as pd

with open("models/fraud_model.pkl", "rb") as file:
    artifact = pickle.load(file)

model = artifact["model"]
scaler = artifact["scaler"]
threshold = artifact["threshold"]

new_transactions = pd.read_csv("data/test.csv").drop(columns=["Class"])
new_transactions_scaled = scaler.transform(new_transactions)

fraud_probability = model.predict_proba(new_transactions_scaled)[:, 1]
fraud_prediction = (fraud_probability >= threshold).astype(int)
```

Note: newly generated experiment models from the current training script save `model` and `threshold`. If you need production-ready inference from newly trained artifacts, persist the fitted scaler alongside the model as well.

## Exploratory Data Analysis

The notebook `notebooks/Credit_Card_eda.ipynb` contains analysis for:

- Dataset shape and column inspection
- Class imbalance
- Feature distributions
- Correlations
- Fraud vs legitimate transaction patterns

## Outputs

Existing outputs:

- `models/fraud_model.pkl`: saved model artifact
- `results/evaluation_results.json`: validation F1-score and PR-AUC
- `results/best_threshold.txt`: tuned threshold and best F1-score

Training outputs from the current pipeline:

- `results/experiments/experiment_<n>.json`
- `models/fraud_model_exp_<n>.pkl`

## Important Notes

- The dataset is extremely imbalanced; use F1-score, recall, precision, and PR-AUC rather than accuracy alone.
- The validation and test sets must not be resampled.
- Threshold tuning should be done on validation data only.
- The current training script expects `config.json` in the current working directory.
- For reproducible inference, save the scaler used during training with every model artifact.
- This project is for educational and research purposes. A production fraud system would also need monitoring, drift detection, model governance, privacy controls, and compliance review.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
