# Employee Attrition MLOps Pipeline

An end-to-end machine learning operations project for predicting employee attrition.

The project demonstrates a reproducible ML workflow using:

- scikit-learn for preprocessing and model training
- DVC and DagsHub for dataset versioning and remote storage
- MLflow for experiment tracking
- pytest for automated testing and data/model validation
- GitHub Actions for continuous integration
- Evidently for production data drift monitoring

## Project Overview

The goal of this project is to predict whether an employee will leave a company using employee demographic, job, compensation, and workplace information.

This is a binary classification problem where the target variable is:

```text
Attrition
```

Possible target values are:

```text
Yes
No
```

The dataset contains 1,470 rows and 35 columns. It contains both numerical and categorical features.

The original dataset did not contain missing values, so missing values were intentionally introduced into selected features to test the preprocessing pipeline's ability to handle incomplete data.

Missing values were introduced into:

- Age
- MonthlyIncome
- YearsAtCompany
- JobRole
- Department

## Project Structure

```text
employee-attrition-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── config.yaml
│   ├── experiment_1.yaml
│   ├── experiment_2.yaml
│   ├── experiment_3.yaml
│   ├── experiment_4.yaml
│   └── experiment_5.yaml
│
├── data/
│   └── employee_attrition.csv.dvc
│
├── reports/
│   └── drift_report.html
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── monitor_drift.py
│   └── simulate_missing.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── compare_experiments.py
├── MONITORING.md
├── requirements.txt
└── README.md
```

The generated model, MLflow database, raw CSV dataset, and HTML monitoring report are excluded from Git where appropriate.

## Installation

Clone the repository:

```bash
git clone https://github.com/sourceak/employee-attrition-mlops.git
cd employee-attrition-mlops
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Versioning with DVC

The training dataset is tracked using DVC instead of Git.

The Git repository contains:

```text
data/employee_attrition.csv.dvc
```

The actual CSV is stored remotely using DagsHub.

The configured DVC remote is:

```text
https://dagshub.com/sourceak/employee-attrition-mlops.dvc
```

After cloning the repository and installing the dependencies, retrieve the dataset with:

```bash
dvc pull
```

The dataset version used during the current training runs has the DVC hash:

```text
5b6c79a5bd908b48b4e0139ccdc733ba
```

Training also records this DVC hash in MLflow so model experiments can be associated with the exact dataset version used.

## Configuration

The main configuration is stored in:

```text
configs/config.yaml
```

It controls:

- Dataset path
- Target column
- Train/test split
- Random seed
- Random Forest hyperparameters
- MLflow experiment name
- Model output path
- Primary evaluation metric
- Minimum acceptable accuracy
- Data drift threshold

This allows important training and monitoring settings to be changed without modifying the Python source code.

## Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocess.py
```

Numerical features use median imputation for missing values.

Categorical features use:

1. Most-frequent-value imputation
2. One-hot encoding

The preprocessing steps and Random Forest classifier are combined into a scikit-learn Pipeline so the same transformations are consistently applied during training and prediction.

## Model Training

The project uses a:

```text
RandomForestClassifier
```

Train the default model with:

```bash
python -m src.train
```

A specific experiment configuration can be supplied with:

```bash
python -m src.train --config configs/experiment_1.yaml
```

The trained model is saved locally to:

```text
models/attrition_model.joblib
```

The `models/` directory is excluded from Git.

## MLflow Experiment Tracking

MLflow tracks each training run.

For every run, the training process records:

- Model hyperparameters
- Train/test configuration
- DVC dataset version
- Accuracy
- Precision
- Recall
- F1 score
- Trained scikit-learn model artifact

At least five different experiment configurations are provided in the `configs/` directory.

The five deliberate experiment configurations produced accuracy results around:

| Experiment | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1 | 0.8435 | 0.5455 | 0.1277 | 0.2069 |
| 2 | 0.8503 | 0.6667 | 0.1277 | 0.2143 |
| 3 | 0.8469 | 0.6250 | 0.1064 | 0.1818 |
| 4 | 0.8469 | 0.6250 | 0.1064 | 0.1818 |
| 5 | 0.8503 | 0.6667 | 0.1277 | 0.2143 |

Accuracy is configured as the primary metric.

Because the positive `Attrition=Yes` class is less common, accuracy alone does not fully describe model quality. Precision, recall, and F1 are also recorded for every run.

### Compare Experiments

Run:

```bash
python compare_experiments.py
```

The script uses:

```python
mlflow.search_runs()
```

to retrieve the experiment runs and identify the run with the highest configured primary metric.

To inspect runs through the MLflow interface, run:

```bash
mlflow ui
```

## Model Evaluation

Run:

```bash
python -m src.evaluate
```

The evaluation script calculates:

- Accuracy
- Precision
- Recall
- F1 score

The configured minimum accuracy is:

```text
0.75
```

If model accuracy falls below this threshold, the evaluation process exits with a non-zero status. This allows the same performance requirement to be enforced automatically in CI.

## Automated Testing

The project uses pytest for preprocessing, data validation, and model validation.

Run all tests with:

```bash
pytest -v
```

The current test suite contains 12 tests.

### Preprocessing Tests

Tests verify:

- Feature/target splitting
- Original DataFrame is not modified
- Numerical missing-value imputation
- Categorical encoding
- Invalid input handling
- Missing target handling
- Empty feature handling

### Data Validation Tests

Tests verify:

- Required columns exist
- Target values are valid
- Important numerical features remain within expected ranges

### Model Validation Tests

Tests verify:

- Prediction output shape and values
- Model accuracy meets the minimum required performance

Current local result:

```text
12 passed
```

## Continuous Integration

GitHub Actions runs the automated MLOps pipeline on:

- Pushes to `main`
- Pull requests targeting `main`

The workflow contains two jobs:

```text
Run Tests
    ↓
Train and Validate Model
```

The training job depends on the testing job using `needs: test`.

The CI pipeline:

1. Checks out the repository
2. Sets up Python
3. Installs dependencies
4. Configures secure DagsHub/DVC authentication
5. Pulls the versioned dataset using DVC
6. Runs pytest
7. Trains the model
8. Evaluates the trained model
9. Fails if model performance is below the configured threshold

DagsHub credentials are stored as GitHub Actions repository secrets and are not committed to the repository.

A successful GitHub Actions pipeline run has been completed for this project.

## Data Drift Monitoring

Production drift monitoring is implemented using Evidently.

Run:

```bash
python -m src.monitor_drift
```

The training dataset is used as reference data. The script creates simulated production data with intentional distribution changes.

The simulation changes:

- Age
- MonthlyIncome
- JobRole

The most recent monitoring result was:

```text
Total features: 34
Drifted features: 3
Overall drift share: 8.82%

Features with detected drift:
- Age
- MonthlyIncome
- JobRole

Allowed drift threshold: 20.00%

PASSED: Drift is within the configured threshold.
```

The threshold is configured in:

```text
configs/config.yaml
```

The Evidently HTML report is generated at:

```text
reports/drift_report.html
```

The report itself is generated locally and excluded from Git.

If the overall share of drifted features exceeds the configured threshold, the monitoring process exits with a non-zero status.

See `MONITORING.md` for an interpretation of the drift results, potential model impact, and recommended response.

## Reproducing the Pipeline

After cloning the repository:

```bash
pip install -r requirements.txt
dvc pull
pytest -v
python -m src.train
python -m src.evaluate
python -m src.monitor_drift
```

To run the five supplied experiment configurations:

```bash
python -m src.train --config configs/experiment_1.yaml
python -m src.train --config configs/experiment_2.yaml
python -m src.train --config configs/experiment_3.yaml
python -m src.train --config configs/experiment_4.yaml
python -m src.train --config configs/experiment_5.yaml
python compare_experiments.py
```

## Technologies

- Python
- pandas
- scikit-learn
- DVC
- DagsHub
- MLflow
- pytest
- Evidently
- Git
- GitHub
- GitHub Actions

## Reproducibility

The project combines configuration files, DVC dataset versioning, deterministic random seeds, MLflow experiment tracking, automated testing, and continuous integration to make the machine learning workflow reproducible and testable.