# TransferIQ

TransferIQ is an AI-driven football transfer valuation project that predicts player transfer values using multi-source signals such as player performance, injury history, contract context, sentiment-related features, and historical transfer-market patterns.

The project combines feature engineering, time-series learning, ensemble modeling, evaluation reporting, and an interactive Streamlit dashboard in one end-to-end workflow. It is structured to support both academic evaluation and practical demonstration.

## Project Objective

The goal of TransferIQ is to estimate transfer values in a more dynamic and data-driven way than static market heuristics. The system is designed to:

- integrate performance, sentiment, injury, and market-context signals
- model temporal behavior using LSTM-based architectures
- compare multiple predictive approaches under a common evaluation pipeline
- provide an interactive interface for exploring player-level predictions

## Key Deliverables Covered

- Dynamic transfer-value prediction model based on multi-source data
- Trained `XGBoost`, multivariate `LSTM`, univariate `LSTM`, and encoder-decoder `LSTM` models
- Evaluation summary comparing model performance
- Sentiment-processing pipeline with VADER/TextBlob-compatible logic and fallback scoring
- Interactive Streamlit dashboard for player-level exploration and model insights
- Reproducible training, inference, and external-data workflow scripts

## Implemented Methodology

### 1. Data Collection And Preparation

The project includes structured workflows for:

- player-performance and transfer-context data processing
- market-value and historical transfer data usage
- sentiment-oriented text processing logic
- injury-data normalization

External-source workflow scripts are included for:

- StatsBomb-style open-data collection
- Transfermarkt-style HTML table scraping
- Twitter/X-style sentiment collection
- local injury-record normalization

These collection workflows are implemented at the script level and can be used with live services when credentials and network access are available. The core project logic also works without live APIs.

### 2. Preprocessing And Feature Engineering

The training pipeline performs:

- missing-value handling
- numerical scaling
- one-hot encoding for categorical fields
- date parsing and temporal feature derivation
- engineered features for performance, injury burden, contract status, mobility, competition context, and market pressure
- sentiment feature construction from structured inputs and optional text inputs

### 3. Model Development

The project currently includes:

- `XGBoost` regression model for structured tabular prediction
- multivariate `LSTM` for chronological sequence modeling
- univariate `LSTM` based on past target history
- encoder-decoder `LSTM` for multi-step forecasting across future windows
- weighted ensemble logic for combining tree-based and sequence-based models

### 4. Evaluation

The system evaluates models using:

- `RMSE`
- `MAE`
- `R2`
- `sMAPE`
- non-zero `RMSE`

Evaluation outputs are saved as artifacts and exposed in the dashboard.

## Latest Training Snapshot

Current saved metrics from `models/metadata/training_summary.json`:

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| XGBoost | 2,304,234 | 733,805 | 0.493 |
| Multivariate LSTM | 3,435,311 | 1,173,396 | -0.128 |
| Univariate LSTM | 3,466,874 | 1,247,202 | -0.149 |
| Encoder-Decoder LSTM | 3,490,903 | 1,261,501 | -0.150 |
| Ensemble | 2,304,234 | 733,805 | 0.493 |

Observations:

- the current best-performing model is `XGBoost`
- the latest ensemble run converged to the tree model weight, indicating the structured-feature model is currently strongest on this dataset
- LSTM-based models are fully implemented and evaluated, but their predictive performance is weaker on the present data split

## Repository Structure

```text
transfer_iq/
+-- app/
|   +-- app.py
|   +-- backend_predict.py
+-- config/
|   +-- .env.example
|   +-- config.yaml
|   +-- requirements.txt
+-- data/
|   +-- raw/
|   +-- processed/
+-- models/
|   +-- metadata/
|   +-- preprocessing/
|   +-- trained/
+-- scripts/
|   +-- bootstrap_external_services.py
|   +-- collect_external_data.py
|   +-- train_transfer_models.py
+-- src/
|   +-- sentiment_pipeline.py
|   +-- transfer_value_system.py
+-- setup.py
```

## Important Files

- [app/app.py](app/app.py): Streamlit dashboard and interactive prediction UI
- [app/backend_predict.py](app/backend_predict.py): terminal-based inference entrypoint
- [scripts/train_transfer_models.py](scripts/train_transfer_models.py): model training runner
- [scripts/collect_external_data.py](scripts/collect_external_data.py): external data-ingestion workflows
- [scripts/bootstrap_external_services.py](scripts/bootstrap_external_services.py): local external-service setup helper
- [src/transfer_value_system.py](src/transfer_value_system.py): shared feature engineering, training, forecasting, and inference logic
- [src/sentiment_pipeline.py](src/sentiment_pipeline.py): sentiment analysis utility with optional VADER/TextBlob support
- [models/metadata/training_summary.json](models/metadata/training_summary.json): saved evaluation summary

## Setup

### Environment

This project is configured to run from the local virtual environment inside the repository.

Install dependencies from:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\config\requirements.txt
```

### Optional External-Service Bootstrap

To prepare local config files and external-source folders:

```powershell
.\venv\Scripts\python.exe scripts\bootstrap_external_services.py
```

This creates a local `config/.env` scaffold from `config/.env.example` if it does not already exist.

## How To Run

### 1. Train Models

```powershell
.\venv\Scripts\python.exe scripts\train_transfer_models.py
```

### 2. Run Backend Prediction

```powershell
.\venv\Scripts\python.exe app\backend_predict.py --sample-library
```

### 3. Launch Streamlit Dashboard

```powershell
.\venv\Scripts\streamlit.exe run app\app.py
```

### 4. Run External Data Workflows

Bootstrap:

```powershell
.\venv\Scripts\python.exe scripts\bootstrap_external_services.py
```

Sync configured sources:

```powershell
.\venv\Scripts\python.exe scripts\collect_external_data.py --sync-all --write-manifest
```

Example StatsBomb-style collection:

```powershell
.\venv\Scripts\python.exe scripts\collect_external_data.py --statsbomb-competition-id 9 --statsbomb-season-id 281
```

## Interactive Dashboard Features

The Streamlit application includes:

- player search and individual valuation workspace
- editable player profile controls
- confidence-aware transfer-value estimate
- bulk scan mode for multiple players
- model intelligence view with feature importance and evaluation plots
- responsive layout for both laptop and mobile viewing

## Sentiment Analysis Support

The project includes a sentiment-processing pipeline that:

- supports `VADER` when installed
- supports `TextBlob` when installed
- falls back to a deterministic lexicon-based scorer when neither package is available

This design keeps the project runnable in evaluation environments while still demonstrating the intended NLP workflow.

## External Data Workflow Notes

TransferIQ includes logic for external data ingestion, but live collection depends on environment setup.

Examples:

- Twitter/X collection requires a bearer token if live API access is used
- Transfermarkt-style scraping requires a valid accessible source URL
- StatsBomb-style collection requires network access to public data endpoints

For mentor or academic evaluation, the project can still be assessed on the implemented logic, structure, training system, and reporting pipeline even without live API execution.

## Evaluation-Focused Summary

From an evaluation perspective, this repository demonstrates:

- end-to-end machine learning workflow design
- multi-source feature integration
- multiple forecasting model families
- comparative model evaluation
- sentiment-aware value estimation
- deployable interactive visualization
- reproducible scripts and saved artifacts

## Current Limitations

- LSTM-based models currently underperform the tree-based model on the available dataset
- live external-source execution depends on credentials and network access
- the strongest sentiment pipeline behavior is achieved when richer raw text data is available

## Future Improvements

- improve sequence modeling with richer player histories and grouped player timelines
- expand external-data ingestion into a scheduled ETL workflow
- strengthen sentiment inputs with higher-volume real text data
- tune encoder-decoder forecasting for stronger multi-window performance
- add automated tests for training and inference paths

