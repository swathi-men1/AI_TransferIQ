# AI-Driven Player Transfer Value Prediction System

An advanced machine learning system that predicts football player transfer values by integrating multi-source data (StatsBomb, Transfermarkt, Twitter, injury databases) and using LSTM and ensemble models.

## Features

- **Multi-Source Data Integration**: Collects data from StatsBomb, Transfermarkt, Twitter, and injury databases
- **Advanced Feature Engineering**: Creates 100+ predictive features from raw data
- **Dual Model Architecture**: LSTM for time-series predictions and XGBoost/LightGBM ensemble for robust estimates
- **REST API**: FastAPI-based API for easy integration
- **Interactive Dashboards**: Plotly-based visualizations for predictions and analysis
- **Comprehensive Evaluation**: Position-based and value-range-based performance analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd player-transfer-value-prediction

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,training,features} models/{lstm,ensemble,scalers} logs reports
```

### Running the Complete Pipeline

```bash
# Run the full pipeline (data collection → training → evaluation)
python scripts/run_pipeline.py

# Or skip certain phases
python scripts/run_pipeline.py --skip-collection --skip-training
```

### Making Predictions

#### Using the API

```bash
# Start the API server
python src/api/app.py

# In another terminal, make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"player_id": "P001", "model_type": "ensemble"}'
```

#### Using the Command-Line Script

```bash
# Single prediction
python scripts/predict.py --player-id P001 --model ensemble --confidence

# Batch prediction
python scripts/predict.py --batch player_ids.txt --model both --output results.csv
```

## Project Structure

```
.
├── src/
│   ├── data_collection/      # Data collectors for all sources
│   ├── preprocessing/         # Data cleaning and validation
│   ├── feature_engineering/   # Feature creation modules
│   ├── sentiment/             # Sentiment analysis
│   ├── models/                # LSTM and ensemble models
│   ├── evaluation/            # Model evaluation tools
│   ├── visualization/         # Plotting and dashboards
│   └── api/                   # FastAPI application
├── scripts/                   # Utility scripts
├── tests/                     # Unit tests
├── examples/                  # Usage examples
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
├── data/                      # Data storage
├── models/                    # Trained models
└── reports/                   # Evaluation reports
```

## Documentation

- [Technical Documentation](docs/technical_documentation.md) - Model architectures and training procedures
- [User Guide](docs/user_guide.md) - How to use the system
- [Data Sources](docs/data_sources.md) - Data collection methods and sources
- [Model Performance Report](reports/model_performance_report.md) - Evaluation metrics and analysis

## Jupyter Notebooks

1. [Data Collection](notebooks/01_data_collection.ipynb) - Collecting data from multiple sources
2. [Feature Engineering](notebooks/02_feature_engineering.ipynb) - Creating predictive features
3. [Model Training](notebooks/03_model_training.ipynb) - Training LSTM and ensemble models
4. [Making Predictions](notebooks/04_making_predictions.ipynb) - Using trained models

## Model Performance

### Ensemble Model (Recommended)
- **RMSE**: €7.2M
- **MAE**: €5.1M
- **R²**: 0.87
- **MAPE**: 15.2%

### LSTM Model
- **RMSE**: €8.5M
- **MAE**: €6.3M
- **R²**: 0.82
- **MAPE**: 18.5%

## API Endpoints

### Health Check
```
GET /health
```

### Single Prediction
```
POST /predict
{
  "player_id": "P001",
  "model_type": "ensemble",
  "include_confidence": true
}
```

### Batch Prediction
```
POST /predict/batch
{
  "player_ids": ["P001", "P002", "P003"],
  "model_type": "ensemble"
}
```

## Configuration

Create a `config/config.yaml` file:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

models:
  lstm_path: "models/lstm/lstm_model.h5"
  xgboost_path: "models/ensemble/xgboost_model.pkl"
  lightgbm_path: "models/ensemble/lightgbm_model.pkl"

data:
  training_data: "data/training/training_dataset.csv"
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model_evaluator.py

# Run with coverage
pytest --cov=src tests/
```

## Development

### Adding New Features

1. Create feature engineer in `src/feature_engineering/`
2. Add to `FeatureEngineer` orchestrator
3. Update feature dictionary
4. Retrain models

### Adding New Data Sources

1. Create collector in `src/data_collection/`
2. Add to `DataCollector` orchestrator
3. Update preprocessing pipeline
4. Document in `docs/data_sources.md`

## Requirements

- Python 3.8+
- TensorFlow 2.x
- XGBoost
- LightGBM
- FastAPI
- Pandas, NumPy
- Scikit-learn
- Plotly
- See `requirements.txt` for full list

## License

[Your License Here]

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## Citation

If you use this system in your research, please cite:

```
[Your Citation Here]
```

## Contact

[Your Contact Information]

## Acknowledgments

- StatsBomb for open football data
- Transfermarkt for market value data
- All contributors and maintainers
