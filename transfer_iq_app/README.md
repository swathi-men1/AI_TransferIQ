# Transfer IQ App

A machine learning application for transfer learning with a FastAPI backend and Streamlit frontend.

## Project Structure

```
transfer_iq_app/
├── README.md                 ← Full documentation
├── requirements.txt          ← All Python packages
├── backend/
│   └── api.py                ← FastAPI backend (port 8000)
├── frontend/
│   └── app.py                ← Streamlit frontend (port 8501)
├── utils/
│   ├── __init__.py           ← Package init
│   ├── data_utils.py         ← Data utilities
│   └── model_utils.py        ← ML model utilities
└── models/                   ← Saved models (auto-created)
```

## Installation

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run Both Backend & Frontend

**Terminal 1 - Start FastAPI Backend:**
```bash
python backend/api.py
```
Backend runs on `http://localhost:8000`

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run frontend/app.py
```
Frontend runs on `http://localhost:8501`

### Option 2: Run Streamlit Standalone (No Backend)
```bash
streamlit run frontend/app.py
```
The app will work with local model training/prediction if backend is unavailable.

## Features

- **Model Training**: Train ML models on uploaded data
- **Predictions**: Make predictions on new inputs
- **API Integration**: Backend API for scalable inference
- **Fallback Mode**: Works offline if backend unavailable

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API health check |
| POST | `/train` | Train model with provided data |
| POST | `/predict` | Make prediction on input |

### Example API Calls

**Train Model:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"data": {"column1": [1, 2, 3], "column2": [4, 5, 6]}}'
```

**Make Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"input": "your_input_data"}'
```

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Development Notes

- Models are saved to the `models/` directory
- Update `utils/data_utils.py` and `utils/model_utils.py` with your custom logic
- The frontend automatically falls back to local prediction if backend is unavailable
