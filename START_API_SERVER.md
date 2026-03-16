# 🚀 Starting the API Server - Quick Guide

## Prerequisites

Before starting the API server, you need to:

### 1. Install Dependencies

```bash
# Install required packages
pip install fastapi uvicorn python-multipart pydantic

# Install ML libraries (if not already installed)
pip install tensorflow xgboost lightgbm scikit-learn pandas numpy joblib
```

### 2. Prepare Mock Models (For Testing)

Since you haven't trained the models yet, let's create mock models for testing:

```bash
# Run the mock model generator
python create_mock_models.py
```

## 🎯 Starting the Server

### Option 1: Using Uvicorn (Recommended)

```bash
# Start the server with auto-reload
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using Python directly

```bash
# Run the app module
python -m src.api.app
```

### Option 3: Using the startup script

```bash
# Run the startup script
python start_server.py
```

## 📡 Accessing the API

Once the server is running, you can access:

### 1. **Interactive API Documentation (Swagger UI)**
```
http://localhost:8000/docs
```
- Try out endpoints directly in your browser
- See request/response schemas
- Test predictions interactively

### 2. **Alternative API Documentation (ReDoc)**
```
http://localhost:8000/redoc
```
- Clean, readable documentation
- Better for reading and understanding

### 3. **Root Endpoint**
```
http://localhost:8000/
```
- Basic API information
- Available endpoints

### 4. **Health Check**
```
http://localhost:8000/health
```
- Check if models are loaded
- Verify API status

## 🧪 Testing the API

### Using cURL (Command Line)

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "12345",
    "model_type": "ensemble",
    "include_confidence": true
  }'
```

#### 3. Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "player_ids": ["12345", "67890", "11111"],
    "model_type": "ensemble",
    "include_confidence": true
  }'
```

### Using Python Requests

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Single prediction
prediction_data = {
    "player_id": "12345",
    "model_type": "ensemble",
    "include_confidence": True
}
response = requests.post(f"{BASE_URL}/predict", json=prediction_data)
print(response.json())

# Batch prediction
batch_data = {
    "player_ids": ["12345", "67890"],
    "model_type": "ensemble",
    "include_confidence": True
}
response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
print(response.json())
```

### Using the Test Script

```bash
# Run the comprehensive test script
python test_api_endpoints.py
```

## 📊 API Endpoints

### GET `/`
- **Description**: Root endpoint with API information
- **Response**: API metadata and available endpoints

### GET `/health`
- **Description**: Health check endpoint
- **Response**: 
  ```json
  {
    "status": "healthy",
    "models_loaded": true
  }
  ```

### POST `/predict`
- **Description**: Predict transfer value for a single player
- **Request Body**:
  ```json
  {
    "player_id": "string",
    "model_type": "ensemble" | "lstm" | "xgboost" | "lightgbm",
    "include_confidence": true
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "result": {
      "player_id": "12345",
      "predicted_value": 50000000.0,
      "confidence_interval": {
        "lower": 45000000.0,
        "upper": 55000000.0,
        "confidence_level": 0.95
      },
      "model_used": "ensemble",
      "timestamp": "2026-03-11T10:30:00"
    },
    "message": "Prediction successful"
  }
  ```

### POST `/predict/batch`
- **Description**: Predict transfer values for multiple players
- **Request Body**:
  ```json
  {
    "player_ids": ["12345", "67890"],
    "model_type": "ensemble",
    "include_confidence": true
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "results": [...],
    "failed_predictions": [],
    "total_requested": 2,
    "total_successful": 2,
    "message": "Successfully predicted 2/2 players"
  }
  ```

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Models not found" warnings
**Solution**: Either train models or create mock models
```bash
# Create mock models for testing
python create_mock_models.py

# OR train real models
python scripts/train_lstm.py
python scripts/run_pipeline.py --train-ensemble
```

### Issue: "Port 8000 already in use"
**Solution**: Use a different port
```bash
uvicorn src.api.app:app --reload --port 8001
```

### Issue: "Player data not found"
**Solution**: Generate training dataset
```bash
python src/data_preparation/generate_dataset.py
```

## 🎨 Next Steps

1. **Open Swagger UI**: Visit `http://localhost:8000/docs`
2. **Try the health check**: Click on `/health` → "Try it out" → "Execute"
3. **Make a prediction**: Click on `/predict` → "Try it out" → Enter player_id → "Execute"
4. **View the response**: See the predicted transfer value and confidence interval

## 📝 Notes

- The API runs on `http://localhost:8000` by default
- Auto-reload is enabled in development mode (changes to code will restart the server)
- All prediction requests are logged for audit purposes
- CORS is enabled for all origins (configure for production)

## 🔒 Production Deployment

For production deployment, consider:

1. **Disable auto-reload**: Remove `--reload` flag
2. **Configure CORS**: Restrict allowed origins
3. **Add authentication**: Implement API key or OAuth
4. **Use HTTPS**: Set up SSL certificates
5. **Add rate limiting**: Prevent abuse
6. **Set up monitoring**: Track API performance
7. **Use a production server**: Gunicorn with Uvicorn workers

```bash
# Production example
gunicorn src.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

**Ready to start?** Run: `uvicorn src.api.app:app --reload`
