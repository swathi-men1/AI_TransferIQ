# 🚀 Quick Start - API Server

Get your API server running in 3 simple steps!

## Step 1: Create Mock Models (30 seconds)

```bash
python create_mock_models.py
```

This creates:
- Mock XGBoost and LightGBM models
- Mock player dataset with 100+ players
- Test player IDs: `12345`, `67890`, `11111`, `22222`, `33333`

## Step 2: Start the Server (5 seconds)

```bash
python start_server.py
```

Or use uvicorn directly:
```bash
uvicorn src.api.app:app --reload
```

## Step 3: Test It! (1 minute)

### Option A: Open Swagger UI (Easiest)
1. Open your browser
2. Go to: **http://localhost:8000/docs**
3. Click on `/predict` endpoint
4. Click "Try it out"
5. Enter player_id: `12345`
6. Click "Execute"
7. See the prediction! 🎉

### Option B: Use the Test Script
```bash
python test_api_endpoints.py
```

### Option C: Use cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"player_id": "12345", "model_type": "ensemble", "include_confidence": true}'
```

## 🎯 What You Get

Once running, you have:

✅ **REST API** at `http://localhost:8000`
- Single predictions: `POST /predict`
- Batch predictions: `POST /predict/batch`
- Health check: `GET /health`

✅ **Interactive Documentation**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

✅ **Test Data**
- 100+ mock players
- 5 test player IDs ready to use

## 📊 Example Response

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

## 🔧 Troubleshooting

### "Module not found" error?
```bash
pip install fastapi uvicorn pydantic pandas numpy
```

### Port 8000 already in use?
```bash
python start_server.py --port 8001
```

### Want to stop the server?
Press `CTRL+C` in the terminal

## 🎨 Next Steps

1. **Build a Frontend**: Create a React/Vue app that calls your API
2. **Train Real Models**: Replace mock models with trained ones
3. **Add Authentication**: Secure your API with API keys
4. **Deploy**: Host on AWS, Azure, or Heroku

## 📚 Full Documentation

See `START_API_SERVER.md` for complete documentation.

---

**Ready?** Run: `python start_server.py` 🚀
