from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

app = FastAPI(title="Transfer IQ API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    data: dict

class PredictRequest(BaseModel):
    input: str

@app.get("/")
def read_root():
    return {"message": "Transfer IQ API is running", "status": "healthy"}

@app.post("/train")
def train_endpoint(request: TrainRequest):
    """Train model endpoint - stores training data for ML pipeline."""
    try:
        # TODO: Implement actual model training logic here
        # For now, just echo back success
        data_size = len(str(request.data))
        return {
            "status": "success",
            "message": "Model training initiated",
            "data_size_chars": data_size,
            "note": "Model training implementation coming soon"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    """Make predictions on player value."""
    try:
        # TODO: Implement actual prediction logic here
        # For now, return dummy prediction
        # Parse input format: "name-role-team"
        parts = request.input.split('-')
        
        # Simple mock prediction based on role
        base_value = {
            "batsman": 5000000,
            "bowler": 3000000,
            "all-rounder": 4000000,
            "unknown": 2000000
        }
        
        role = parts[1].lower() if len(parts) > 1 else "unknown"
        prediction = base_value.get(role, 2000000)
        
        return {
            "status": "success",
            "prediction": prediction,
            "input_received": request.input,
            "note": "Mock prediction - actual ML model coming soon"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Transfer IQ API server....")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)
