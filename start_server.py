"""
Start API Server Script

Simple script to start the FastAPI server with proper configuration.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pandas',
        'numpy'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✓ All required packages are installed")
    return True


def check_mock_models():
    """Check if mock models exist."""
    model_files = [
        "models/ensemble/xgboost_model.pkl",
        "models/ensemble/lightgbm_model.pkl",
        "data/training/training_dataset.csv"
    ]
    
    missing = []
    for file_path in model_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print("\n⚠️  Mock models not found:")
        for file_path in missing:
            print(f"   - {file_path}")
        print("\nCreate them with:")
        print("   python create_mock_models.py")
        return False
    
    print("✓ Mock models found")
    return True


def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server using uvicorn."""
    print("\n" + "=" * 60)
    print("Starting Player Transfer Value Prediction API")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check mock models
    models_exist = check_mock_models()
    if not models_exist:
        response = input("\nDo you want to create mock models now? (y/n): ")
        if response.lower() == 'y':
            print("\nCreating mock models...")
            subprocess.run([sys.executable, "create_mock_models.py"])
        else:
            print("\n⚠️  Starting server without models (some endpoints may not work)")
    
    print("\n" + "=" * 60)
    print(f"🚀 Starting server at http://{host}:{port}")
    print("=" * 60)
    print("\n📚 API Documentation:")
    print(f"   Swagger UI: http://localhost:{port}/docs")
    print(f"   ReDoc:      http://localhost:{port}/redoc")
    print("\n🔍 Endpoints:")
    print(f"   Root:       http://localhost:{port}/")
    print(f"   Health:     http://localhost:{port}/health")
    print(f"   Predict:    http://localhost:{port}/predict")
    print(f"   Batch:      http://localhost:{port}/predict/batch")
    print("\n💡 Test Player IDs: 12345, 67890, 11111, 22222, 33333")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        # Start the server
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("🛑 Server stopped")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
