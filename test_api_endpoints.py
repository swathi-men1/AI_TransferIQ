"""
Test API Endpoints

This script tests all the API endpoints to verify they're working correctly.
"""

import requests
import json
import time
from typing import Dict, Any
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test player IDs
TEST_PLAYER_IDS = ["12345", "67890", "11111", "22222", "33333"]


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def print_success(text: str):
    """Print success message."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text: str):
    """Print error message."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text: str):
    """Print info message."""
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")


def print_response(response: Dict[Any, Any]):
    """Print formatted JSON response."""
    print(f"{Fore.WHITE}{json.dumps(response, indent=2)}{Style.RESET_ALL}")


def test_root_endpoint():
    """Test the root endpoint."""
    print_header("Testing Root Endpoint (GET /)")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            print_response(response.json())
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to API server")
        print_info("Make sure the server is running: uvicorn src.api.app:app --reload")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_health_endpoint():
    """Test the health check endpoint."""
    print_header("Testing Health Check (GET /health)")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            data = response.json()
            print_response(data)
            
            if data.get('models_loaded'):
                print_success("Models are loaded")
            else:
                print_info("Models not loaded - run create_mock_models.py first")
            
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_single_prediction():
    """Test single player prediction."""
    print_header("Testing Single Prediction (POST /predict)")
    
    player_id = TEST_PLAYER_IDS[0]
    
    payload = {
        "player_id": player_id,
        "model_type": "ensemble",
        "include_confidence": True
    }
    
    print_info(f"Request payload:")
    print_response(payload)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            data = response.json()
            print_response(data)
            
            if data.get('success'):
                result = data.get('result', {})
                predicted_value = result.get('predicted_value', 0)
                print_success(f"Predicted Value: €{predicted_value:,.2f}")
                
                ci = result.get('confidence_interval')
                if ci:
                    print_success(f"Confidence Interval: €{ci['lower']:,.2f} - €{ci['upper']:,.2f}")
            
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response.json())
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_batch_prediction():
    """Test batch player prediction."""
    print_header("Testing Batch Prediction (POST /predict/batch)")
    
    payload = {
        "player_ids": TEST_PLAYER_IDS[:3],  # Test with first 3 players
        "model_type": "ensemble",
        "include_confidence": True
    }
    
    print_info(f"Request payload:")
    print_response(payload)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success(f"Status Code: {response.status_code}")
            data = response.json()
            
            print_info(f"Total Requested: {data.get('total_requested')}")
            print_info(f"Total Successful: {data.get('total_successful')}")
            
            if data.get('failed_predictions'):
                print_error(f"Failed: {data.get('failed_predictions')}")
            
            print_info("\nPredictions:")
            for result in data.get('results', []):
                player_id = result.get('player_id')
                value = result.get('predicted_value', 0)
                print(f"  Player {player_id}: €{value:,.2f}")
            
            return True
        else:
            print_error(f"Status Code: {response.status_code}")
            print_response(response.json())
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_different_model_types():
    """Test predictions with different model types."""
    print_header("Testing Different Model Types")
    
    player_id = TEST_PLAYER_IDS[0]
    model_types = ["ensemble", "xgboost", "lightgbm"]
    
    results = {}
    
    for model_type in model_types:
        print_info(f"\nTesting {model_type.upper()} model...")
        
        payload = {
            "player_id": player_id,
            "model_type": model_type,
            "include_confidence": False
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    value = data['result']['predicted_value']
                    results[model_type] = value
                    print_success(f"{model_type}: €{value:,.2f}")
                else:
                    print_error(f"{model_type}: Prediction failed")
            else:
                print_error(f"{model_type}: HTTP {response.status_code}")
                
        except Exception as e:
            print_error(f"{model_type}: {e}")
    
    if results:
        print_info("\nComparison:")
        for model_type, value in results.items():
            print(f"  {model_type:12s}: €{value:,.2f}")
        return True
    
    return False


def test_error_handling():
    """Test error handling with invalid requests."""
    print_header("Testing Error Handling")
    
    # Test with non-existent player
    print_info("\n1. Testing with non-existent player ID...")
    payload = {
        "player_id": "invalid_player_999999",
        "model_type": "ensemble",
        "include_confidence": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code in [200, 404]:
            data = response.json()
            if not data.get('success'):
                print_success("Correctly handled non-existent player")
            else:
                print_info("Prediction succeeded (mock model behavior)")
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error: {e}")
    
    # Test with invalid model type
    print_info("\n2. Testing with invalid model type...")
    payload = {
        "player_id": TEST_PLAYER_IDS[0],
        "model_type": "invalid_model",
        "include_confidence": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 422:
            print_success("Correctly rejected invalid model type")
        else:
            print_info(f"Status code: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error: {e}")
    
    return True


def run_all_tests():
    """Run all API tests."""
    print_header("API Endpoint Testing Suite")
    print_info(f"Testing API at: {BASE_URL}")
    print_info(f"Test Player IDs: {', '.join(TEST_PLAYER_IDS)}")
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_endpoint),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Different Model Types", test_different_model_types),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Fore.CYAN}Results: {passed}/{total} tests passed{Style.RESET_ALL}")
    
    if passed == total:
        print_success("\n🎉 All tests passed!")
    else:
        print_info(f"\n⚠️  {total - passed} test(s) failed")
    
    print_header("Next Steps")
    print("1. Open Swagger UI: http://localhost:8000/docs")
    print("2. Try the interactive API documentation")
    print("3. Test with your own player IDs")
    print("4. Build a frontend to visualize predictions")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print_info("\n\nTests interrupted by user")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
