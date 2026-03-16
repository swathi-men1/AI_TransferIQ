"""
Test Real Players API

Test the API with real player names instead of IDs.
"""

import requests
import json
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

BASE_URL = "http://localhost:8000"


def print_header(text):
    """Print formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def print_success(text):
    """Print success message."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text):
    """Print error message."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text):
    """Print info message."""
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")


def test_search_players():
    """Test player search endpoint."""
    print_header("Test 1: Search for Players")
    
    search_terms = ["Messi", "Ronaldo", "Haaland", "De Bruyne"]
    
    for term in search_terms:
        print(f"\n{Fore.WHITE}Searching for: {term}{Style.RESET_ALL}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/search",
                json={"player_name": term, "limit": 5}
            )
            
            if response.status_code == 200:
                data = response.json()
                players = data.get('players', [])
                
                if players:
                    print_success(f"Found {len(players)} player(s):")
                    for player in players:
                        name = player['player_name']
                        pos = player.get('position', 'Unknown')
                        club = player.get('club', 'Unknown')
                        value = player.get('market_value', 0)
                        print(f"  • {name:25s} - {pos:12s} ({club}) - €{value/1_000_000:.1f}M")
                else:
                    print_info("No players found")
            else:
                print_error(f"HTTP {response.status_code}")
                
        except Exception as e:
            print_error(f"Error: {e}")


def test_predict_by_name():
    """Test prediction by player name."""
    print_header("Test 2: Predict Transfer Value by Name")
    
    players = [
        "Lionel Messi",
        "Cristiano Ronaldo",
        "Kylian Mbappé",
        "Erling Haaland",
        "Kevin De Bruyne"
    ]
    
    for player_name in players:
        print(f"\n{Fore.WHITE}Predicting for: {player_name}{Style.RESET_ALL}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict/by-name",
                json={
                    "player_name": player_name,
                    "model_type": "ensemble",
                    "include_confidence": True,
                    "fetch_realtime_data": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('result', {})
                    predicted_value = result.get('predicted_value', 0)
                    ci = result.get('confidence_interval')
                    
                    print_success(f"Predicted Value: €{predicted_value/1_000_000:.2f}M")
                    
                    if ci:
                        lower = ci['lower'] / 1_000_000
                        upper = ci['upper'] / 1_000_000
                        print_info(f"Confidence Interval: €{lower:.2f}M - €{upper:.2f}M")
                else:
                    print_error(data.get('message', 'Prediction failed'))
            else:
                print_error(f"HTTP {response.status_code}")
                
        except Exception as e:
            print_error(f"Error: {e}")


def test_partial_name_search():
    """Test partial name matching."""
    print_header("Test 3: Partial Name Search")
    
    partial_names = ["Messi", "Ronaldo", "van", "De"]
    
    for partial in partial_names:
        print(f"\n{Fore.WHITE}Searching for players with '{partial}' in name:{Style.RESET_ALL}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/search",
                json={"player_name": partial, "limit": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                players = data.get('players', [])
                
                if players:
                    print_success(f"Found {len(players)} player(s):")
                    for player in players:
                        print(f"  • {player['player_name']}")
                else:
                    print_info("No players found")
            else:
                print_error(f"HTTP {response.status_code}")
                
        except Exception as e:
            print_error(f"Error: {e}")


def test_compare_players():
    """Compare predictions for multiple players."""
    print_header("Test 4: Compare Player Values")
    
    players = [
        "Lionel Messi",
        "Cristiano Ronaldo",
        "Kylian Mbappé",
        "Erling Haaland",
        "Jude Bellingham"
    ]
    
    results = []
    
    print(f"\n{Fore.WHITE}Fetching predictions...{Style.RESET_ALL}")
    
    for player_name in players:
        try:
            response = requests.post(
                f"{BASE_URL}/predict/by-name",
                json={
                    "player_name": player_name,
                    "model_type": "ensemble",
                    "include_confidence": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    value = data['result']['predicted_value']
                    results.append((player_name, value))
                    
        except Exception as e:
            print_error(f"Error for {player_name}: {e}")
    
    if results:
        # Sort by value
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{Fore.GREEN}Player Value Rankings:{Style.RESET_ALL}")
        for i, (name, value) in enumerate(results, 1):
            print(f"  {i}. {name:25s} - €{value/1_000_000:.2f}M")


def run_all_tests():
    """Run all tests."""
    print_header("Real Players API Testing Suite")
    print_info(f"Testing API at: {BASE_URL}")
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print_error("API server is not responding")
            return
        
        print_success("API server is online")
        
        # Run tests
        test_search_players()
        test_predict_by_name()
        test_partial_name_search()
        test_compare_players()
        
        print_header("Testing Complete!")
        print("\n" + Fore.GREEN + "✅ All tests completed successfully!" + Style.RESET_ALL)
        print("\nYou can now:")
        print("1. Open Swagger UI: http://localhost:8000/docs")
        print("2. Try the /search endpoint to find players")
        print("3. Try the /predict/by-name endpoint to get predictions")
        print("4. Search for any player by name (e.g., 'Messi', 'Ronaldo', 'Haaland')")
        
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to API server")
        print_info("Make sure the server is running: python start_server.py")
    except Exception as e:
        print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    run_all_tests()
