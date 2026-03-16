"""
Create Real Players Dataset

This script creates a dataset with real player names for testing.
You can later replace this with actual data from Transfermarkt/StatsBomb.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Real player data (famous players for testing)
REAL_PLAYERS = [
    # Forwards
    {"name": "Lionel Messi", "age": 36, "position": "Forward", "club": "Inter Miami", "nationality": "Argentina"},
    {"name": "Cristiano Ronaldo", "age": 39, "position": "Forward", "club": "Al Nassr", "nationality": "Portugal"},
    {"name": "Kylian Mbappé", "age": 25, "position": "Forward", "club": "Real Madrid", "nationality": "France"},
    {"name": "Erling Haaland", "age": 23, "position": "Forward", "club": "Manchester City", "nationality": "Norway"},
    {"name": "Harry Kane", "age": 30, "position": "Forward", "club": "Bayern Munich", "nationality": "England"},
    {"name": "Robert Lewandowski", "age": 35, "position": "Forward", "club": "Barcelona", "nationality": "Poland"},
    {"name": "Mohamed Salah", "age": 31, "position": "Forward", "club": "Liverpool", "nationality": "Egypt"},
    {"name": "Vinícius Júnior", "age": 23, "position": "Forward", "club": "Real Madrid", "nationality": "Brazil"},
    {"name": "Neymar Jr", "age": 32, "position": "Forward", "club": "Al Hilal", "nationality": "Brazil"},
    {"name": "Karim Benzema", "age": 36, "position": "Forward", "club": "Al Ittihad", "nationality": "France"},
    
    # Midfielders
    {"name": "Kevin De Bruyne", "age": 32, "position": "Midfielder", "club": "Manchester City", "nationality": "Belgium"},
    {"name": "Luka Modrić", "age": 38, "position": "Midfielder", "club": "Real Madrid", "nationality": "Croatia"},
    {"name": "Jude Bellingham", "age": 20, "position": "Midfielder", "club": "Real Madrid", "nationality": "England"},
    {"name": "Bruno Fernandes", "age": 29, "position": "Midfielder", "club": "Manchester United", "nationality": "Portugal"},
    {"name": "Pedri", "age": 21, "position": "Midfielder", "club": "Barcelona", "nationality": "Spain"},
    {"name": "Frenkie de Jong", "age": 26, "position": "Midfielder", "club": "Barcelona", "nationality": "Netherlands"},
    {"name": "Casemiro", "age": 32, "position": "Midfielder", "club": "Manchester United", "nationality": "Brazil"},
    {"name": "Rodri", "age": 27, "position": "Midfielder", "club": "Manchester City", "nationality": "Spain"},
    {"name": "Toni Kroos", "age": 34, "position": "Midfielder", "club": "Real Madrid", "nationality": "Germany"},
    {"name": "İlkay Gündoğan", "age": 33, "position": "Midfielder", "club": "Barcelona", "nationality": "Germany"},
    
    # Defenders
    {"name": "Virgil van Dijk", "age": 32, "position": "Defender", "club": "Liverpool", "nationality": "Netherlands"},
    {"name": "Rúben Dias", "age": 26, "position": "Defender", "club": "Manchester City", "nationality": "Portugal"},
    {"name": "Antonio Rüdiger", "age": 30, "position": "Defender", "club": "Real Madrid", "nationality": "Germany"},
    {"name": "Marquinhos", "age": 29, "position": "Defender", "club": "Paris Saint-Germain", "nationality": "Brazil"},
    {"name": "Joško Gvardiol", "age": 22, "position": "Defender", "club": "Manchester City", "nationality": "Croatia"},
    {"name": "William Saliba", "age": 22, "position": "Defender", "club": "Arsenal", "nationality": "France"},
    {"name": "Eder Militão", "age": 26, "position": "Defender", "club": "Real Madrid", "nationality": "Brazil"},
    {"name": "Kim Min-jae", "age": 27, "position": "Defender", "club": "Bayern Munich", "nationality": "South Korea"},
    {"name": "Alessandro Bastoni", "age": 24, "position": "Defender", "club": "Inter Milan", "nationality": "Italy"},
    {"name": "Theo Hernández", "age": 26, "position": "Defender", "club": "AC Milan", "nationality": "France"},
    
    # Goalkeepers
    {"name": "Thibaut Courtois", "age": 31, "position": "Goalkeeper", "club": "Real Madrid", "nationality": "Belgium"},
    {"name": "Alisson Becker", "age": 31, "position": "Goalkeeper", "club": "Liverpool", "nationality": "Brazil"},
    {"name": "Ederson", "age": 30, "position": "Goalkeeper", "club": "Manchester City", "nationality": "Brazil"},
    {"name": "Marc-André ter Stegen", "age": 31, "position": "Goalkeeper", "club": "Barcelona", "nationality": "Germany"},
    {"name": "Gianluigi Donnarumma", "age": 25, "position": "Goalkeeper", "club": "Paris Saint-Germain", "nationality": "Italy"},
    {"name": "Mike Maignan", "age": 28, "position": "Goalkeeper", "club": "AC Milan", "nationality": "France"},
    {"name": "Jan Oblak", "age": 31, "position": "Goalkeeper", "club": "Atlético Madrid", "nationality": "Slovenia"},
    {"name": "Emiliano Martínez", "age": 31, "position": "Goalkeeper", "club": "Aston Villa", "nationality": "Argentina"},
]


def generate_realistic_stats(player):
    """Generate realistic stats based on player position and age."""
    position = player['position']
    age = player['age']
    
    # Base stats by position
    if position == "Forward":
        goals = np.random.randint(15, 35)
        assists = np.random.randint(5, 15)
        shots = np.random.randint(80, 150)
        passes_pct = np.random.uniform(70, 85)
        tackles = np.random.randint(10, 30)
        interceptions = np.random.randint(5, 20)
        base_value = np.random.uniform(40, 150)
    elif position == "Midfielder":
        goals = np.random.randint(5, 20)
        assists = np.random.randint(8, 20)
        shots = np.random.randint(40, 100)
        passes_pct = np.random.uniform(80, 92)
        tackles = np.random.randint(40, 80)
        interceptions = np.random.randint(30, 60)
        base_value = np.random.uniform(30, 120)
    elif position == "Defender":
        goals = np.random.randint(0, 8)
        assists = np.random.randint(0, 10)
        shots = np.random.randint(10, 40)
        passes_pct = np.random.uniform(82, 90)
        tackles = np.random.randint(60, 120)
        interceptions = np.random.randint(50, 100)
        base_value = np.random.uniform(25, 100)
    else:  # Goalkeeper
        goals = 0
        assists = 0
        shots = 0
        passes_pct = np.random.uniform(70, 85)
        tackles = 0
        interceptions = 0
        base_value = np.random.uniform(20, 80)
    
    # Age adjustment (peak at 27-28)
    age_factor = 1.0 - abs(age - 27.5) * 0.02
    age_factor = max(0.6, min(1.0, age_factor))
    
    market_value = base_value * age_factor * 1_000_000
    
    return {
        'goals': goals,
        'assists': assists,
        'shots': shots,
        'passes_pct': round(passes_pct, 1),
        'tackles': tackles,
        'interceptions': interceptions,
        'minutes': np.random.randint(2000, 3500),
        'appearances': np.random.randint(25, 45),
        'market_value': market_value,
        'fee': market_value * np.random.uniform(0.8, 1.2),
        'injury_risk': np.random.uniform(0.1, 0.5),
        'sentiment_score': np.random.uniform(0.3, 0.9),
        'contract_years': np.random.randint(1, 5),
    }


def create_real_players_dataset():
    """Create dataset with real player names."""
    print("=" * 60)
    print("Creating Real Players Dataset")
    print("=" * 60)
    
    # Create directory
    Path("data/training").mkdir(parents=True, exist_ok=True)
    
    # Build dataset
    players_data = []
    
    for i, player in enumerate(REAL_PLAYERS):
        stats = generate_realistic_stats(player)
        
        player_record = {
            'player_id': f"real_{i:03d}",
            'player_name': player['name'],
            'age': player['age'],
            'position': player['position'],
            'club': player['club'],
            'nationality': player['nationality'],
            **stats
        }
        
        players_data.append(player_record)
        print(f"✓ Added {player['name']} ({player['position']}, {player['club']})")
    
    # Create DataFrame
    df = pd.DataFrame(players_data)
    
    # Save to CSV
    output_path = Path("data/training/training_dataset.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created dataset with {len(df)} real players")
    print(f"📁 Saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total Players: {len(df)}")
    print(f"\nBy Position:")
    print(df['position'].value_counts().to_string())
    print(f"\nTop 5 Most Valuable Players:")
    top_5 = df.nlargest(5, 'market_value')[['player_name', 'position', 'club', 'market_value']]
    for _, row in top_5.iterrows():
        print(f"  {row['player_name']:25s} - €{row['market_value']/1_000_000:.1f}M ({row['club']})")
    
    print("\n" + "=" * 60)
    print("Test the API with these player names:")
    print("=" * 60)
    print("• Lionel Messi")
    print("• Cristiano Ronaldo")
    print("• Kylian Mbappé")
    print("• Erling Haaland")
    print("• Kevin De Bruyne")
    print("• Virgil van Dijk")
    print("• Thibaut Courtois")
    print("\nOr search for any player by name using the /search endpoint!")
    print("=" * 60)


if __name__ == "__main__":
    create_real_players_dataset()
