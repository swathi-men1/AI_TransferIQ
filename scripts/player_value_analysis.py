import pandas as pd
import matplotlib.pyplot as plt

print("Loading dataset...")

data = pd.read_csv("output/clean_ml_dataset.csv")

# ================================
# Top 10 Most Expensive Players
# ================================

top_players = (
    data.groupby("Player")["Sold_Price_CR"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Most Valuable IPL Players")

print(top_players)

plt.figure()

top_players.plot(kind="bar")

plt.title("Top 10 Most Valuable IPL Players")

plt.ylabel("Average Auction Price (CR)")

plt.xlabel("Player")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ================================
# Price Trend for Individual Player
# ================================

player_name = input("\nEnter player name to see price trend: ")

player_data = data[data["Player"].str.lower() == player_name.lower()]

if player_data.empty:
    print("Player not found in dataset")
else:

    player_data = player_data.sort_values("Season_Year")

    plt.figure()

    plt.plot(
        player_data["Season_Year"],
        player_data["Sold_Price_CR"],
        marker="o"
    )

    plt.title(f"Auction Price Trend for {player_name}")

    plt.xlabel("Season")

    plt.ylabel("Auction Price (CR)")

    plt.grid()

    plt.show()
