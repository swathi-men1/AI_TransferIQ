import pandas as pd

datasets = {
    "Batting": r"Cricket Datasets/all_season_batting_card.csv",
    "Bowling": r"Cricket Datasets/all_season_bowling_card.csv",
    "Auction": r"Cricket Datasets/IPL AUCTION.csv",
    "Match": r"Cricket Datasets/Match.csv",
    "Player": r"Cricket Datasets/Player.csv",
    "Season": r"Cricket Datasets/Season.csv"
}

for name, path in datasets.items():

    print("\n==============================")
    print("DATASET:", name)
    print("==============================")

    df = pd.read_csv(path)

    print("\nShape (Rows, Columns):")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample rows:")
    print(df.head())