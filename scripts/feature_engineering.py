import pandas as pd

print("Loading datasets...")

batting = pd.read_csv("Cricket Datasets/all_season_batting_card.csv")
bowling = pd.read_csv("Cricket Datasets/all_season_bowling_card.csv")
auction = pd.read_csv("Cricket Datasets/IPL AUCTION.csv")

# =========================
# BATTING FEATURES
# =========================

print("Creating batting features...")

batting_features = batting.groupby(["season", "fullName"]).agg({
    "runs": "sum",
    "ballsFaced": "sum",
    "fours": "sum",
    "sixes": "sum",
}).reset_index()

batting_features["strike_rate"] = (
    batting_features["runs"] /
    batting_features["ballsFaced"]
) * 100

# =========================
# BOWLING FEATURES
# =========================

print("Creating bowling features...")

bowling_features = bowling.groupby(["season", "fullName"]).agg({
    "overs": "sum",
    "conceded": "sum",
    "wickets": "sum"
}).reset_index()

bowling_features["economy"] = (
    bowling_features["conceded"] /
    bowling_features["overs"]
)

# =========================
# MERGE BATTING + BOWLING
# =========================

print("Merging batting and bowling...")

performance = pd.merge(
    batting_features,
    bowling_features,
    on=["season", "fullName"],
    how="outer"
)

performance = performance.fillna(0)

performance.rename(
    columns={"fullName": "Player"},
    inplace=True
)

# =========================
# MERGE AUCTION DATA
# =========================

print("Merging auction dataset...")

final = pd.merge(
    auction,
    performance,
    left_on=["Player", "Season_Year"],
    right_on=["Player", "season"],
    how="left"
)

final = final.fillna(0)

# =========================
# ADVANCED FEATURES
# =========================

print("Creating advanced features...")

final["Impact_Score"] = (
    final["runs"] +
    (20 * final["wickets"]) +
    (5 * final["Catches_Last_Season"])
)

final["Experience_Score"] = (
    final["International_Experience_Years"] *
    final["Matches_Last_Season"]
)

final["Form_Index"] = (
    final["Batting_Average_Last_Season"] *
    final["Strike_Rate_Last_Season"]
)

# =========================
# SAVE DATASET
# =========================

print("Final dataset shape:")
print(final.shape)

final.to_csv("output/final_ml_dataset.csv", index=False)

print("Dataset saved to output/final_ml_dataset.csv")