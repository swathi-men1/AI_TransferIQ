import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


print("Loading dataset...")
df = pd.read_csv("output/clean_ml_dataset.csv")


# =====================================================
# MODEL FEATURES
# =====================================================
features = [
    "Runs_Last_Season",
    "Strike_Rate_Last_Season",
    "Batting_Average_Last_Season",
    "Wickets_Last_Season",
    "Economy_Last_Season",
    "International_Experience_Years",
    "Previous_Auction_Price_CR"
]

target = "Sold_Price_CR"


X = df[features]
y = df[target]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Training models...")

xgb = XGBRegressor()
rf = RandomForestRegressor()

xgb.fit(X_scaled, y)
rf.fit(X_scaled, y)

print("Models ready!")

analyzer = SentimentIntensityAnalyzer()


# =====================================================
# PLAYER SEARCH
# =====================================================
def search_player(name):

    players = df[df["Player"].str.contains(name, case=False, na=False)]

    if players.empty:

        all_players = df["Player"].unique()

        suggestion = difflib.get_close_matches(name, all_players, n=1)

        if suggestion:
            print("Did you mean:", suggestion[0])
            players = df[df["Player"] == suggestion[0]]
        else:
            return None

    player = players.sort_values("Season_Year", ascending=False).iloc[0]

    return player


# =====================================================
# FIXED PREDICTION ENGINE
# =====================================================

def predict_price(player):

    runs = player["Runs_Last_Season"]
    strike = player["Strike_Rate_Last_Season"]
    avg = player["Batting_Average_Last_Season"]
    wickets = player["Wickets_Last_Season"]
    economy = player["Economy_Last_Season"]
    exp = player["International_Experience_Years"]
    prev = player["Previous_Auction_Price_CR"]

    # Normalize batting performance
    batting_score = (
        (runs / 800) * 4 +
        (strike / 180) * 3 +
        (avg / 50) * 3
    )

    # Normalize bowling performance
    bowling_score = (
        (wickets / 25) * 4 +
        ((10 - economy) / 10) * 3
    )

    # Experience bonus
    experience_score = (exp / 15) * 2

    # Previous price small influence
    previous_score = prev * 0.15

    total_score = batting_score + bowling_score + experience_score + previous_score

    # Clamp realistic auction range
    predicted_price = max(0.5, min(total_score, 24.75))

    return round(predicted_price, 2)

# =====================================================
# MAIN MENU
# =====================================================
while True:

    print("\n🏏 IPL Auction Intelligence System")
    print("1 Predict Auction Value")
    print("2 Sentiment Analysis")
    print("3 Analytics Graphs")
    print("4 Player Details")
    print("5 Player Comparison")
    print("6 Player Value Analysis")
    print("7 Value-for-Money Players")
    print("8 Feature Importance")
    print("9 Model Evaluation")
    print("10 Exit")

    option = input("Enter option: ").strip()


# =====================================================
# OPTION 1 MANUAL PREDICTION
# =====================================================
    if option == "1":

        runs = float(input("Runs: "))
        strike = float(input("Strike rate: "))
        avg = float(input("Batting average: "))
        wickets = float(input("Wickets: "))
        economy = float(input("Economy: "))
        exp = float(input("Experience years: "))
        prev = float(input("Previous price (CR): "))

        player = {
            "Runs_Last_Season": runs,
            "Strike_Rate_Last_Season": strike,
            "Batting_Average_Last_Season": avg,
            "Wickets_Last_Season": wickets,
            "Economy_Last_Season": economy,
            "International_Experience_Years": exp,
            "Previous_Auction_Price_CR": prev
        }

        price = predict_price(player)

        print("\nPredicted Auction Value:", price, "CR")



# =====================================================
# OPTION 2 SENTIMENT ANALYSIS
# =====================================================
    elif option == "2":

        from scripts.sentiment_analysis import analyze_sentiment

        text = input("Enter cricket opinion text: ")

        score = analyze_sentiment(text)

        print("\nSentiment Score:", score)

        if score >= 0.3:
            print("Positive Cricket Sentiment")
        elif score <= -0.3:
            print("Negative Cricket Sentiment")
        else:
            print("Neutral Cricket Sentiment")


# =====================================================
# OPTION 3 ANALYTICS GRAPHS (CLEAN DASHBOARD)
# =====================================================
    elif option == "3":

        print("\nGenerating Analytics Dashboard...")

        plot_df = df.sample(min(1500, len(df)))

        fig, axes = plt.subplots(2, 4, figsize=(24,12))

        # Auction Price Distribution
        axes[0,0].hist(plot_df["Sold_Price_CR"], bins=25, color="orange")
        axes[0,0].set_title("Auction Price Distribution", pad=15)
        axes[0,0].set_xlabel("Auction Price (CR)")
        axes[0,0].set_ylabel("Players")

        # Runs vs Price
        axes[0,1].scatter(plot_df["Runs_Last_Season"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="blue")
        axes[0,1].set_title("Runs vs Auction Price", pad=15)
        axes[0,1].set_xlabel("Runs Last Season")
        axes[0,1].set_ylabel("Price (CR)")

        # Strike Rate vs Price
        axes[0,2].scatter(plot_df["Strike_Rate_Last_Season"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="green")
        axes[0,2].set_title("Strike Rate vs Price", pad=15)
        axes[0,2].set_xlabel("Strike Rate")
        axes[0,2].set_ylabel("Price (CR)")

        # Batting Average vs Price
        axes[0,3].scatter(plot_df["Batting_Average_Last_Season"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="purple")
        axes[0,3].set_title("Batting Average vs Price", pad=15)
        axes[0,3].set_xlabel("Batting Average")
        axes[0,3].set_ylabel("Price (CR)")

        # Wickets vs Price
        axes[1,0].scatter(plot_df["Wickets_Last_Season"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="red")
        axes[1,0].set_title("Wickets vs Price", pad=15)
        axes[1,0].set_xlabel("Wickets Last Season")
        axes[1,0].set_ylabel("Price (CR)")

        # Economy vs Price
        axes[1,1].scatter(plot_df["Economy_Last_Season"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="brown")
        axes[1,1].set_title("Economy vs Price", pad=15)
        axes[1,1].set_xlabel("Economy Rate")
        axes[1,1].set_ylabel("Price (CR)")

        # Experience vs Price
        axes[1,2].scatter(plot_df["International_Experience_Years"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="teal")
        axes[1,2].set_title("Experience vs Price", pad=15)
        axes[1,2].set_xlabel("Experience (Years)")
        axes[1,2].set_ylabel("Price (CR)")

        # Previous Price vs Current Price
        axes[1,3].scatter(plot_df["Previous_Auction_Price_CR"], plot_df["Sold_Price_CR"],
                          alpha=0.4, s=20, color="black")
        axes[1,3].set_title("Previous Price vs Current Price", pad=15)
        axes[1,3].set_xlabel("Previous Auction Price")
        axes[1,3].set_ylabel("Current Price")

        # Add grid for all graphs
        for ax in axes.flat:
            ax.grid(True)

        # Proper spacing between graphs
        plt.subplots_adjust(wspace=0.35, hspace=0.35)

        plt.show()

# =====================================================
# OPTION 4 PLAYER DETAILS (CLEAN VERSION)
# =====================================================
    elif option == "4":

        name = input("Enter Player Name: ")

        player = search_player(name)

        if player is None:
            print("Player not found")
            continue

        print("\n==============================")
        print("PLAYER PROFILE")
        print("==============================")

        print("\nBasic Information")
        print("Player:", player["Player"])
        print("Country:", player["Country"])
        print("Player Type:", player["Player_Type"])
        print("Role:", player["Role_Specific"])
        print("Team:", player["Team_Bought"])
        print("Auction Status:", player["Auction_Status"])

        print("\nBatting Performance")
        print("Matches:", player["Matches_Last_Season"])
        print("Runs:", player["Runs_Last_Season"])
        print("Strike Rate:", player["Strike_Rate_Last_Season"])
        print("Batting Average:", player["Batting_Average_Last_Season"])

        print("\nBowling Performance")
        print("Wickets:", player["Wickets_Last_Season"])
        print("Economy:", player["Economy_Last_Season"])
        print("Bowling Average:", player["Bowling_Average_Last_Season"])

        print("\nFielding")
        print("Catches:", player["Catches_Last_Season"])
        print("Stumpings:", player["Stumpings_Last_Season"])

        print("\nExperience")
        print("International Experience:", player["International_Experience_Years"], "years")
        print("Captaincy Experience:", player["Captaincy_Experience"])
        print("Injury Status:", player["Injury_Status"])

        print("\nAuction Information")
        print("Base Price:", player["Base_Price_CR"], "CR")
        print("Previous Auction Price:", player["Previous_Auction_Price_CR"], "CR")
        print("Sold Price:", player["Sold_Price_CR"], "CR")

        price = predict_price(player)

        print("\nPredicted Auction Value:", price, "CR")

# =====================================================
# OPTION 5 PLAYER COMPARISON
# =====================================================
    elif option == "5":

        p1 = input("Enter Player 1: ")
        p2 = input("Enter Player 2: ")

        player1 = search_player(p1)
        player2 = search_player(p2)

        if player1 is None or player2 is None:
            print("Player not found")
            continue

        print("\nPLAYER COMPARISON\n")

        for col in df.columns:

            v1 = player1[col]
            v2 = player2[col]

            if pd.isna(v1):
                v1 = "N/A"

            if pd.isna(v2):
                v2 = "N/A"

            print(col)
            print(player1["Player"], ":", v1)
            print(player2["Player"], ":", v2)
            print("-"*40)

        price1 = predict_price(player1)
        price2 = predict_price(player2)

        print("\nPredicted Auction Value")
        print(player1["Player"], ":", price1, "CR")
        print(player2["Player"], ":", price2, "CR")


# =====================================================
# OPTION 6 VALUE ANALYSIS
# =====================================================
    elif option == "6":

        while True:

            print("\nPlayer Value Analysis")
            print("1 Top Players by Year")
            print("2 Player Auction Price Trend")
            print("3 Back")

            sub = input("Select option: ")

            if sub == "1":

                year = int(input("Enter Auction Year: "))

                season = df[df["Season_Year"] == year]

                top = (
                    season.groupby("Player")["Sold_Price_CR"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(25)
                )

                print("\nTop Players")
                print(top)

            elif sub == "2":

                name = input("Enter Player Name: ")

                player_df = df[df["Player"].str.contains(name, case=False, na=False)]

                trend = (
                    player_df.groupby("Season_Year")["Sold_Price_CR"]
                    .mean()
                    .sort_index()
                )

                print("\nAuction Price History")
                print(trend)

                plt.figure()
                trend.plot(marker="o")
                plt.title("Auction Price Trend")
                plt.show()

            elif sub == "3":
                break


# =====================================================
# OPTION 7 VALUE FOR MONEY
# =====================================================
    elif option == "7":

        year = int(input("Enter Auction Year: "))
        role = input("Enter role (batsman/bowler): ").lower()

        season = df[df["Season_Year"] == year]

        if role == "batsman":
            season = season[season["Runs_Last_Season"] > 0]

        if role == "bowler":
            season = season[season["Wickets_Last_Season"] > 0]

        players = season.groupby("Player").mean(numeric_only=True)

        players["Performance"] = (
            players["Runs_Last_Season"] +
            players["Wickets_Last_Season"]*20
        )

        players["Value_Index"] = players["Performance"] / players["Sold_Price_CR"]

        top = players.sort_values("Value_Index",ascending=False).head(10)

        print("\nTop Value-for-Money Players")
        print(top[["Performance","Sold_Price_CR","Value_Index"]])


# =====================================================
# OPTION 8 FEATURE IMPORTANCE
# =====================================================
    elif option == "8":

        importance = pd.DataFrame({
            "Feature":features,
            "Importance":xgb.feature_importances_
        })

        print("\nFeature Importance")
        print(importance.sort_values("Importance",ascending=False))


# =====================================================
# OPTION 9 MODEL EVALUATION
# =====================================================
    elif option == "9":

        preds = xgb.predict(X_scaled)

        mae = mean_absolute_error(y,preds)
        rmse = np.sqrt(mean_squared_error(y,preds))
        r2 = r2_score(y,preds)

        print("\nModel Evaluation")
        print("MAE:",round(mae,3))
        print("RMSE:",round(rmse,3))
        print("R2 Score:",round(r2,3))


# =====================================================
# EXIT
# =====================================================
    elif option == "10":

        print("Exiting system...")
        break

    else:
        print("Invalid option")
