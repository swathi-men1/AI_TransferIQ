import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


print("Loading dataset...")

data = pd.read_csv("output/clean_ml_dataset.csv")

target = "Sold_Price_CR"

drop_columns = [
    "Player",
    "Team_Bought",
    "Auction_Status"
]

model_data = data.drop(columns=drop_columns, errors="ignore")

y = model_data[target]

X = model_data.drop(columns=[target])

X = pd.get_dummies(X)

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training model...")

model = XGBRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ===============================
# GRAPH 1 — ACTUAL VS PREDICTED
# ===============================

plt.figure()

plt.scatter(y_test, predictions)

plt.xlabel("Actual Auction Price (CR)")
plt.ylabel("Predicted Auction Price (CR)")

plt.title("Actual vs Predicted Auction Price")

plt.show()


# ===============================
# GRAPH 2 — PRICE DISTRIBUTION
# ===============================

plt.figure()

plt.hist(data["Sold_Price_CR"], bins=30)

plt.title("Auction Price Distribution")

plt.xlabel("Price (CR)")
plt.ylabel("Players")

plt.show()


# ===============================
# GRAPH 3 — RUNS VS PRICE
# ===============================

if "Runs_Last_Season" in data.columns:

    plt.figure()

    plt.scatter(data["Runs_Last_Season"], data["Sold_Price_CR"])

    plt.title("Runs vs Auction Price")

    plt.xlabel("Runs")

    plt.ylabel("Price")

    plt.show()


# ===============================
# GRAPH 4 — WICKETS VS PRICE
# ===============================

if "Wickets_Last_Season" in data.columns:

    plt.figure()

    plt.scatter(data["Wickets_Last_Season"], data["Sold_Price_CR"])

    plt.title("Wickets vs Auction Price")

    plt.xlabel("Wickets")

    plt.ylabel("Price")

    plt.show()
