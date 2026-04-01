import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

print("Loading CLEAN ML dataset...")

data = pd.read_csv("output/clean_ml_dataset.csv")

print("Dataset shape:", data.shape)

# ================================
# REMOVE DATA LEAKAGE COLUMNS
# ================================

columns_to_remove = [
    "Sold_Price_USD_000"
]

data = data.drop(columns=columns_to_remove, errors="ignore")

# ================================
# REMOVE NON-PREDICTIVE COLUMNS
# ================================

drop_columns = [
    "Player",
    "Team_Bought",
    "Auction_Status"
]

data = data.drop(columns=drop_columns, errors="ignore")

# ================================
# TARGET VARIABLE
# ================================

target = "Sold_Price_CR"

y = data[target]

# ================================
# FEATURES
# ================================

X = data.drop(columns=[target])

# ================================
# ENCODE CATEGORICAL VARIABLES
# ================================

print("Encoding categorical features...")

X = pd.get_dummies(X)

print("Total features after encoding:", X.shape[1])

# ================================
# TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ================================
# SCALING
# ================================

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# TRAIN MODEL
# ================================

print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

print("Model training complete")

# ================================
# PREDICTION
# ================================

predictions = model.predict(X_test)

# ================================
# EVALUATION
# ================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ================================
# FEATURE IMPORTANCE
# ================================

print("\nCalculating feature importance...")

importances = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")

print(importance_df.head(10))

# ================================
# SAVE FEATURE IMPORTANCE
# ================================

importance_df.to_csv(
    "output/feature_importance.csv",
    index=False
)

print("\nFeature importance saved to output/feature_importance.csv")