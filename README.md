AI_TransferIQ -- IPL Auction Intelligence System

A complete data analytics and machine learning project to estimate IPL player auction values, inspect player performance, compare players, analyze value-for-money picks, and run cricket-specific sentiment analysis.

## 1. Project Overview

The IPL auction process involves large financial decisions under uncertainty. Teams must evaluate players across batting, bowling, experience, and previous auction history while balancing budgets.

This project builds a practical decision-support system that:
- preprocesses and engineers cricket performance features,
- trains ML models for auction value estimation,
- provides interactive analytics and player insights,
- includes a menu-driven application for end users.

The main application entry point is:
- `main_app.py`

## 2. Problem Statement

IPL franchises and analysts need a consistent way to estimate a player's fair auction value from historical and recent performance indicators.

Challenges addressed:
- fragmented historical datasets,
- inconsistent quality and missing/outlier records,
- multiple performance dimensions (batting, bowling, fielding, experience),
- need for explainable and usable outputs for non-technical users.

## 3. What This Project Is Doing

This project implements an end-to-end pipeline:
1. Read raw cricket and auction datasets.
2. Build engineered features from batting and bowling cards.
3. Merge with auction records.
4. Clean data by removing duplicates, invalid values, and outliers.
5. Train regression models for player valuation.
6. Serve insights through a terminal menu app.

It also includes:
- cricket-domain sentiment scoring using VADER + custom lexicon,
- visualization utilities for model and player analysis,
- feature-importance reporting.

## 4. Project Objectives

- Predict player auction value in Crores (CR).
- Compare players using historical attributes.
- Identify top value-for-money players per year and role.
- Provide visual analytics on auction and performance trends.
- Evaluate model quality using MAE, RMSE, and R2.

## 5. Repository Structure

```text
.
|-- main_app.py
|-- player_sentiment_dataset.csv
|-- Cricket Datasets/
|   |-- all_season_batting_card.csv
|   |-- all_season_bowling_card.csv
|   |-- IPL AUCTION.csv
|   |-- Match.csv
|   |-- Player.csv
|   |-- Season.csv
|   |-- SA (1).csv
|   |-- SA (2).csv
|   |-- SA (3).csv
|-- output/
|   |-- final_ml_dataset.csv
|   |-- clean_ml_dataset.csv
|   |-- feature_importance.csv
|-- scripts/
|   |-- data_explorer.py
|   |-- feature_engineering.py
|   |-- data_cleaning.py
|   |-- train_models.py
|   |-- player_value_analysis.py
|   |-- sentiment_analysis.py
|   |-- visualizations.py
|-- tfenv/
```

## 6. Data Used

Primary input files:
- `Cricket Datasets/all_season_batting_card.csv`
- `Cricket Datasets/all_season_bowling_card.csv`
- `Cricket Datasets/IPL AUCTION.csv`
- `Cricket Datasets/Match.csv`
- `Cricket Datasets/Player.csv`
- `Cricket Datasets/Season.csv`

Core training dataset consumed by the app:
- `output/clean_ml_dataset.csv`

## 7. Feature Engineering and Data Pipeline

### 7.1 Data exploration
Script: `scripts/data_explorer.py`
- prints shape, columns, and sample rows for each major dataset.

### 7.2 Feature engineering
Script: `scripts/feature_engineering.py`
- aggregates batting by `(season, fullName)`:
  - runs, balls faced, fours, sixes,
  - strike rate = runs / ballsFaced * 100.
- aggregates bowling by `(season, fullName)`:
  - overs, conceded, wickets,
  - economy = conceded / overs.
- merges batting + bowling + auction data.
- creates advanced features:
  - `Impact_Score = runs + (20 * wickets) + (5 * Catches_Last_Season)`
  - `Experience_Score = International_Experience_Years * Matches_Last_Season`
  - `Form_Index = Batting_Average_Last_Season * Strike_Rate_Last_Season`
- outputs:
  - `output/final_ml_dataset.csv`

### 7.3 Data cleaning
Script: `scripts/data_cleaning.py`
- removes duplicate rows,
- replaces inf/-inf with NaN,
- drops missing values,
- removes outliers using IQR for all numeric columns,
- outputs:
  - `output/clean_ml_dataset.csv`

### 7.4 Model training
Script: `scripts/train_models.py`
- loads clean dataset,
- removes leakage/non-predictive columns,
- encodes categoricals via one-hot encoding,
- scales features with MinMaxScaler,
- trains XGBoost regressor,
- evaluates with MAE, RMSE, R2,
- exports feature importance:
  - `output/feature_importance.csv`

## 8. ML Approach Inside Main Application

Main app: `main_app.py`

Model-related behavior:
- loads `output/clean_ml_dataset.csv`,
- defines selected feature subset:
  - Runs_Last_Season,
  - Strike_Rate_Last_Season,
  - Batting_Average_Last_Season,
  - Wickets_Last_Season,
  - Economy_Last_Season,
  - International_Experience_Years,
  - Previous_Auction_Price_CR,
- scales with MinMaxScaler,
- trains two regressors:
  - XGBRegressor,
  - RandomForestRegressor.

Prediction path used for menu outputs:
- custom heuristic `predict_price(player)` function,
- combines batting score, bowling score, experience score, and previous price effect,
- clamps predicted value into a realistic range:
  - minimum 0.5 CR,
  - maximum 24.75 CR.

## 9. Complete Menu Options and How Each Works

The application displays 10 options.

### Option 1: Predict Auction Value
Purpose:
- manual value prediction from user-entered player stats.

Inputs:
- runs,
- strike rate,
- batting average,
- wickets,
- economy,
- experience years,
- previous auction price (CR).

Logic:
- sends inputs to `predict_price` heuristic.

Output:
- single predicted auction value in CR.

### Option 2: Sentiment Analysis
Purpose:
- classify cricket text sentiment.

Inputs:
- free-form cricket opinion text.

Logic:
- calls `scripts/sentiment_analysis.py` function `analyze_sentiment`.
- uses VADER compound score with cricket-positive lexicon extension.

Output:
- numeric sentiment score and label:
  - Positive if score >= 0.3,
  - Negative if score <= -0.3,
  - Neutral otherwise.

### Option 3: Analytics Graphs
Purpose:
- show dashboard-level visual analytics.

Logic:
- samples up to 1500 records,
- renders 2x4 grid of plots:
  - Auction Price Distribution,
  - Runs vs Price,
  - Strike Rate vs Price,
  - Batting Average vs Price,
  - Wickets vs Price,
  - Economy vs Price,
  - Experience vs Price,
  - Previous Price vs Current Price.

Output:
- matplotlib dashboard window.

### Option 4: Player Details
Purpose:
- full profile view for a specific player.

Inputs:
- player name (supports close-match suggestion).

Logic:
- `search_player` finds latest season row.
- prints profile sections:
  - basic info,
  - batting,
  - bowling,
  - fielding,
  - experience,
  - auction details.
- also computes predicted value via `predict_price`.

Output:
- structured player profile + predicted auction value.

### Option 5: Player Comparison
Purpose:
- compare two players side by side.

Inputs:
- player 1 name,
- player 2 name.

Logic:
- fetches latest-season rows for both,
- iterates all columns and prints values,
- handles missing values as N/A,
- computes predicted value for each player.

Output:
- full column-wise comparison + both predicted prices.

### Option 6: Player Value Analysis
Purpose:
- sub-menu for value trends and top players.

Sub-option 1: Top Players by Year
- input: auction year,
- groups by player and prints top 25 by mean sold price.

Sub-option 2: Player Auction Price Trend
- input: player name,
- groups selected player by season,
- prints and plots season-wise auction trend.

Sub-option 3: Back
- returns to main menu.

### Option 7: Value-for-Money Players
Purpose:
- find players with high performance per auction cost.

Inputs:
- auction year,
- role (`batsman` or `bowler`).

Logic:
- filters season and role-relevant data,
- computes:
  - `Performance = Runs_Last_Season + 20 * Wickets_Last_Season`
  - `Value_Index = Performance / Sold_Price_CR`
- ranks descending by Value_Index.

Output:
- top 10 value-for-money players with Performance, Sold_Price_CR, Value_Index.

### Option 8: Feature Importance
Purpose:
- show feature contribution from trained XGBoost model in the app.

Logic:
- reads `xgb.feature_importances_` for selected main features.

Output:
- sorted feature importance table.

### Option 9: Model Evaluation
Purpose:
- evaluate model fit quality in the app runtime.

Logic:
- predicts on scaled dataset using `xgb.predict(X_scaled)`,
- computes:
  - MAE,
  - RMSE,
  - R2 Score.

Output:
- printed evaluation metrics.

### Option 10: Exit
Purpose:
- terminate application safely.

Output:
- prints exit message and stops loop.

## 10. Technologies and Libraries Used

Language:
- Python 3.x

Core libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- vaderSentiment
- difflib (Python standard library)

Environment:
- local virtual environment (`tfenv`).

## 11. How to Run the Project

## 11.1 Prerequisites
- Python 3.10+ recommended.
- pip available.
- all required CSV files present in project folders.

## 11.2 Setup virtual environment (if needed)

PowerShell:

```powershell
python -m venv tfenv
.\tfenv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas numpy matplotlib scikit-learn xgboost vaderSentiment
```

## 11.3 Generate data and model artifacts (recommended full run)

```powershell
python scripts/data_explorer.py
python scripts/feature_engineering.py
python scripts/data_cleaning.py
python scripts/train_models.py
```

Optional analysis scripts:

```powershell
python scripts/visualizations.py
python scripts/player_value_analysis.py
```

## 11.4 Run the main interactive app

```powershell
python main_app.py
```

## 12. Expected Outputs

Created in `output/`:
- `final_ml_dataset.csv` from feature engineering,
- `clean_ml_dataset.csv` from data cleaning,
- `feature_importance.csv` from model training script.

Interactive outputs:
- terminal-based analytics and player reports,
- matplotlib graphs from options 3 and 6,
- sentiment scores and labels,
- valuation and model quality metrics.

## 13. Assumptions and Current Limitations

- Main app trains models at startup, which can increase launch time.
- Option 9 evaluates on same loaded dataset (not a holdout split) inside the main app.
- Option 1 manual prediction uses heuristic formula rather than direct model prediction.
- Data quality depends on source CSV completeness and consistency.
- No REST API or web UI currently; app is terminal-based.

## 14. Possible Improvements

- Add robust train/validation/test workflow in the main app.
- Persist trained models to disk and load instantly on startup.
- Add CLI argument mode for automation.
- Build web dashboard for non-technical stakeholders.
- Add unit tests and data validation checks.
- Introduce experiment tracking and model versioning.

## 15. Quick Run Summary

If you only want the app to run quickly (assuming output files already exist):

```powershell
.\tfenv\Scripts\Activate.ps1
python main_app.py
```

## 16. Credits

Project domain:
- IPL cricket auction analytics, performance intelligence, and valuation support.

If you want, this README can be further upgraded with:
- architecture diagram,
- sample screenshots,
- model performance table from actual run values,
- dataset data dictionary per column.
