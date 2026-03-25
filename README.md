# AI Transfer IQ

## Project Overview

AI Transfer IQ is an advanced, high-precision analytics platform designed to solve one of the most complex challenges in modern sports: the accurate valuation of football player market prices. In an era where transfer fees are increasingly volatile, this platform provides stakeholders—such as scouts, clubs, and agencies—with institutional-grade insights driven by machine learning.

The core objective of AI Transfer IQ is to move beyond traditional, static valuation methods. By integrating multi-dimensional data streams, including on-field performance, physical durability, social sentiment, and historical market trends, the system predicts the "Next Market Surge"—the expected change in a player's valuation over the coming period.

---

## Technical Deep Dive: The Analytical Core

### 1. Hybrid XGBoost Model
The system's heartbeat is a Hybrid XGBoost (eXtreme Gradient Boosting) engine. Unlike standard regression models that predict absolute values, AI Transfer IQ utilizes a **Log-Return methodology**. This approach is superior for financial and market forecasting as it:
- Normalizes the distribution of valuation changes.
- Accounts for the asymmetric nature of market surges vs. drops.
- Focuses on the "Transfer IQ Score"—the logarithmic delta between current and future market states.

### 2. DART Booster Implementation
The engine leverages **DART (Dropout Additive Regression Trees)**. By incorporating dropout during the boosting process, the model prevents over-specialization in outlier cases (like "superstars") and maintains balanced accuracy across the entire player database, from emerging talents to established veterans.

### 3. NLP & Social Sentiment Integration
Market value is often driven as much by perception as by performance. AI Transfer IQ integrates a sentiment analysis pipeline using:
- **TextBlob & VADER**: For real-time scoring of news articles and social media trends.
- **Transformers (NLP)**: For deep semantic analysis of player-related narratives, converting qualitative buzz into quantitative market signals.

---

## Dataset & Feature Engineering

The platform processes thousands of data points across a variety of features, each meticulously engineered to provide predictive power:

### Performance Metrics
- **Performance Rating**: An aggregated score (1-10) reflecting a player's consistency and impact on the pitch.
- **Goals & Assists (G/A)**: The primary offensive output markers.
- **Minutes Played**: A proxy for a player's reliability and status within a squad.
- **Form**: A derived feature calculating (Goals + Assists) / (Minutes Played), providing a per-minute productivity index.

### Physical & Contractual Data
- **Injury Ratio**: Calculated as Days Injured / Minutes Played, this feature assesses the "risk-adjusted" value of a player.
- **Contract Duration**: A critical market lever; players with shorter contracts often face higher volatility in their transfer valuations.

### Market Trends
- **3-Month Rolling Averages**: Captures recent momentum in performance and market value.
- **Volatility Clusters**: Identifies historical patterns of rapid value spikes, helping the model anticipate the next surge.


## Project Architecture & File Descriptions

The AI Transfer IQ platform is organized into a modular, multi-layered architecture that separates data analysis, model intelligence, and production delivery.

### 1. Root Directory: The Production Bridge
- **app.py**: The central nervous system of the production environment. This Flask-based server manages the integration between the frontend dashboard and the trained XGBoost model. It handles sub-second inference requests, performs real-time feature transformation (using system scalers), and serves the entire web interface.
- **requirements.txt**: A comprehensive manifest of all project dependencies, including Flask for the API, XGBoost for intelligence, and NLP libraries (TextBlob, VADER) for sentiment scoring.

### 2. Frontend Layer (frontend/): Cinema-Grade User Experience
- **index.html**: Defines the high-tech, SEO-optimized structure of the Transfer IQ dashboard. It includes the "Market Value Engine" form and the dynamic result visualization area.
- **styles.css**: Implements the "Visual Wow" design language. It uses premium CSS tokens for glassmorphism, cinematic animations (like the IQ pulse and shimmering gradients), and a fully responsive grid system.
- **script.js**: Manages the bridge between the user and the AI. It handles form interactions, UI state transitions, and real-time communication with the Flask `/predict` API.

### 3. Analytical Layer (notebooks/): The Intelligence Lifecycle
- **1_eda.py (Exploratory Data Analysis)**: The foundation of the analytical pipeline. This script identifies correlation clusters between performance metrics and market value, ensuring only the most predictive features are used for training.
- **2_preprocess.py (Data Pipeline)**: Handles the heavy lifting of feature engineering. It performs normalization, scales market values, and one-hot encodes categorical data (like player positions) for mathematical optimization.
- **3_evaluate.py (Model Benchmarking)**: A rigorous testing script that evaluates model performance using RMSE, MAE, and R2 scores, ensuring that only the most accurate version of the model reaches production.
- **XGboost_train.py (Hybrid Training)**: The core training script for the XGBoost engine. It implements the Log-Return methodology and DART booster, outputting the final `xgboost_model.json`.
- **lstm_train.py (Sequential Modeling)**: Explores sequential dependencies in player valuations using Long Short-Term Memory networks, providing a secondary benchmark for trend analysis.
- **sentiment_analysis.py (NLP Pipeline)**: Leverages VADER and TextBlob to quantify qualitative "market buzz" into a numerical `social_sentiment_score`, adding a critical social dimension to the valuation.

### 4. Intelligence & Infrastructure Layer
- **models/ (XGBoost Models)**: Contains the serialized intelligence of the project. The `.json` files represent the trained decision trees capable of predicting complex market surges.
- **reports/metrics/ (Scalers & Data)**: Houses the `scaler_y.pkl` and `scaler_X.pkl` files used by both the training pipeline and the production server to maintain data consistency.
- **data/ (Raw & Processed Data)**: The storage layer for the datasets that fuel the IQ engine, including the final `transferiq_processed.csv`.

---

## The Analytical Pipeline

### Phase 1: Exploratory Data Analysis (EDA)
The pipeline begins with `1_eda.py`, which identifies correlation clusters and outliers in the raw dataset. This step ensures that the training data is clean and that the features have the highest possible predictive relevance.

### Phase 2: Preprocessing & Scaling
In `2_preprocess.py`, data is normalized using **StandardScaler**. The current market value is scaled into a stable range, and categorical variables (like field positions) are one-hot encoded to ensure compatibility with mathematical optimization.

### Phase 3: Sentiment Analysis
Using `sentiment_analysis.py`, textual data is ingested and converted into a `social_sentiment_score`. This score is merged into the main dataset, providing a "social buzz" dimension to the valuation.

### Phase 4: Training & Evaluation
The `XGboost_train.py` script executes the hybrid training sequence. It outputs serialized model files (`.json`) and scikit-learn scalers (`.pkl`), which are then used by the production environment for real-time inference.

---

## Production Architecture

### Backend: Flask Inference Server
The `app.py` server acts as the production bridge. It:
- Serves the static web dashboard.
- Hosts the `/predict` API endpoint.
- Handles real-time feature transformation (scaling user inputs to match the model's training distribution).

### Frontend: Cinema-Grade Dashboard
The **AI Transfer IQ Dashboard** is designed with "Visual Wow" principles. 
- **Glassmorphism**: A UI design language that uses translucent elements and backdrop-blurs to create a premium, futuristic feel.
- **Cinematic Animations**: Smooth transitions and interactive data reveal sequences ensure that the complex data is presented in an engaging, professional manner.

---

## Detailed Installation & Usage

### 1. Environment Setup
Install the complete set of analytical and production dependencies:
```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution
To replicate the full IQ Analytical Sequence:
- **Analyze Data**: `python notebooks/1_eda.py`
- **Preprocess**: `python notebooks/2_preprocess.py`
- **Sentiment Scoring**: `python notebooks/sentiment_analysis.py`
- **Train Model**: `python notebooks/XGboost_train.py`

### 3. Deploying the Environment
To launch the live valuation dashboard:
```bash
python app.py
```
Visit `http://localhost:5000` to access the interactive interface.

---

© 2026 AI Transfer IQ. The future of football valuation, engineered for excellence.
