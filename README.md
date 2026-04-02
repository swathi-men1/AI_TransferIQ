# AI Transfer IQ

AI Transfer IQ is an advanced, artificial intelligence-powered platform designed to predict football player market values with industry-standard precision. By integrating performance statistics, temporal trends, injury histories, and real-time social sentiment data, the system provides a holistic and highly accurate valuation of athletes across global transfer windows.

## Project Overview

The AI Transfer IQ platform utilizes a multi-stage analytical machine learning pipeline to generate actionable insights:

1. Exploratory Data Analysis (EDA): Provides insightful statistical visualizations of player attribute distributions, correlative behaviors, and general market value trends.
2. Industry-Level Preprocessing: Features advanced feature engineering (such as dynamic momentum and risk calculation) alongside Principal Component Analysis (PCA) based dimensionality reduction. This allows the models to handle complex, high-dimensional player data without succumbing to overfitting.
3. Time-Series Forecasting (LSTM Architecture):
   - Univariate LSTM: Tracks and forecasts historical transfer value trends using sequence data.
   - Multivariate LSTM: Integrates longitudinal performance metrics, injury occurrences, and public sentiment shifts over time to understand complex correlations.
   - Encoder-Decoder LSTM: Enables multi-step forecasting to predict valuation trajectories multiple transfer windows into the future.
4. Ensemble Modeling (XGBoost): A highly robust regressor model that operates as the primary valuation engine. It combines all engineered features and time-series outputs to provide the final, definitive transfer value prediction with high predictive power.
5. Cinematic Dashboard Frontend: A premium, glassmorphism-themed web interface built natively in HTML, CSS, and Vanilla JavaScript. It provides real-time interaction and data visualization via a lightweight, high-performance architecture without relying on heavy external frontend frameworks.

## Installation and Setup

### 1. System Requirements
- Python 3.9 or higher for the backend machine learning training, preprocessing routines, and API services.
- A modern web browser capable of rendering CSS variables and backdrop-filters to view the frontend interface.

### 2. Backend Setup
First, clone the repository to your local environment. Then, install the required Python dependencies to execute the analytical pipeline and start the machine learning server.

```bash
git clone <repository-url>
cd ai-transfer-iq
pip install -r requirements.txt
```

### 3. Frontend Setup
Because the frontend is built entirely with native, standard web technologies (HTML, CSS, JavaScript), there is no package manager (npm) or build step required. The application is completely portable. You simply navigate to the frontend directory and open the files in your browser.

## Project Structure

AI_TRANSFERIQ/
├── backend/
│   ├── models/
│   │   ├── lstm_tuned_final.keras
│   │   └── xgboost_final.pkl
│   ├── app.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
└── TODO.md

## Execution Sequence

To execute the full data pipeline from the ground up and subsequently launch the web application, follow this operational sequence:

### Step 1: Exploratory Data Analysis
Execute the EDA script to generate requisite correlation heatmaps and player distribution plots within the figures directory.

### Step 2: Data Preprocessing
Run the core preprocessing script to handle missing values, engineer derivative features (such as continuous value trends and sentiment momentum constraints), and compute the PCA reduction. This guarantees the definitive dataset structure for training.


### Step 3: LSTM Model Training
Train the recurrent neural network time-series models sequentially, including the Univariate, Performance, Sentiment, and Encoder-Decoder LSTM architectures.


### Step 4: XGBoost Ensemble Training
Train the final gradient-boosted ensemble regressor which consolidates the principal component features iteratively for the ultimate market value estimation.


### Step 5: Start the Backend API
Initialize the Python Flask server instance. The application will securely load the generated preprocessing pipeline artifacts and deep learning models into local memory, proceeding to listen for HTTP POST payloads on port 5000.


### Step 6: Launch the Dashboard
With the server actively listening, launch the presentation dashboard directly in your preferred browser to safely interact with the machine learning models.
```bash
cd frontend
start index.html
```
*(Note: On macOS, substitute the launch command with `open index.html`. On Linux environments, substitute with `xdg-open index.html`)*

## Model Architecture Details

The intelligence system is built employing a strict, robust, and highly modular machine learning architecture consisting of distinct layers:
- Data Ingestion Layer: Multi-source data unification encompassing raw performance statistics, physical injury timelines, and Natural Language Processing (VADER) sentiment derivation metrics from simulated social discussions.
- Preprocessing Layer: Implementation of industry-standard scaling strategies combined dynamically with Principal Component Analysis to effectively isolate the most statistically significant mathematical variance across all columns.
- Forecasting Layer: Advanced historical sequence modeling applied via recurrent neural networks (short-term memory architectures) comprehensively formatted for multivariate temporal analysis.
- Final Ensemble Layer: Advanced gradient-boosted decision trees (the XGBoost algorithm) structurally refining the final statistical prediction by properly evaluating the output sequences of the LSTMs against the static, dimensionality-reduced data features.

## Design Philosophy

The AI Transfer IQ user interface is deliberately engineered with a premium cinematic aesthetic, strictly focusing on maximizing native web rendering performance natively.
- Glassmorphism: Structural implementation of sleek, translucent user interface layers utilizing native CSS background blurring effects for high-end immersion.
- Dynamic Animations: Mathematically smooth state transitions and heavily hardware-accelerated CSS keyframe animations that yield an elite interaction experience.
- Functional Micro-interactions: Highly responsive cursor hover states, algorithmic fluid value numeration logic, and natively integrated charting graphics displaying deterministic multi-step future market values seamlessly over time.

## License
This project is securely licensed under the MIT License.
