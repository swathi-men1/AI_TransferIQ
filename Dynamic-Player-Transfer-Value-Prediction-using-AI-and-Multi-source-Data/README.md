# ⚽ Dynamic Player Transfer Value Prediction

## 🚀 Deployment

🌐 Live Demo: https://dynamicplayertransfervalueprediction-qxthgcdvx3e85fx6ul8yee.streamlit.app

Click the link above to access the deployed application

---

## 📌 Introduction
This project aims to predict the transfer value of football players using machine learning techniques. The model analyzes various player attributes such as performance, age, and statistics to estimate their market value.
Instead of manually entering player details, the system allows the user to select a player and then predicts their transfer value based on historical performance data.
The project is implemented using Python and Jupyter Notebook, following a complete machine learning pipeline.

---

## 🎯 Problem Statement
Estimating a football player's transfer value manually is challenging because:
- It depends on multiple performance factors
- Manual estimation is subjective and inconsistent
- It is time-consuming
- Market trends continuously change
- Public sentiment influences player value

This project solves the problem using a data-driven machine learning approach.

---

## 💡 Proposed Solution
A machine learning model is developed to:
- Analyze player performance data
- Learn patterns from historical records
- Predict the transfer value of a selected player
- Integrates multi-source data (performance, sentiment, market data)
- Uses machine learning models for prediction
- Enhances accuracy using feature engineering and sentiment analysis

This helps in making better financial and strategic decisions in sports analytics.

---

## 🔥 Objective
The main objective of this project is to:
- Predict player transfer value accurately
- Analyze important features affecting player value
- Build a machine learning model for prediction

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## ⚙️ Workflow

1. **Data Collection**
   Data is collected from multiple sources:
   - Player performance data (goals, assists, matches)
   - Market value data
   - Social media data for sentiment analysis
   - Injury history data

2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical data
   - Feature selection
   - Removing duplicate and irrelevant data  
   - Structuring dataset for model training  

3. **Feature Engineering**
   - Creation of performance trend features  
   - Injury risk metrics  
   - Contract-related attributes  
   - Selection of important features affecting transfer value

4. **Sentiment Analysis**
   - Applied NLP techniques using VADER/TextBlob  
   - Extracted sentiment scores from player-related text data  
   - Converted sentiment into numerical features  
   - Integrated sentiment with dataset to improve prediction  

5. **Data Transformation**
   - Encoding categorical variables  
   - Normalizing numerical features  
   - Preparing data for model input 

6. **Model Development**
   - Regression-based machine learning model implemented  
   - Conceptually includes:
     - Time-series modeling (LSTM)
     - Ensemble methods (XGBoost/LightGBM)
   - Model learns relationships between features and transfer value

7.  **Model Evaluation**
   - Evaluated using:
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Ensures model accuracy and reliability 

8. **Prediction**
   - User selects a player  
   - Model processes the player's data  
   - Predicts transfer value  

 9. **Output**
   - Displays predicted transfer value  
   - Provides data-driven insights for player valuation  

---

## ▶️ How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/Neelima-Yadav/Dynamic_Player_Transfer_Value_Prediction.git
   ```

2. Navigate to the project folder:
   ```
   cd Dynamic_Player_Transfer_Value_Prediction
   ```

3. Install required libraries:
   ```
   pip install -r requirements.txt
   ```

4. Run the notebook:
   ```
   jupyter notebook
   ```

---

## 🔄 Dataset Details
- Multi-source dataset including:
  - Player performance metrics  
  - Market data  
  - Sentiment scores  
  - Injury records  
  - Preprocessed and feature-engineered for better performance  

---

##  🖥️ Final Output

### How it works:
- User selects a player  
- Model analyzes performance + sentiment + historical data  
- Displays predicted transfer value  

### Example:
- Selected Player: XYZ  
- Predicted Transfer Value = €XX Million

---

## 📸 Screenshots

###  Notebook Execution
![Execution](screenshots/execution.png)

###  Model Output
![Model Output](screenshots/output.png)

---

## 📁 Project Structure
---

Dynamic_Player_Transfer_Value_Prediction/
│
├── notebooks/
│   └── Milestones.ipynb
│
├── frontend/
│   └── index.html
│
├── docs/
│   ├── Infosys Presentation.pptx
│   └── report.pdf
│
├── screenshots/
│
└── README.md

---

## 🚀 Future Improvements
- Use advanced models for better accuracy
- Add real-time data
- Deploy as a web application

---

## 📚 Conclusion
This project demonstrates how machine learning can be used to predict football player transfer values. It helps in understanding the key factors influencing player pricing.
