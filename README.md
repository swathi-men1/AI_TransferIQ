<div align="center">

# ⚽ TransferIQ Pro  
### AI-Powered Football Valuation & Future Trend Forecasting System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge)

</div>

---

## 📖 About The Project

**TransferIQ Pro** is an AI-powered football analytics platform designed to predict player transfer values using data-driven techniques.

In real-world football markets, player valuations are often influenced by hype, media narratives, and agent negotiations. This system eliminates subjectivity by combining:

- 📊 Performance Metrics  
- 🩹 Injury Risk Analysis  
- 🌍 Public Sentiment (NLP)

The system not only predicts a player’s **current market value** but also forecasts their **future valuation trend for the next 3 years** using deep learning.

---

## 📸 Dashboard Preview
<img width="1350" height="641" alt="image" src="https://github.com/user-attachments/assets/6988dcb7-da53-4d3f-85d4-6fd15b617586" />


![Dashboard](./assets/dashboard.png)

---

## 🚀 Key Features

- ⚡ **Real-Time Valuation**  
  Predicts current transfer value using **XGBoost Regressor**

- 📈 **Future Trend Forecasting**  
  Uses **LSTM (Deep Learning)** to predict 3-year value trajectory

- 🧠 **Explainable AI (XAI)**  
  Provides:
  - Confidence Score  
  - Risk Level  
  - Market Tier  
  - Career Stage  
  - AI-based Explanation  

- 📊 **Interactive Dashboard**  
  Built using **Chart.js**, includes:
  - Line Chart (Trend)
  - Radar Chart (Player Profile)
  - Real-time updates

- ⚔️ **Player Comparison Mode**  
  Compare two players side-by-side

- 📦 **Scenario Simulation**  
  - Best Case  
  - Worst Case  

- 📥 **Export Reports**  
  Download prediction results as JSON

---

## 🧠 System Architecture

```text
User Input (UI)
      ↓
XGBoost Model → Current Value Prediction
      ↓
LSTM Model → Future Trend Forecast
      ↓
Domain Logic Layer (Position, Contract, Age)
      ↓
Explainable AI Module
      ↓
Dashboard Visualization (Chart.js)
```
----
⚙️ How It Works
```
1. User inputs player attributes (age, performance, injury, sentiment)
2. XGBoost predicts current transfer value
3. LSTM analyzes temporal patterns for future forecasting
4. Business logic adjusts realism (position, contract, age)
5. Backend generates insights and explanations
6. Frontend displays results interactively
```
----
📊 Example Output:
* 💰 Current Value: €20.2M
* 📉 Trend: Declining
* ⚠️ Risk: Medium
* 🏆 Tier: High
* 🧬 Career Stage: Peak Player
* 🤖 Insight: Balanced profile with moderate growth potential

----
## 🛠️ Tech Stack

| Category | Technologies Used |
| :--- | :--- |
| **Machine Learning** | `XGBoost`, `scikit-learn`, `Pandas`, `NumPy` |
| **Deep Learning** | `TensorFlow`, `Keras (LSTM)` |
| **Backend API** | `Python`, `Flask`, `Flask-CORS` |
| **Frontend / UI** | `HTML5`, `CSS3`, `Vanilla JavaScript`, `Chart.js` |
| **NLP** | `NLTK (VADER)`, `TextBlob` |

----
## 📁 Project Structure
```
TransferIQ/
├── Dataset/
│   ├── Cleaned_dataset.csv
│   ├── Final_raw_dataset.csv
│   ├── Processed_Dataset.csv
│   ├── only_sentiment_data.csv
│   ├── player_injuries.csv
│   └── transfermarkt_player_values.csv
├── Model/
│   ├── X_lstm_data.npy
│   ├── y_lstm_target.npy
│   ├── lstm_prep_py.py
│   ├── train_lstm_py.py
│   └── train_xgboost_py.py
├── src/
│   └── sentiment_analysis.py
├── index.html
├── app.py
├── transferiq_lstm.keras
├── transferiq_model.json
└── requirements.txt
```
----

🧪 Installation & Setup
git clone https://github.com/Ankit-298/TransferIQ-Pro
cd TransferIQ-Pro

pip install -r requirements.txt

python app.py

👉 Open browser:
http://127.0.0.1:5000

Deployment Link:https://transferiq-pro.onrender.com/

## 🚀 Future Improvements

* 🔗 **Real-time API Integration:** Fetching live data from Transfermarkt or FIFA datasets.
* ⏱️ **Match-Level Time-Series:** Utilizing granular match-by-match data for even more accurate LSTM training.
* 🧠 **Advanced Deep Learning:** Exploring Transformer-based architectures for better sequential forecasting.
* ☁️ **Cloud Deployment:** Scaling the Flask backend and models using AWS or Google Cloud Platform.
* 📱 **Mobile Responsive UI:** Optimizing the Chart.js dashboard for seamless mobile and tablet viewing.

🤝 Contribution

Contributions are welcome! Feel free to fork and improve the system.

📜 License

This project is for educational and research purposes.

<div align="center">

🔥 TransferIQ Pro – Where Data Meets Football Intelligence

</div> ```
