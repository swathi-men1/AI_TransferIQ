<div align="center">
  
# ⚽ TransferIQ Pro
**AI-Powered Football Valuation & Future Trend Forecasting Ecosystem**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?style=for-the-badge&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge)

</div>

---

## 📖 About The Project

**TransferIQ Pro** is an advanced, data-driven football analytics platform. In the real-world football market, player values are often manipulated by hype. This project cuts through the noise by integrating real-time performance metrics, injury risk indexes, and Natural Language Processing (NLP) sentiment to predict a player's exact current market value. 

Going a step further, it utilizes a Deep Learning **LSTM** neural network to forecast the player's financial trajectory for the next 3 years.

### 📸 Dashboard Sneak Peek
> *(Add your dashboard screenshot here)*
> ![TransferIQ Dashboard Placeholder](https://via.placeholder.com/800x400?text=Dashboard+Screenshot+Here)

---

## ✨ Premium Features

- ⚡ **Real-Time Base Valuation:** Calculates the exact current transfer value using an optimized **XGBoost Regressor**.
- 📈 **Time-Series Forecasting:** Predicts the next 3 years of market value trends using **Deep Learning (LSTM)** on reverse-engineered sequences.
- 🧠 **NLP Sentiment Integration:** Analyzes public hype and media rumors (via VADER & TextBlob) to adjust player valuations.
- 📊 **Interactive Analytics Dashboard:** A sleek, dark-themed UI built with **Chart.js** that dynamically updates line and radar charts.
- 🛡️ **Explainable AI (XAI):** Generates live confidence scores, investment risk levels, and career stage categorization to justify the AI's output.

---

## 🛠️ Tech Stack

| Category | Technologies Used |
| :--- | :--- |
| **Machine Learning** | `XGBoost`, `scikit-learn`, `Pandas`, `NumPy` |
| **Deep Learning** | `TensorFlow`, `Keras (LSTM)` |
| **Backend API** | `Python`, `Flask`, `Flask-CORS` |
| **Frontend / UI** | `HTML5`, `CSS3`, `Vanilla JavaScript`, `Chart.js` |
| **NLP** | `NLTK (VADER)`, `TextBlob` |

---

## 📁 Repository Structure

```text
TransferIQ/
├── Dataset/                        # Raw, cleaned, and processed data files
│   ├── Processed_Dataset.csv
│   ├── only_sentiment_data.csv
│   └── ...
├── Model/                          # Model training scripts and generated sequences
│   ├── train_lstm_py.py
│   ├── train_xgboost_py.py
│   └── ... 
├── src/                            # Data collection and NLP scripts
│   └── sentiment_analysis.py       
├── templates/                      # Flask HTML Templates
│   └── index.html                  # TransferIQ Dashboard UI
├── app.py                          # Main Flask API Server
├── transferiq_lstm.keras           # Final Trained LSTM Model
├── transferiq_model.json           # Final Trained XGBoost Model
└── requirements.txt                # Python Dependencies
