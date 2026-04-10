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

<h3 align="center">📸 Project Previews</h3>

<table align="center" border="0">
  <tr>
    <td align="center">
      <b>Image 1</b><br>
     <a href="https://private-user-images.githubusercontent.com/183088946/576606683-e9dd7e0f-318a-46b6-80a4-624a77d2b7a2.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjk4NzUsIm5iZiI6MTc3NTgyOTU3NSwicGF0aCI6Ii8xODMwODg5NDYvNTc2NjA2NjgzLWU5ZGQ3ZTBmLTMxOGEtNDZiNi04MGE0LTYyNGE3N2QyYjdhMi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzU5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iYzA2NjhmY2JjMDdkM2M2ODZiODI4NGNkNGZiYjQ3ZDBkZGZjMDg0NTEzZTQ3MDZhMTMyMmNmODJlZDNmYmJiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.o4tuz9phnKpWeXzmmxhnSFFw3uup-qi-r-qpaqML31c"></a>
        <img src=""  alt="Image 1">
      </a>
    </td>
    <td align="center">
      <b>Image 2</b><br>
      <a href="https://private-user-images.githubusercontent.com/183088946/576589223-37f802e7-699d-49b6-8ffc-5143ba229a50.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjg4ODMsIm5iZiI6MTc3NTgyODU4MywicGF0aCI6Ii8xODMwODg5NDYvNTc2NTg5MjIzLTM3ZjgwMmU3LTY5OWQtNDliNi04ZmZjLTUxNDNiYTIyOWE1MC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzQzMDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04ODI2YzkwNzhkYjFlMWM0ZWZmZjk0NjQwMTIxOTM3ODYxZGYxMzk4ZjY4YmE5ZDUyNjhkYjBhZWU2MWQ5NThlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.GQhGXMwG2qo9zQjucrEEZq0baJdEbzOKXBBprYzK1HE">
        <img src="IMAGE_URL_2.jpg" width="250" alt="Image 2">
      </a>
    </td>
    <td align="center">
      <b>Image 3</b><br>
      <a href="https://private-user-images.githubusercontent.com/183088946/576589857-1169f81d-cc09-4678-9abd-89c70251efb3.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjk4NzUsIm5iZiI6MTc3NTgyOTU3NSwicGF0aCI6Ii8xODMwODg5NDYvNTc2NTg5ODU3LTExNjlmODFkLWNjMDktNDY3OC05YWJkLTg5YzcwMjUxZWZiMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzU5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hZWEwYzNhMTExN2E2NzVjY2E0Yjc4NGRlNGU3NzVhZTFmMzk0M2UzNDljNTcwNzYxMTY3YmQ2YjZhZGZjMjYyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.8eEKZdOw1HyQcxp1dhicfZh8kwAS9MWa3QD-JYo3yg8">
        <img src="IMAGE_URL_3.jpg" width="250" alt="Image 3">
      </a>
    </td>
  </tr>
  
  <tr>
    <td align="center">
      <b>Image 4</b><br>
      <a href="https://private-user-images.githubusercontent.com/183088946/576590129-3fcc5df5-7268-485e-a5b5-1b418ec8363f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjk4NzUsIm5iZiI6MTc3NTgyOTU3NSwicGF0aCI6Ii8xODMwODg5NDYvNTc2NTkwMTI5LTNmY2M1ZGY1LTcyNjgtNDg1ZS1hNWI1LTFiNDE4ZWM4MzYzZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzU5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yYzQ0MDNiMjY3N2Y2MTliNTIyODUxNjg5MzczOTAyMzgxNjM5YTA3ZTY4MDI4OGEzZDE5MjBjN2U3NDk2ODAyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.46JNFx4lX6UV8uffAmSEwMoZkG0_dOeGepB6wqZGm-8">
        <img src="IMAGE_URL_4.jpg" width="250" alt="Image 4">
      </a>
    </td>
    <td align="center">
      <b>Image 5</b><br>
      <a href="https://private-user-images.githubusercontent.com/183088946/576593305-070cfd7b-a531-46b7-842d-98a95358c608.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjk4NzUsIm5iZiI6MTc3NTgyOTU3NSwicGF0aCI6Ii8xODMwODg5NDYvNTc2NTkzMzA1LTA3MGNmZDdiLWE1MzEtNDZiNy04NDJkLTk4YTk1MzU4YzYwOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzU5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ODZhMDgwMThlOWIyNDE5ZThmYzVkOGMzMDljMmZiMWE4ZmY4NmI5OWYxMDMyZjI4MjhkZDQ0ZWMwMDY5ZmQyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.Ot_zwchNcyi7Uhl6P3Hk7UMAx7EyWpmUnvP7ydaup8g">
        <img src="IMAGE_URL_5.jpg" width="250" alt="Image 5">
      </a>
    </td>
    <td align="center">
      <b>Image 6</b><br>
      <a href="https://private-user-images.githubusercontent.com/183088946/576592147-d5d90438-a519-43f9-8eb5-e58db28abcaa.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjk4NzUsIm5iZiI6MTc3NTgyOTU3NSwicGF0aCI6Ii8xODMwODg5NDYvNTc2NTkyMTQ3LWQ1ZDkwNDM4LWE1MTktNDNmOS04ZWI1LWU1OGRiMjhhYmNhYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzU5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ODVlMjhmZmI1YWVhYzk1OTVhMmYyNTllNGQ4ZDE3Zjc4NDExNjE0MTJhYzVmN2ZhOTM4MjFiMTYyN2JiMWVjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.HZMbVDyxowuhZcqdNoCLMgWnYl-VMeQYwnwoQpSAcs8">
        <img src="IMAGE_URL_6.jpg" width="250" alt="Image 6">
      </a>
    </td>
  </tr>

  <tr>
    <td></td> <td align="center">
    <b>Image 7</b><br>
      <a href="https://private-user-images.githubusercontent.com/183088946/576592276-e2c90b99-3f15-4b95-9afc-8a67bbbb4a13.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzU4Mjk4NzUsIm5iZiI6MTc3NTgyOTU3NSwicGF0aCI6Ii8xODMwODg5NDYvNTc2NTkyMjc2LWUyYzkwYjk5LTNmMTUtNGI5NS05YWZjLThhNjdiYmJiNGExMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxMFQxMzU5MzVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xZjQxNGVhNmRmYTViM2U0NzhiOTEwNzc5MTJmMTEyYWRmNTExMjcyMzIxODczMWE3MmY1MmM0ZWIwZDNkZmRhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.vy2Jtxjkmb8AQGUv0XY3DSVTEg2vf8HANrQ8njyqmQQ">
        <img src="IMAGE_URL_7.jpg" width="250" alt="Image 7">
      </a>
    </td>
    <td></td> </tr>
</table>

## 🚀 Core Modules & Features

TransferIQ Pro is divided into four highly interactive modules to serve different scouting and analytical needs:

### 🧪 1. Player Lab (Interactive Sandbox)
A dynamic testing ground to evaluate individual players.
* **Parameter Tuning:** Adjust sliders for Age, Performance, Injury Risk, Sentiment, and Contract Years.
* **Instant Valuation:** Get real-time Market Value (in EUR and INR).
* **3-Year Trajectory:** Visualizes future depreciation or growth via interactive line charts.
* **Explainable AI (XAI):** Generates automated text insights, risk verdicts (Buy/Hold/Pass), and Confidence Scores based on the player's profile matrix.

### ⚔️ 2. Compare Mode (Head-to-Head)
Directly pit two players against each other to analyze market dominance.
* **Value Gap Analysis:** Automatically calculates the monetary difference and premium percentage between Player A and Player B.
* **Attribute Duels:** Visual progress bars comparing Performance, Durability, and Experience side-by-side.
* **Dual-Radar Charts:** Overlays the statistical footprint of both players on a single radar chart to easily spot strengths and weaknesses.

### 📦 3. Data Scan (Bulk Processing)
Built for club scouts and data analysts to evaluate entire datasets at once.
* **CSV & Text Upload:** Upload scouting datasets or paste raw comma-separated text.
* **Neural Scan Engine:** Processes hundreds of players simultaneously, calculating their base values and 3-year forecasts.
* **Interactive Grid:** Renders an actionable data table highlighting the top prospects, trend icons (📈/📉), and risk tiers.
* **Export Reports:** One-click functionality to download the processed AI evaluations as a structured CSV report.

### 📊 4. Model Insights (Live Telemetry)
A transparent look into the brain of TransferIQ Pro.
* **Live Telemetry:** Tracks total scans processed, the highest valuation of the session, and live inference speed (ms).
* **Architecture Transparency:** Displays model composition (XGBoost Ensemble + LSTM Recurrent Heuristics) and real-time R² accuracy.
* **Feature Importance Matrix:** A visual bar chart indicating exactly how much weight the AI assigns to Performance, Age, Injury, and Sentiment.

---

## 🧠 System Architecture

```text
User Input (UI / CSV)
      ↓
XGBoost Model → Current Value Prediction
      ↓
Domain Logic Layer (Position, Contract, Age Multipliers)
      ↓
LSTM / Sequence Models → Future Trend Forecast
      ↓
Explainable AI Module → Generates Verdicts & Insights
      ↓
Dashboard Visualization (Chart)
----
```
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

Live Deployment Link:https://transferiq-pro.onrender.com/

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
