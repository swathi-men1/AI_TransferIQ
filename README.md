# TransferIQ: Dynamic Player Transfer Value Prediction

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.95+-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20XGBoost-orange.svg" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Design-Responsive%20UI-success.svg" alt="UI/UX">
</div>

## 📌 Project Overview
**TransferIQ** is an advanced AI-powered data science platform designed to predict football player transfer market values. The system utilizes historical performance metadata, comprehensive injury records, and real-time social media sentiment analysis to deliver highly accurate valuations.

By fusing **LSTM (Long Short-Term Memory)** neural networks for time-series forecasting with **Ensemble Models (XGBoost/LightGBM)** to reduce overfitting, TransferIQ achieves robust and transparent transfer value predictions.

---

## ✨ Key Features
- **Multi-Source Data Pipelines**: Aggregates StatsBomb performance data, Transfermarkt financial values, and Twitter APIs.
- **Sentiment Analysis Processing**: Incorporates VADER and TextBlob NLP algorithms to determine off-pitch public perception impact.
- **Hybrid AI Architecture**: Utilizes an Encoder-Decoder LSTM network stacked with an XGBoost/LightGBM weighted ensemble.
- **Confidence Intervals**: Employs model discrepancy metrics to output reliable confidence ranges (Lower/Upper Bounds) for every valuation.
- **Full-Stack Ecosystem**: Includes a complete RESTful prediction API (FastAPI) alongside a beautifully crafted, self-contained interactive web application.

---

## 🛠️ Technology Stack
- **Backend & API**: Python, FastAPI, Uvicorn
- **Machine Learning**: TensorFlow, Keras, XGBoost, LightGBM, Scikit-learn
- **Data Engineering**: Pandas, Numpy, NLTK (VADER/TextBlob)
- **Frontend App**: HTML5, Vanilla JavaScript, Vanilla CSS, Chart.js

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed and clone this repository.
```bash
git clone https://github.com/yourusername/transfer-iq.git
cd transfer-iq
```

### 2. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Prediction Server
Start the local FastAPI backend to load the ML models into memory:
```bash
python start_server.py
```
*The API will be available at `http://localhost:8000/docs`.*

### 4. Launching the Interactive Website
Simply open the unified application interface in any modern browser:
```bash
# Windows
start transferiq_website.html
# macOS
open transferiq_website.html
```
*Or host it using Python's built-in server `python -m http.server 8080`.*

---

## 📅 Project Roadmap & Tracker
Included in this repository is the complete 8-week developmental timeline we followed, hosted as its own application!
To view the timeline:
```bash
cd project_tracker
start_tracker.cmd
```
Access the dashboard at `http://localhost:8081`.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the models or the dashboard styling:
1. Fork the project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

> *Developed for the 2026 Data Science Academic Capstone*.
