import streamlit as st
import pandas as pd
import numpy as np
import requests

# Backend API URL
API_URL = "http://localhost:8001"

st.set_page_config(page_title="Transfer IQ", layout="wide")
st.title("🏏 Transfer IQ - IPL Player Value Prediction")

st.markdown("""
A machine learning system to predict IPL player transfer values using:
- XGBoost models
- LSTM neural networks
- Ensemble predictions
""")

st.sidebar.header("⚙️ Options")

# Training section
st.header("📚 Model Training")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Auction Data")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"File shape: {df.shape}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

with col2:
    st.subheader("Actions")
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                response = requests.post(
                    f"{API_URL}/train",
                    json={"data": {"sample": "data"}},
                    timeout=5
                )
                if response.status_code == 200:
                    st.success("✅ Model trained successfully!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Backend returned error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to backend at http://localhost:8000")
                st.info("Make sure backend is running: `python backend/api.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Prediction section
st.header("🔮 Make Predictions")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Player Details")
    player_name = st.text_input("Player Name")
    role = st.selectbox("Playing Role", ["Batsman", "Bowler", "All-rounder", "Unknown"])
    team = st.text_input("Team")
    
with col2:
    st.subheader("Quick Predict")
    if st.button("🎯 Predict Value", use_container_width=True):
        with st.spinner("Fetching prediction..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"input": f"{player_name}-{role}-{team}"},
                    timeout=5
                )
                if response.status_code == 200:
                    result = response.json().get("prediction", "No prediction")
                    st.success(f"💰 Predicted Value: {result}")
                else:
                    st.error(f"❌ Backend error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to backend")
                st.info("Start the backend with: `python backend/api.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Status section
st.sidebar.divider()
st.sidebar.subheader("📊 System Status")
try:
    response = requests.get(f"{API_URL}/", timeout=2)
    st.sidebar.success("✅ Backend: Connected")
except:
    st.sidebar.error("❌ Backend: Disconnected")
