import streamlit as st

st.set_page_config(page_title="Player Value Prediction", layout="centered")

st.title("⚽ Dynamic Football Player Transfer Value Prediction")
st.write("This app demonstrates a simple prediction system for estimating player value based on performance metrics.")

st.header("Enter Player Details")

age = st.number_input("Age", min_value=15, max_value=45, value=25)
goals = st.number_input("Goals Scored", min_value=0, value=10)
assists = st.number_input("Assists", min_value=0, value=5)
matches = st.number_input("Matches Played", min_value=0, value=20)
rating = st.slider("Player Rating (0-10)", 0.0, 10.0, 7.0)

st.divider()

# Simple dummy logic (replace later with ML model if you have one)
if st.button("Predict Transfer Value"):

    value = (
        (goals * 5) +
        (assists * 3) +
        (matches * 1.5) +
        (rating * 10) +
        (age * -2)
    )

    st.success(f"💰 Estimated Transfer Value: € {round(value, 2)} Million")

    st.info("Note: This is a demo model (rule-based). You can replace it with ML model later.")