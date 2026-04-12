import streamlit as st

st.set_page_config(page_title="Player Value Prediction")

st.title("⚽ Player Transfer Value Prediction")

age = st.number_input("Age", 15, 45, 25)
goals = st.number_input("Goals", 0, 100, 7)
assists = st.number_input("Assists", 0, 100, 5)
matches = st.number_input("Matches", 0, 200, 20)

predict = st.button("Predict Transfer Value")

if predict:
    value = (goals * 5) + (assists * 3) + (matches * 1.5) - (age * 2)

    st.success(f"💰 Estimated Value: € {round(value,2)} Million")