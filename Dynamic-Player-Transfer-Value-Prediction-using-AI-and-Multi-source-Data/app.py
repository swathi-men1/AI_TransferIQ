import streamlit as st

st.title("⚽ Player Transfer Value Prediction")

st.write("This project predicts the market value of a football player based on input features.")

# Example inputs
age = st.number_input("Enter Player Age", min_value=15, max_value=45)
goals = st.number_input("Enter Number of Goals", min_value=0)

if st.button("Predict"):
    st.success(f"Estimated Value: Based on Age {age} and Goals {goals}")