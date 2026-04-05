import pandas as pd 
import joblib
import numpy as np
import streamlit as st

# Load the model
model = joblib.load('XGB_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('Label_scaler(Addiction_level).pkl')

# Define the Streamlit app
def main():
    st.title("📊 Productivity Prediction App")
    st.write("Enter the following details to predict productivity score:")

    # Get user input
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    daily_screen_time = st.number_input("Daily Screen Time (hours)", min_value=0.0, max_value=24.0, value=5.0)
    social_media_hours = st.number_input("Social Media Hours (hours)", min_value=0.0, max_value=24.0, value=3.0)
    study_hours = st.number_input("Study Hours (hours)", min_value=0.0, max_value=24.0, value=4.0)
    sleep_hours = st.number_input("Sleep Hours (hours)", min_value=3.0, max_value=12.0, value=7.0)
    notifications_per_day = st.number_input("Notifications per Day", min_value=0, max_value=1000, value=100)
    focus_score = st.slider("Focus Score", min_value=1.0, max_value=100.0, value=50.0)

    # FIX: use selectbox instead of text input
    addiction_level = st.selectbox("Addiction Level", ["Low", "Medium", "High"])

    if st.button('Predict'):
        
        # Encode addiction level correctly
        addiction_encoded = le.transform([addiction_level])[0]

        # IMPORTANT: same order as training data
        input_data = np.array([[
            age,
            daily_screen_time,
            social_media_hours,
            study_hours,
            sleep_hours,
            notifications_per_day,
            focus_score,
            addiction_encoded
        ]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        prediction = np.clip(prediction, 0, 100)

        # Show result
        st.success(f"Predicted Productivity Score: {prediction[0]:.2f}")

        # Optional interpretation
        if prediction[0] > 70:
            st.info("🔥 High Productivity")
        elif prediction[0] > 40:
            st.info("⚖️ Moderate Productivity")
        else:
            st.warning("📉 Low Productivity")

# Run app
if __name__ == "__main__":
    main()