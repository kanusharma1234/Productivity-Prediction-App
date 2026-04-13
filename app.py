import pandas as pd 
import joblib
import numpy as np
import streamlit as st

# Page Config
st.set_page_config(page_title="Productivity Predictor", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    h1, h2, h3 {
        color: #38bdf8;
        text-align: center;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: black;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
model = joblib.load('XGB_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('Label_scaler(Addiction_level).pkl')

# Title
st.title("📊 Productivity Prediction System")
st.markdown("### 🔍 Predict your productivity based on daily habits")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 25)
    daily_screen_time = st.number_input("Screen Time (hrs)", 0.0, 24.0, 5.0)
    social_media_hours = st.number_input("Social Media (hrs)", 0.0, 24.0, 3.0)
    study_hours = st.number_input("Study Hours", 0.0, 24.0, 4.0)

with col2:
    sleep_hours = st.number_input("Sleep Hours", 3.0, 12.0, 7.0)
    notifications_per_day = st.number_input("Notifications", 0, 1000, 100)
    focus_score = st.slider("Focus Score", 1.0, 100.0, 50.0)
    addiction_level = st.selectbox("Addiction Level", ["Low", "Medium", "High"])

# Predict Button
if st.button("🚀 Predict Productivity"):

    addiction_encoded = le.transform([addiction_level])[0]

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

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction = np.clip(prediction, 0, 100)

    score = prediction[0]

    st.markdown("---")
    st.subheader("📈 Prediction Result")

    # Show metric
    st.metric(label="Productivity Score", value=f"{score:.2f}")

    # Progress bar
    st.progress(int(score))

    # Interpretation
    if score > 70:
        st.success("🔥 High Productivity")
    elif score > 40:
        st.warning("⚖️ Moderate Productivity")
    else:
        st.error("📉 Low Productivity")

    st.markdown("### 💡 Tip:")
    st.info("Improve sleep, reduce screen time, and manage notifications for better productivity!")
