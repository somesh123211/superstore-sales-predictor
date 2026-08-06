import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and metadata
model = joblib.load('sales_model.pkl')
meta = joblib.load('ui_metadata.pkl')

st.title("🚀 Superstore Sales Predictor")
st.markdown("Enter transaction details to predict the log-transformed Sales value.")

# Create input fields based on training features
col1, col2 = st.columns(2)

with col1:
    ship_mode = st.selectbox("Ship Mode", meta['Ship Mode'])
    segment = st.selectbox("Segment", meta['Segment'])
    region = st.selectbox("Region", meta['Region'])
    category = st.selectbox("Category", meta['Category'])
    sub_cat = st.selectbox("Sub-Category", meta['Sub-Category'])

with col2:
    state = st.selectbox("State", meta['State'])
    city = st.selectbox("City", meta['City'])
    quantity = st.number_input("Quantity", min_value=1, value=1)
    discount = st.slider("Discount", 0.0, 0.8, 0.0)
    postal = st.number_input("Postal Code", value=42420)

# Temporal features (defaults for prediction)
year, month, day, dow = 2023, 1, 1, 0

if st.button("Predict Sales"):
    input_data = pd.DataFrame([{
        'Ship Mode': ship_mode, 'Segment': segment, 'Region': region,
        'Category': category, 'Sub-Category': sub_cat, 'State': state, 'City': city,
        'Postal Code': postal, 'Quantity': quantity, 'Discount': discount,
        'Year': year, 'Month': month, 'Day': day, 'DayOfWeek': dow
    }])
    
    # The pipeline handles scaling and encoding internally
    log_prediction = model.predict(input_data)[0]
    actual_sales = np.expm1(log_prediction)
    
    st.success(f"Predicted Sales: ${actual_sales:.2f}")
    st.info(f"Log-Scaled Value: {log_prediction:.4f}")
