import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the mapping from numerical drug codes to names
drug_mapping = {1: 'drugA', 2: 'drugB', 3: 'drugC', 4: 'drugX', 5: 'DrugY'}

# --- Streamlit UI ---
st.set_page_config(page_title="Drug Classification App", layout="centered")
st.title('Drug Recommendation System')
st.write('Enter patient details to get a drug recommendation.')

# Input widgets for user data
age = st.slider('Age', min_value=15, max_value=74, value=25)
sex = st.radio('Sex', options=['Male', 'Female'])
bp = st.selectbox('Blood Pressure (BP)', options=['High', 'Normal', 'Low'])
cholesterol = st.selectbox('Cholesterol', options=['High', 'Normal'])
na_to_k = st.number_input('Na_to_K Ratio', min_value=0.0, max_value=40.0, value=15.0, step=0.1)

# Preprocessing user input
def preprocess_input(age, sex, bp, cholesterol, na_to_k):
    # Map categorical inputs to numerical values as used in training
    sex_mapped = 1 if sex == 'Male' else 0
    bp_mapped = {'High': 1, 'Normal': 2, 'Low': 3}[bp]
    cholesterol_mapped = {'High': 1, 'Normal': 2}[cholesterol]

    # Create a DataFrame for a single prediction
    input_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex_mapped,
        'BP': bp_mapped,
        'Cholesterol': cholesterol_mapped,
        'Na_to_K': na_to_k
    }])

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)
    return scaled_input

# Prediction button
if st.button('Predict Drug'):
    # Preprocess the user input
    processed_input = preprocess_input(age, sex, bp, cholesterol, na_to_k)

    # Make prediction
    prediction_numerical = model.predict(processed_input)[0]

    # Map numerical prediction back to drug name
    recommended_drug = drug_mapping.get(prediction_numerical, 'Unknown Drug')

    # Display the result
    st.success(f"The recommended drug is: **{recommended_drug}**")
