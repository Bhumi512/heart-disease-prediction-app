#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Heart Disease Prediction")

# Collect user input
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', options=[0, 1])
chest_pain = st.selectbox('Chest Pain Type', options=[1, 2, 3, 4])
bp = st.number_input('Blood Pressure', min_value=90, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
ekg = st.selectbox('Electrocardiographic results', options=[0, 1, 2])
max_hr = st.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise Induced Angina', options=[0, 1])
st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=0.0)
slope_st = st.selectbox('Slope of the peak exercise ST segment', options=[1, 2, 3])
num_vessels = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
thallium = st.selectbox('Thallium Stress Test Result', options=[3, 6, 7])

# Prediction button
if st.button('Predict Heart Disease'):
    # Prepare the input data
    input_data = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg, max_hr, exercise_angina, st_depression, slope_st, num_vessels, thallium]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction == 1:
        st.write("The model predicts the presence of heart disease.")
    else:
        st.write("The model predicts the absence of heart disease.")


# In[ ]:




