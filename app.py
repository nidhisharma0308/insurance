import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model = pickle.load(open('insurance.pkl', 'rb'))

# Title of the app
st.title("Insurance Premium Predictor")

# Input fields for user to provide data
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ("yes", "no"))
region = st.selectbox("Region", ("southwest", "southeast", "northwest", "northeast"))

# Convert non-numerical inputs to numerical values
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[region]

# Create a DataFrame for the input
input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                          columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

# Display input data
st.write("Input Data:", input_data)

# Make a prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"The estimated insurance premium is RS{prediction:.2f}")

# Run the Streamlit app using: streamlit run app.py
