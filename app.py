import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import google.generativeai as genai
from dotenv import load_dotenv
import os
import random
load_dotenv()

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Load trained model and preprocessing objects
def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

rf_model = load_pickle('rf_model.pkl')
scaler = load_pickle('scaler.pkl')
label_encoders = load_pickle('label_encoders.pkl')
categorical_features = ['Region', 'Customer Type', 'Payment Method', 'Product Line']
numerical_features = ['Quantity', 'Unit Price', 'Total Sales']

df = pd.read_excel('dataset.xlsx', sheet_name='Sheet1')

st.title("Profitability Prediction App")
st.write("Enter the details below to predict profitability.")

# User input
quantity = st.number_input("Quantity", min_value=1, max_value=int(df['Quantity'].max()), value=1)
unit_price = st.number_input("Unit Price", min_value=float(df['Unit Price'].min()), max_value=float(df['Unit Price'].max()), value=float(df['Unit Price'].min()))
total_sales = st.number_input("Total Sales", min_value=float(df['Total Sales'].min()), max_value=float(df['Total Sales'].max()), value=float(unit_price*quantity))

region = st.selectbox("Region", df['Region'].unique())
customer_type = st.selectbox("Customer Type", df['Customer Type'].unique())
payment_method = st.selectbox("Payment Method", df['Payment Method'].unique())
product_line = st.selectbox("Product Line", df['Product Line'].unique())

# Create input DataFrame
input_data = pd.DataFrame([[quantity, unit_price, total_sales, region, customer_type, payment_method, product_line]],
                           columns=numerical_features + categorical_features)
temp_data=input_data

# Label encoding for categorical variables
for col in categorical_features:
    input_data[col] = label_encoders[col].transform([input_data[col][0]])

# Normalize numerical features
input_numerical = scaler.transform(input_data[numerical_features])
input_numerical_df = pd.DataFrame(input_numerical, columns=numerical_features)

# Merge features
input_final = pd.concat([input_numerical_df, input_data[categorical_features]], axis=1)

# Predict
if st.button("Predict Profitability"):
    prediction = rf_model.predict(input_final)[0]
    result = "Profitable" if prediction == 1 else "Unprofitable"
    st.success(f"The predicted profitability status is: {result}")
    # insights_prompt = f"Based on the following inputs:\n{temp_data}\nAnd the predicted result : {result} , provide actionable insights and recommendations to improve the business profit."
    insights_prompt = f"Based on the following inputs:\n{temp_data}\n(Data Ranges: Quantity [{df['Quantity'].min()}-{df['Quantity'].max()}], Unit Price [{df['Unit Price'].min()}-{df['Unit Price'].max()}], Total Sales [{df['Total Sales'].min()}-{df['Total Sales'].max()}])\nAnd the predicted result: {result}, provide actionable insights and recommendations for business."

    # Request insights from Gemini
    response = model.generate_content(insights_prompt)
    
    # Display the insights and recommendations
    st.write("Insights & Recommendations:")
    st.write(response.text)
