import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import pickle

model = tf.keras.models.load_model("model.h5")

with open("ohe_geo.pkl", "rb") as file:
    ohe_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    le_gender = pickle.load(file) 

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", ohe_geo.categories_[0])
gender = st.selectbox("Gender", le_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit score ")
estimated_salary = st.number_input("Estimated salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Numebr of Products", 1, 4)
has_cr_card = st.selectbox("Have credit card", [0,1])
is_active_member = st.selectbox("Is active memeber", [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = ohe_geo.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data.reset_index(drop= True), geo_encoded_df], axis = 1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

st.write(f"Churn Probability: {prediction_probab:.2f}")

if prediction_probab > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")