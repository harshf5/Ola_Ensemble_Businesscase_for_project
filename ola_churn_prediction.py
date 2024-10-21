import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.header('Ola driver Prediction App')


#inputs from user

col1, col2= st.columns(2)

with col1:
    year = st.selectbox(
        'Enter the year of joining: ',
        [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013])

with col2:
    Gender_ip= st.selectbox(
        'Enter the Fuel Type: ',
        ('Male', 'Female'))

col1, col2= st.columns(2)
with col1:
    Education_Level= st.selectbox(
        'Enter the Education Level : ',
        (0 ,1, 2))
with col2:
    Age= st.slider('Age ', 18, 65, 21)

col1, col2= st.columns(2)
# Taking income input from the user
with col1:
    income = st.slider("Enter your income:", 10000,200000,15000)
with col2:
    total_business=st.slider("Enter the business value generated:", 0,10000000,50000)


# Apply log transformation, but handle the case where income is zero
if income > 0:
    Income_log = np.log(income)
    # st.write(f"Log transformed income: {log_income}")
# else:
#     st.write("Income must be greater than zero for log transformation.")

if income > 0:
    Total_Business_Value_log1 = np.log(total_business)
    # st.write(f"Log transformed income: {Total_Business_Value_log1}")
# else:
#     st.write("Income must be greater than zero for log transformation.")

encode_dict = {
    "Gender": {'Male': 1, 'Female': 0}
}

#relation building is left
def model_pred(Age,Gender_encoded,Education_Level,year,Income_log,Total_Business_Value_log1):
    with open("classifier.pkl", 'rb') as file:
        model = pickle.load(file)
        input_features=[[24,Age, Gender_encoded,	0.411474,Education_Level,3,1,2,8,year,	0,	0,	Income_log,	Total_Business_Value_log1]]
        return model.predict(input_features)[0]



if st.button('Predict'):
    Gender_encoded= encode_dict['Gender'][Gender_ip]
    Churn_predict=model_pred(Age,Gender_encoded,Education_Level,year,Income_log,Total_Business_Value_log1)
    # Define a dictionary to map the binary values to more meaningful labels
    output_mapping = {0: "Will Stay", 1: "Will Leave"}

    # Retrieve the prediction output (assuming the model returns 0 or 1)
    encoded_output = output_mapping.get(Churn_predict, "Unknown")
    st.write(f"The driver is predicted to: {encoded_output}")