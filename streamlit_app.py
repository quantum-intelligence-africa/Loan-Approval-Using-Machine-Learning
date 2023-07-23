import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# LabelEncoder for preprocessing
label_encoder = LabelEncoder()

# Function to preprocess the input data
def preprocess_input(data):
    data['Gender'] = label_encoder.transform([data['Gender']])
    data['Married'] = label_encoder.transform([data['Married']])
    data['Education'] = label_encoder.transform([data['Education']])
    data['Self_Employed'] = label_encoder.transform([data['Self_Employed']])
    data['Property_Area'] = label_encoder.transform([data['Property_Area']])
    return data

# Main Streamlit app
def main():
    st.title('Loan Approval Prediction')

    # User input form
    st.header('Enter Applicant Details')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.slider('Dependents', 0, 10, 0)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income', min_value=0, step=100)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0, step=100)
    loan_amount = st.number_input('Loan Amount', min_value=0, step=100)
    loan_amount_term = st.slider('Loan Amount Term', 1, 480, 360)
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

    # Create a dictionary from user inputs
    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Preprocess the user input data
    user_data_df = pd.DataFrame([user_data])
    preprocessed_data = preprocess_input(user_data_df)

    if st.button('Predict'):
        # Make the prediction using the loaded model
        predicted_approval_status = model.predict(preprocessed_data)[0]
        st.subheader('Prediction:')
        if predicted_approval_status == 1:
            st.write('Loan Approved')
        else:
            st.write('Loan Not Approved')

if __name__ == '__main__':
    main()
