from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# LabelEncoder for preprocessing
label_encoder = LabelEncoder()

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and display prediction result
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the request
    gender = request.form['gender']
    married = request.form['married']
    dependents = int(request.form['dependents'])
    education = request.form['education']
    self_employed = request.form['self_employed']
    applicant_income = int(request.form['applicant_income'])
    coapplicant_income = int(request.form['coapplicant_income'])
    loan_amount = int(request.form['loan_amount'])
    loan_amount_term = int(request.form['loan_amount_term'])
    credit_history = int(request.form['credit_history'])
    property_area = request.form['property_area']

    # Convert the form data to a DataFrame
    data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    }
    dummy_data_df = pd.DataFrame(data)

    # Label encoding for categorical features
    dummy_data_df['Gender'] = label_encoder.fit_transform(dummy_data_df['Gender'])
    dummy_data_df['Married'] = label_encoder.fit_transform(dummy_data_df['Married'])
    dummy_data_df['Education'] = label_encoder.fit_transform(dummy_data_df['Education'])
    dummy_data_df['Self_Employed'] = label_encoder.fit_transform(dummy_data_df['Self_Employed'])
    dummy_data_df['Property_Area'] = label_encoder.fit_transform(dummy_data_df['Property_Area'])

    # Make the prediction using the loaded model
    predicted_approval_status = model.predict(dummy_data_df)[0]

    return render_template('result.html', prediction=predicted_approval_status)
    

if __name__ == '__main__':
    app.run(debug=True)
