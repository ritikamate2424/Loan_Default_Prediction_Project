import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load saved models and encoders
with open('logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load Neural Network model
nn_model = load_model('nn_model.h5')

st.title("📊 Loan Default Prediction App")

# Safe encoder function
def safe_label_encode(encoder, value, column_name):
    if value not in encoder.classes_:
        st.error(f"❌ Invalid input for '{column_name}': '{value}' not seen during training.\n\nExpected options: {list(encoder.classes_)}")
        st.stop()
    return encoder.transform([value])[0]

# User Inputs
age = st.number_input("Age", 18, 100)
income = st.number_input("Income ($)", 0)
loan_amount = st.number_input("Loan Amount ($)", 0)
credit_score = st.number_input("Credit Score", 300, 900)
months_employed = st.number_input("Months Employed", 0)
num_credit_lines = st.number_input("Number of Credit Lines", 0)
interest_rate = st.number_input("Interest Rate (%)", 0.0)
loan_term = st.number_input("Loan Term (months)", 0)
dti_ratio = st.number_input("Debt-to-Income Ratio", 0.0)

# Use options from encoders to prevent invalid input
education = st.selectbox("Education", list(label_encoders['Education'].classes_))
employment_type = st.selectbox("Employment Type", list(label_encoders['EmploymentType'].classes_))
marital_status = st.selectbox("Marital Status", list(label_encoders['MaritalStatus'].classes_))
loan_purpose = st.selectbox("Loan Purpose", list(label_encoders['LoanPurpose'].classes_))

has_mortgage = st.selectbox("Has Mortgage?", ['Yes', 'No'])
has_dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
has_cosigner = st.selectbox("Has Co-Signer?", ['Yes', 'No'])

# Convert binary inputs
has_mortgage = 1 if has_mortgage == 'Yes' else 0
has_dependents = 1 if has_dependents == 'Yes' else 0
has_cosigner = 1 if has_cosigner == 'Yes' else 0

# Safely encode categorical variables
education = safe_label_encode(label_encoders['Education'], education, 'Education')
employment_type = safe_label_encode(label_encoders['EmploymentType'], employment_type, 'EmploymentType')
marital_status = safe_label_encode(label_encoders['MaritalStatus'], marital_status, 'MaritalStatus')
loan_purpose = safe_label_encode(label_encoders['LoanPurpose'], loan_purpose, 'LoanPurpose')

# Prepare feature vector
input_data = np.array([[age, income, loan_amount, credit_score, months_employed,
                        num_credit_lines, interest_rate, loan_term, dti_ratio,
                        education, employment_type, marital_status, has_mortgage,
                        has_dependents, loan_purpose, has_cosigner]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Default"):
    logistic_pred = logistic_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_scaled)[0]
    nn_pred_prob = nn_model.predict(input_scaled)[0][0]
    nn_pred = 1 if nn_pred_prob > 0.5 else 0

    st.subheader("🔍 Prediction Results:")
    st.write(f"**Logistic Regression:** {'Default' if logistic_pred == 1 else 'No Default'}")
    st.write(f"**Random Forest:** {'Default' if rf_pred == 1 else 'No Default'}")
    st.write(f"**Neural Network:** {'Default' if nn_pred == 1 else 'No Default'} (Confidence: {nn_pred_prob:.2f})")
