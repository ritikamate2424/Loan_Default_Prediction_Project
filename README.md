# Loan_Default_Prediction_Project

**1. Project Overview
**
The goal of this project is to develop a predictive model that can accurately determine whether a loan applicant is likely to default on their loan. Early prediction of loan default helps financial institutions reduce risk, optimize loan approval processes, and minimize financial losses. This project uses machine learning models — Logistic Regression, Random Forest, and Neural Networks — to classify loan applicants into "Default" or "No Default" categories based on various applicant features.

A user-friendly interface built with Streamlit allows users to input loan and personal details and receive instant predictions from the models.

2. Dataset Details (Loan_default.csv)
The dataset contains historical loan records with features relevant to the applicant's financial status and loan conditions. Key columns include:

Column Name	Description
Age	Applicant's age (years)
Income	Annual income in dollars
LoanAmount	Amount of loan requested or granted
CreditScore	Credit score of the applicant
MonthsEmployed	Total months the applicant has been employed
NumCreditLines	Number of credit lines the applicant has
InterestRate	Loan interest rate (%)
LoanTerm	Loan term in months
DTIRatio	Debt-to-Income ratio
Education	Education level (High School, Bachelor, Master, PhD)
EmploymentType	Employment status (Employed, Self-Employed, Unemployed)
MaritalStatus	Marital status (Single, Married, Divorced)
HasMortgage	Whether applicant has a mortgage (Yes/No)
HasDependents	Whether applicant has dependents (Yes/No)
LoanPurpose	Purpose of the loan (Home, Car, Business, Education)
HasCoSigner	Whether loan has a co-signer (Yes/No)
Default	Target variable indicating loan default (1 = Default, 0 = No Default)

The dataset consists of both numerical and categorical variables, requiring preprocessing before model training.

3. Data Preprocessing Steps
Handling Missing Values: Checked for and handled missing or null values, either by imputation or removal.

Encoding Categorical Variables: Used LabelEncoder to convert categorical features (Education, EmploymentType, MaritalStatus, LoanPurpose) into numerical labels compatible with machine learning models.

Converting Binary Variables: Converted Yes/No features (HasMortgage, HasDependents, HasCoSigner) into binary numeric values (1 and 0).

Feature Scaling: Applied standard scaling (StandardScaler) to numerical features to normalize their range, which is especially important for Neural Networks.

Train-Test Split: Split the data into training and testing sets to evaluate model generalization.

4. Model Training Details
Logistic Regression
A linear model used for binary classification.

Trained on scaled features to predict the probability of default.

Suitable for interpretability and baseline comparison.

Random Forest Classifier
An ensemble of decision trees to improve robustness and reduce overfitting.

Captures nonlinear relationships between features.

Used 100 trees with random_state=42 for reproducibility.

Neural Network
Constructed using TensorFlow/Keras.

Architecture:

Input layer corresponding to number of features.

Multiple dense (fully connected) layers with ReLU activations.

Dropout layers to prevent overfitting.

Output layer with sigmoid activation for binary classification.

Compiled with binary cross-entropy loss and Adam optimizer.

5. Evaluation Metrics with Code Snippets
To evaluate model performance, these metrics were computed on the test dataset:

Accuracy: Overall correctness.

Precision: Correctness of positive predictions.

Recall: Coverage of actual positives.

F1-score: Harmonic mean of precision and recall.

Confusion Matrix: Breakdown of true positives, true negatives, false positives, and false negatives.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic Regression predictions
logistic_pred = logistic_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_pred))
print("Classification Report:\n", classification_report(y_test, logistic_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_pred))

# Random Forest predictions
rf_pred = rf_model.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Neural Network evaluation
loss, accuracy = nn_model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Accuracy: {accuracy:.2f}")

# Neural Network predictions
nn_pred = (nn_model.predict(X_test_scaled) > 0.5).astype("int32")


6. Streamlit UI Code Snippet
The application allows interactive input and outputs predictions from all three models.

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load models and encoders
with open('logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
nn_model = load_model('nn_model.h5')

st.title("Loan Default Prediction")

# User inputs
age = st.number_input("Age", 18, 100)
income = st.number_input("Income ($)", 0)
# (Other inputs skipped for brevity...)

education = st.selectbox("Education", ['High School', 'Bachelor', 'Master', 'PhD'])
employment_type = st.selectbox("Employment Type", ['Employed', 'Self-Employed', 'Unemployed'])
# (Other inputs...)

# Encode and scale inputs, then predict
education_enc = label_encoders['Education'].transform([education])[0]
# (Encode other categorical features...)

input_data = np.array([[age, income, /* other features */, education_enc, /* ... */]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    pred_logistic = logistic_model.predict(input_scaled)[0]
    pred_rf = rf_model.predict(input_scaled)[0]
    pred_nn_prob = nn_model.predict(input_scaled)[0][0]
    pred_nn = 1 if pred_nn_prob > 0.5 else 0
    
   st.write(f"Logistic Regression Prediction: {'Default' if pred_logistic else 'No Default'}")
   st.write(f"Random Forest Prediction: {'Default' if pred_rf else 'No Default'}")
   st.write(f"Neural Network Prediction: {'Default' if pred_nn else 'No Default'} (Confidence: {pred_nn_prob:.2f})")

7. Performance Comparison and Explanation
Model	Accuracy (Test Set)	Notes
Logistic Regression	~X%	Good baseline; interpretable
Random Forest	~Y%	Better handling of nonlinearities
Neural Network	~Z%	Slightly higher accuracy; requires more data

The Random Forest often outperforms Logistic Regression due to capturing complex feature interactions.

Neural Networks may achieve the highest accuracy but require more tuning and data.

Confidence scores from the Neural Network provide probabilistic insight into predictions.

