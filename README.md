readme = """# Customer Churn Prediction

## Live Demo
https://sejalbhupal-churn-prediction-appapp-ifhjwf.streamlit.app

## Overview
A machine learning web application that predicts whether a telecom
customer will churn (leave the service) based on their account details.
Built using Python, Scikit-learn, and Streamlit.

## Problem Statement
Telecom companies lose significant revenue when customers cancel their
service. This app helps identify high-risk customers early so the company
can take action — like offering discounts or better plans — before the
customer leaves.

## Tech Stack
- Python 3
- Pandas, NumPy — data processing
- Scikit-learn — machine learning
- Matplotlib, Seaborn — data visualization
- Streamlit — web application
- Joblib — model saving and loading

## Dataset
- Name: IBM Telco Customer Churn
- Records: 7043 customers
- Features: 21 columns including gender, tenure, contract type,
  payment method, monthly charges, and more
- Target: Churn (Yes/No)

## Project Structure
churn-project/
    app/
        app.py         -> Streamlit web application
        model.pkl      -> Trained Random Forest model
        scaler.pkl     -> Fitted StandardScaler
    data/
        WA_Fn-UseC_-Telco-Customer-Churn.csv -> Raw dataset
    notebook/
        churn_analysis.ipynb -> EDA and model training notebook
    README.md
    requirements.txt

## Machine Learning Pipeline
1. Data Cleaning
   - Converted TotalCharges to numeric
   - Removed missing values
   - Dropped customerID (not useful for prediction)

2. Feature Engineering
   - Label encoded all categorical columns
   - Standardized numerical features using StandardScaler

3. Model Training
   - Algorithm: Random Forest Classifier
   - Train/Test Split: 80/20
   - Parameters: n_estimators=100, random_state=42

4. Model Evaluation
   - Accuracy: ~80%
   - Metrics: Precision, Recall, F1-Score
   - Confusion Matrix plotted

## Features Used for Prediction
- Gender, Senior Citizen, Partner, Dependents
- Tenure (months with company)
- Phone Service, Multiple Lines
- Internet Service, Online Security, Online Backup
- Device Protection, Tech Support
- Streaming TV, Streaming Movies
- Contract Type (Month-to-month, One year, Two year)
- Paperless Billing, Payment Method
- Monthly Charges, Total Charges

## How to Run Locally
1. Clone the repository
   git clone https://github.com/sejalbhupal/churn-prediction.git

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   cd app
   streamlit run app.py

4. Open browser at
   http://localhost:8501

## How to Use the App
1. Fill in customer details in the left sidebar
2. Click the Predict Churn Risk button
3. See the result:
   - Low Risk (below 40%) -> Customer likely to stay
   - Medium Risk (40-70%) -> Customer might leave
   - High Risk (above 70%) -> Customer very likely to churn

## Results
The Random Forest model achieved approximately 80% accuracy on the
test set. Key factors affecting churn include:
- Contract type (month-to-month customers churn more)
- Tenure (newer customers churn more)
- Monthly charges (higher charges lead to more churn)
- Internet service type (fiber optic customers churn more)

## Author
Sejal Bhupal
GitHub: https://github.com/sejalbhupal
Live App: https://sejalbhupal-churn-prediction-appapp-ifhjwf.streamlit.app
"""

with open('../README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("README.md written successfully!")
