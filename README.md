readme = """# Customer Churn Prediction

## Overview
A machine learning web app that predicts whether a telecom customer
will churn (leave the service) based on their account details.

## Problem Statement
Telecom companies lose revenue when customers leave.
This app helps identify high-risk customers so the company
can take action before they churn.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Random Forest)
- Streamlit (Web App)
- Matplotlib, Seaborn

## Model Performance
- Algorithm: Random Forest Classifier
- Features used: 19 customer attributes
- Dataset: IBM Telco Customer Churn (7043 records)

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   cd app
   streamlit run app.py

3. Open browser at:
   http://localhost:8501

## Project Structure
churn-project/
    app/
        app.py         -> Streamlit web app
        model.pkl      -> Trained Random Forest model
        scaler.pkl     -> Fitted StandardScaler
    data/
        WA_Fn-UseC_-Telco-Customer-Churn.csv
    notebook/
        churn_analysis.ipynb -> EDA and model training
    README.md
    requirements.txt

## Features Used
- Gender, Senior Citizen, Partner, Dependents
- Tenure, Phone Service, Internet Service
- Online Security, Tech Support, Streaming
- Contract type, Payment Method
- Monthly Charges, Total Charges

## Result
- Low Risk   -> Customer likely to stay
- Medium Risk -> Customer might leave
- High Risk  -> Customer very likely to churn

## Author
Sejal
"""

with open('../README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("README.md written!")