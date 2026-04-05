import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(base_dir, "model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")
data_path   = os.path.join(base_dir, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def train_model():
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().copy()
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna().copy()
    df["Churn"] = df["Churn"].astype(int)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    return rf, scaler

if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
    with st.spinner("Setting up model for first time... please wait"):
        model, scaler = train_model()
    st.success("Model ready!")
    st.rerun()
else:
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.markdown("Fill in the customer details on the left and click Predict.")
st.sidebar.header("Customer Details")

gender           = st.sidebar.selectbox("Gender",            ["Female", "Male"])
senior_citizen   = st.sidebar.selectbox("Senior Citizen",    ["No", "Yes"])
partner          = st.sidebar.selectbox("Partner",           ["No", "Yes"])
dependents       = st.sidebar.selectbox("Dependents",        ["No", "Yes"])
tenure           = st.sidebar.slider("Tenure (months)",      0, 72, 12)
phone_service    = st.sidebar.selectbox("Phone Service",     ["No", "Yes"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines",    ["No phone service", "No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
online_security  = st.sidebar.selectbox("Online Security",   ["No internet service", "No", "Yes"])
online_backup    = st.sidebar.selectbox("Online Backup",     ["No internet service", "No", "Yes"])
device_prot      = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support     = st.sidebar.selectbox("Tech Support",      ["No internet service", "No", "Yes"])
streaming_tv     = st.sidebar.selectbox("Streaming TV",      ["No internet service", "No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",  ["No internet service", "No", "Yes"])
contract         = st.sidebar.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
paperless        = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method   = st.sidebar.selectbox("Payment Method",    [
    "Bank transfer (automatic)", "Credit card (automatic)",
    "Electronic check", "Mailed check"
])
monthly_charges  = st.sidebar.number_input("Monthly Charges ($)",  0.0, 200.0, 65.0)
total_charges    = st.sidebar.number_input("Total Charges ($)",    0.0, 10000.0, 1000.0)

def preprocess_input():
    raw = {
        "gender":           [gender],
        "SeniorCitizen":    [1 if senior_citizen == "Yes" else 0],
        "Partner":          [partner],
        "Dependents":       [dependents],
        "tenure":           [int(tenure)],
        "PhoneService":     [phone_service],
        "MultipleLines":    [multiple_lines],
        "InternetService":  [internet_service],
        "OnlineSecurity":   [online_security],
        "OnlineBackup":     [online_backup],
        "DeviceProtection": [device_prot],
        "TechSupport":      [tech_support],
        "StreamingTV":      [streaming_tv],
        "StreamingMovies":  [streaming_movies],
        "Contract":         [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod":    [payment_method],
        "MonthlyCharges":   [float(monthly_charges)],
        "TotalCharges":     [float(total_charges)],
    }
    df = pd.DataFrame(raw)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])
    df = df.astype(float)
    return scaler.transform(df)

if st.button("Predict Churn Risk"):
    X_input = preprocess_input()
    prob    = model.predict_proba(X_input)[0][1]
    st.subheader("Result")
    st.metric("Churn Probability", f"{prob * 100:.1f}%")
    if prob > 0.7:
        st.error("High Risk - This customer is very likely to churn.")
    elif prob > 0.4:
        st.warning("Medium Risk - This customer may churn.")
    else:
        st.success("Low Risk - This customer is unlikely to churn.")
