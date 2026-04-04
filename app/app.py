import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

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
        "tenure":           [tenure],
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
        "MonthlyCharges":   [monthly_charges],
        "TotalCharges":     [total_charges],
    }
    df = pd.DataFrame(raw)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])
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

    # Churn probability bar chart
    st.subheader("Probability Chart")
    fig, ax = plt.subplots()
    ax.barh(["Stay", "Churn"], [1 - prob, prob], color=["green", "red"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

    # Customer summary
    st.subheader("Customer Summary")
    summary = {
        "Gender": gender, "Senior Citizen": senior_citizen,
        "Partner": partner, "Dependents": dependents,
        "Tenure (months)": tenure, "Contract": contract,
        "Internet Service": internet_service,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }
    st.table(pd.DataFrame(summary, index=["Value"]).T)
