import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/fraud_model.pkl")

# Final features used in training (after drop_first=True)
FEATURE_COLUMNS = ['amount', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# App UI
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("🔍 Real-Time Financial Fraud Detection")
st.markdown("Predict whether a transaction is fraudulent based on user inputs.")

# Input form
with st.form("fraud_form"):
    amount = st.number_input("💵 Transaction Amount", min_value=0.0, step=100.0, format="%.2f")

    # Transaction types model has seen
    transaction_type = st.selectbox("🔁 Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    submit = st.form_submit_button("Check Fraud")

if submit:
    # One-hot encode transaction type
    type_encoding = [1 if f"type_{transaction_type}" == col else 0 for col in FEATURE_COLUMNS[1:]]
    
    # Combine with amount
    features = [amount] + type_encoding
    input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.metric("Fraud Probability", f"{prob * 100:.2f}%")

    if pred == 1:
        st.error("🚨 Fraudulent transaction detected!")
    else:
        st.success("✅ Transaction appears legitimate.")

    with st.expander("🔍 View Model Input Features"):
        st.write(input_df)
