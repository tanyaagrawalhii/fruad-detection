import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# --- CONFIG ---
MODEL_PATH = "model/fraud_model.pkl"
DRIVE_FILE_ID = "1U6XZZW6ULE3Xv0lterQNiCaHOGMAkOuO"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# --- Ensure model exists ---
os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.warning("📥 Downloading model from Google Drive...")
    try:
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# --- Load trained model ---
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Final features used in training ---
FEATURE_COLUMNS = ['amount', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# --- Streamlit UI ---
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("🔍 Real-Time Financial Fraud Detection")
st.markdown("Use the form below to check if a transaction is potentially fraudulent.")

# --- Input form ---
with st.form("fraud_form"):
    amount = st.number_input("💵 Transaction Amount", min_value=0.0, step=100.0, format="%.2f")
    transaction_type = st.selectbox("🔁 Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    submit = st.form_submit_button("Check Fraud")

# --- Prediction ---
if submit:
    # One-hot encode transaction type
    type_encoding = [1 if f"type_{transaction_type}" == col else 0 for col in FEATURE_COLUMNS[1:]]
    features = [amount] + type_encoding
    input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Predict
    try:
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.metric("📊 Fraud Probability", f"{prob * 100:.2f}%")
        if pred == 1:
            st.error("🚨 Fraudulent transaction detected!")
        else:
            st.success("✅ Transaction appears legitimate.")
        
        with st.expander("🔍 View Model Input Features"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
