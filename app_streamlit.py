
import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "https://price-optimization-n20m.onrender.com/predict/"

st.set_page_config(page_title="Price Optimization Predictor", page_icon="💸", layout="wide")
st.title("💸 Price Optimization - Predict Customer Continuation after Price Increase")
st.write("Business Scenario: Predict which customers will continue buying after a price increase (Booking platforms: Expedia, Kayak).")

menu = st.sidebar.radio("Select mode", ["1️⃣ Single Prediction", "2️⃣ Batch Prediction (Excel)", "3️⃣ Visual Analytics"])

# ----------- 1️⃣ Single-customer prediction -----------
if menu == "1️⃣ Single Prediction":
    st.header("Single-customer Prediction")

    total_spent = st.number_input("Total Spent", value=500.0)
    avg_order_value = st.number_input("Avg Order Value", value=100.0)
    avg_purchase_frequency = st.number_input("Avg Purchase Frequency", value=3.0)
    days_since_last_purchase = st.number_input("Days Since Last Purchase", value=30)
    discount_behavior = st.slider("Discount Behavior", min_value=0.0, max_value=1.0, value=0.5)
    loyalty_program_member = st.selectbox("Loyalty Program Member", options=[0, 1])
    days_in_advance = st.number_input("Days in Advance", value=14)
    flight_type = st.selectbox("Flight Type", options=["domestic", "international"])
    cabin_class = st.selectbox("Cabin Class", options=["economy", "business"])

    if st.button("Predict"):
        input_data = {
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "avg_purchase_frequency": avg_purchase_frequency,
            "days_since_last_purchase": days_since_last_purchase,
            "discount_behavior": discount_behavior,
            "loyalty_program_member": loyalty_program_member,
            "days_in_advance": days_in_advance,
            "flight_type": flight_type,
            "cabin_class": cabin_class
        }

        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'✅ Will Buy' if result['will_buy_after_price_increase'] else '❌ Will Not Buy'}")
            st.info(f"Probability: {result['probability']:.2%}")
        else:
            st.error(f"API Error: {response.status_code}")

# ----------- 2️⃣ Batch Prediction -----------
elif menu == "2️⃣ Batch Prediction (Excel)":
    st.header("Batch Prediction - Upload Excel file")

    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        df_input = pd.read_excel(uploaded_file)
        st.write("📄 Uploaded Data Preview:", df_input.head())

        if st.button("Run Batch Prediction"):
            predictions = []
            probabilities = []

            for _, row in df_input.iterrows():
                input_data = {
                    "total_spent": row["total_spent"],
                    "avg_order_value": row["avg_order_value"],
                    "avg_purchase_frequency": row["avg_purchase_frequency"],
                    "days_since_last_purchase": row["days_since_last_purchase"],
                    "discount_behavior": row["discount_behavior"],
                    "loyalty_program_member": int(row["loyalty_program_member"]),
                    "days_in_advance": int(row["days_in_advance"]),
                    "flight_type": row["flight_type"],
                    "cabin_class": row["cabin_class"]
                }

                response = requests.post(API_URL, json=input_data)
                if response.status_code == 200:
                    result = response.json()
                    predictions.append(result["will_buy_after_price_increase"])
                    probabilities.append(result["probability"])
                else:
                    predictions.append(None)
                    probabilities.append(None)

            df_input["Prediction"] = predictions
            df_input["Probability"] = probabilities

            st.write("✅ Predictions Complete. Results preview:")
            st.write(df_input.head())

            st.download_button("Download Results as Excel",
                               data=df_input.to_excel(index=False),
                               file_name="predictions.xlsx")

# ----------- 3️⃣ Visual Analytics -----------
elif menu == "3️⃣ Visual Analytics":
    st.header("Visual Analytics - Batch Results")

    uploaded_file = st.file_uploader("Upload Predictions Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        df_results = pd.read_excel(uploaded_file)
        st.write("📄 Data Preview:", df_results.head())

        if "Prediction" in df_results.columns and "Probability" in df_results.columns:
            st.subheader("📊 Aggregate Stats")
            pct_continue = df_results["Prediction"].mean() * 100
            st.metric("Predicted to Continue Buying", f"{pct_continue:.2f}%")

            st.subheader("Probability Distribution")
            plt.figure(figsize=(8, 4))
            sns.histplot(df_results["Probability"].dropna(), bins=20, kde=True)
            plt.xlabel("Probability of Continuing")
            plt.ylabel("Frequency")
            st.pyplot(plt.gcf())

            st.subheader("Prediction Count")
            st.bar_chart(df_results["Prediction"].value_counts())
        else:
            st.error("File must contain 'Prediction' and 'Probability' columns.")
