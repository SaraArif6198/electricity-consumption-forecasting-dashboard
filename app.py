# app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

# Load trained XGBoost model
model = xgb.XGBRegressor()
model = joblib.load("model/xgboost_model.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Energy Forecast", layout="centered")
st.title("Energy Consumption Forecasting")
with st.expander("📘 What does this model actually do?"):
    st.markdown(""" 
    This tool uses a smart machine learning model to **predict future electricity usage** based on time-related patterns.

    Here’s how it works in simple terms:

    - 🕒 **Hour of the day** – Energy usage is usually higher during the morning or evening when people are active.
    - 📅 **Day of the week** – Weekdays often have different consumption patterns than weekends.
    -  **Weekend or not** – People may use appliances differently on weekends.

    The model has learned these patterns from past electricity data and uses them to **forecast upcoming usage** (even if we don’t provide previous power readings).

    So, when you pick a time or upload your data, it analyzes **when** the energy is needed — and predicts **how much power will likely be used** based on past behavior.
    """)


st.markdown("""

Welcome! 👋  
This tool helps you **predict future energy usage** for the next few hours based on past patterns in household electricity consumption.

We use a smart prediction model (called **XGBoost**, a type of machine learning) trained on hourly data that considers:
- **What hour is it?** (e.g., 2 PM)
- **What day of the week is it?** (e.g., Monday)
- **Is it a weekend?** (e.g., No)

Then, using what it has learned from past data, it predicts:

> 💡 “Usually on **Mondays at 2 PM**, homes consume around **0.73 kW** — so that’s my forecast!”
""")

# ── Data Source Selection ──
data_option = st.radio("Choose Data Source", ["Use default data", "Upload your own data"],
                       help="You can either:\n\n- Select a forecast horizon (24h, 3d, 7d, or custom) and forecast based on default pattern data\n- Or upload your own power consumption CSV file and forecast based on your historical data")

if data_option == "Upload your own data":
    method = st.radio("Choose input method", ["Single Record (Form)", "Multiple Records (Upload CSV)"])

    if method == "Single Record (Form)":
        st.subheader("Enter a single data record")
        date_input = st.date_input("Date")
        time_input = st.time_input("Time")
        power_input = st.number_input("Global Active Power (kW)", min_value=0.0, format="%.3f")

        if st.button("Predict Consumption"):
            try:
                datetime_obj = datetime.combine(date_input, time_input)
                hour = datetime_obj.hour
                dayofweek = datetime_obj.weekday()
                is_weekend = 1 if dayofweek >= 5 else 0

                features = pd.DataFrame([[hour, dayofweek, is_weekend]],
                                        columns=['hour', 'dayofweek', 'is_weekend'])
                prediction = model.predict(features)[0]

                # Create a result DataFrame
                result_df = pd.DataFrame({
                    "Datetime": [datetime_obj],
                    "Predicted Power (kW)": [round(prediction, 3)]
                })

                st.subheader("Prediction Result")
                st.dataframe(result_df)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    else:
        st.info("Please upload a CSV file with columns: 'Date', 'Time', and 'Global_active_power'.\nExample: Date format '16/12/2006', Time format '17:24:00', Power in kW.")

        sample_data = """Date;Time;Global_active_power
16/12/2006;17:24:00;4.216
16/12/2006;17:25:00;5.360
16/12/2006;17:26:00;5.374
"""
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_data,
            file_name="sample_format.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file, sep=';', low_memory=False)
                required_cols = {'Date', 'Time', 'Global_active_power'}
                if not required_cols.issubset(user_df.columns):
                    st.error(f"❌ Missing columns: {required_cols - set(user_df.columns)}")
                else:
                    user_df['Datetime'] = pd.to_datetime(user_df['Date'] + ' ' + user_df['Time'], errors='coerce')
                    user_df = user_df.dropna(subset=['Datetime'])
                    user_df.set_index('Datetime', inplace=True)
                    user_df['Global_active_power'] = pd.to_numeric(user_df['Global_active_power'], errors='coerce')
                    user_df = user_df[['Global_active_power']].dropna()

                    if user_df.empty:
                        st.warning("⚠️ Uploaded file has no valid data after cleaning.")
                    else:
                        # Resample and feature engineering
                        df_hourly = user_df.resample('H').mean()
                        df_hourly['hour'] = df_hourly.index.hour
                        df_hourly['dayofweek'] = df_hourly.index.dayofweek
                        df_hourly['is_weekend'] = df_hourly['dayofweek'].isin([5, 6]).astype(int)

                        # Forecast horizon
                        forecast_horizon = st.slider("Forecast horizon (hours)", 1, 168, 24)

                        # Generate future timestamps
                        last_time = df_hourly.index[-1]
                        future_times = [last_time + timedelta(hours=i+1) for i in range(forecast_horizon)]

                        forecast_input = pd.DataFrame({
                            'hour': [t.hour for t in future_times],
                            'dayofweek': [t.weekday() for t in future_times],
                            'is_weekend': [1 if t.weekday() >= 5 else 0 for t in future_times],
                            'timestamp': future_times
                        })

                        forecast_input['Predicted Power (kW)'] = model.predict(
                            forecast_input[['hour', 'dayofweek', 'is_weekend']]
                        )

                        # Display chart and table
                        st.subheader("Forecast Plot")
                        st.line_chart(forecast_input.set_index("timestamp")["Predicted Power (kW)"])

                        st.subheader("Forecast Data")
                        with st.expander("ℹ️ What do the columns mean?", expanded=False):
                            st.markdown("""
                            - **hour**: The hour of the day (0–23). For example, 2 PM is 14.  
                            - **dayofweek**: Day of the week (0 = Monday, ..., 6 = Sunday).  
                            - **is_weekend**: A flag showing if it’s a weekend (1 = Yes, 0 = No).  
                            - **Predicted Power (kW)**: The model’s estimated electricity use for that time.
                            """)

                        st.dataframe(forecast_input.rename(columns={"timestamp": "Datetime"}))

            except Exception as e:
                st.error(f"⚠️ An error occurred while processing the file: {e}")

else:
    # ── Forecast from default pattern data ──
    option = st.selectbox(
        "Choose forecast period",
        ("Next 24 hours", "Next 3 days", "Next 7 days", "Custom")
    )

    preset_map = {
        "Next 24 hours": 24,
        "Next 3 days": 3 * 24,
        "Next 7 days": 7 * 24
    }

    if option != "Custom":
        n_hours = preset_map[option]
    else:
        n_hours = st.slider("Custom forecast horizon (hours)", min_value=1, max_value=168, value=24)

    # Generate future timestamps from now
    start_time = datetime.now()
    future_times = [start_time + timedelta(hours=i) for i in range(1, n_hours + 1)]

    # Feature Engineering
    future_df = pd.DataFrame({
        "hour": [t.hour for t in future_times],
        "dayofweek": [t.weekday() for t in future_times],
        "is_weekend": [1 if t.weekday() >= 5 else 0 for t in future_times],
        "timestamp": future_times
    })

    # Predict
    X_future = future_df[["hour", "dayofweek", "is_weekend"]]
    future_df["Predicted Power (kW)"] = model.predict(X_future)

    # Display results
    st.subheader("Forecast Plot")
    st.line_chart(future_df.set_index("timestamp")["Predicted Power (kW)"])

    st.subheader("Forecast Data")
    st.dataframe(future_df[["timestamp", "Predicted Power (kW)"]].rename(columns={"timestamp": "Datetime"}))

