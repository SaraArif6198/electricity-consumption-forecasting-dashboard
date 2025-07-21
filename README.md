#  DeveloperHub Task 3 – Electricity Consumption Forecasting using Machine Learning

##  Task Objective  
This task focuses on building a machine learning model that predicts **hourly electricity consumption** using historical power usage data. The final product is a **Streamlit-based interactive dashboard** that helps users forecast energy usage based on simple inputs like time and day.


## 📁 Dataset  
- **Name**: Household Power Consumption  
- **Source**: UCI Machine Learning Repository  


##  Features Engineered
- `hour` – Hour of the day (0–23)  
- `dayofweek` – Day of the week (0=Monday, 6=Sunday)  
- `is_weekend` – Boolean flag (1 for Saturday/Sunday)

### Target:
- `Global_active_power` – Electricity consumption (in kW) for the **next hour**

## 🛠️ Tools & Libraries Used
- `pandas` – Data manipulation and preprocessing  
- `matplotlib`, `seaborn` – Data exploration and visualization  
- `xgboost` – Time series regression modeling  
- `scikit-learn` – Evaluation metrics  
- `streamlit` – Interactive dashboard development


##  Approach

### 1. Data Loading & Cleaning
- Combined `Date` and `Time` into a single datetime index  
- Handled missing values  
- Filtered and structured the dataset for hourly modeling

### 2. Feature Engineering
- Resampled the data to **hourly averages**  
- Created three time-based features:
  - `hour`  
  - `dayofweek`  
  - `is_weekend`

### 3. Target Definition
- Defined the prediction target as the **next hour’s energy usage** by shifting values back 1 hour

### 4. Model Training
- Trained three models:  
  - ARIMA  
  - Prophet  
  - ✅ XGBoost (Best performer)
- Chose **XGBoost Regressor** due to superior accuracy, speed, and flexibility

### 5. Evaluation Metrics
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**

##  Results & Findings
- XGBoost showed **strong predictive power** with time-based features
- Captured consumption patterns across:
  - Days of the week  
  - Weekends vs weekdays  
  - Hourly trends
- Allows users to plan energy usage, reduce costs, and schedule appliances intelligently


## 🖥️ Interactive Streamlit Dashboard  
🔗 [Live Dashboard Link](https://electricity-consumption-forecasting-dashboard.streamlit.app/)

### 👥 User Features:
- Select from **predefined forecast durations**:
  - Next 24 hours  
  - Next 3 days  
  - Next 7 days  
  - Custom (up to 168 hours)
- Choose how to provide data:
  - **Use default historical data**
  - **Upload your own data**
    - Upload a CSV file with proper format (semicolon-separated)
    - OR enter a single row manually using a form
- See results as:
  - 📈 Interactive line chart  
  - 📋 Data table

###  Model Explanation (for non-technical users)
> The model predicts energy usage based on:
> - **What hour** it is (e.g., 2 PM)
> - **What day** it is (e.g., Monday)
> - **Is it a weekend?** (Yes or No)
>
> Using patterns from past data, it predicts things like:
> “On Mondays at 2 PM, people usually consume 0.73 kW — so that’s my prediction.”

📌 **Note**: All explanations and error messages are written in plain, human-friendly language.


## ✅ Conclusion
This task demonstrated the complete ML pipeline:
-  Time series data processing  
-  Feature engineering  
- Model training & evaluation  
-  Deployment via Streamlit

✅ The dashboard helps users forecast electricity usage easily — up to **7 days in advance** — making it a powerful tool for **smart home energy planning**.


## 🔗 Useful Links
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Matplotlib Docs](https://matplotlib.org/)
- [Seaborn Docs](https://seaborn.pydata.org/)

---

📨 **Submitted as part of the DeveloperHub Internship Program**
