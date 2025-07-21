# DeveloperHub Task 3 – Electricity Consumption Forecasting using Machine Learning


 **Task Objective**  
This task focuses on building a machine learning model that predicts hourly electricity consumption using historical power usage data. The final product is a Streamlit-based interactive web dashboard that forecasts future energy usage based on time-related patterns.


📁 **Dataset**  
- **Name**: Household Power Consumption  
- **Source**: UCI Machine Learning Repository  



 **Features Engineered**:
- `hour` (Hour of the day)  
- `dayofweek` (Day of the week)  
- `is_weekend` (Weekend flag: 1 if Saturday or Sunday)  

 **Target**:
- `Global_active_power` (Electricity consumption in kilowatts, 1 hour ahead)



 **Tools & Libraries Used**  
- **Pandas** – Data manipulation and preprocessing  
- **Matplotlib & Seaborn** – Exploratory data analysis and visualizations  
- **XGBoost** – Time series regression modeling  
- **Scikit-learn** – Model evaluation  
- **Streamlit** – Web dashboard development  



**Approach**

### 1. Dataset Loading & Initial Exploration
- Loaded the dataset using `pandas.read_csv()`
- Combined `Date` and `Time` into a single `Datetime` column
- Set `Datetime` as the index for time series operations
- Dropped missing values and cleaned the data

### 2. Data Resampling & Preprocessing
- Resampled data to **hourly averages** for more stable modeling
- Created new time-based features:
  - `hour`: Extracted from timestamp  
  - `dayofweek`: To capture weekly seasonality  
  - `is_weekend`: To distinguish weekends vs weekdays

### 3. Target Engineering
- Defined the target as the **next hour’s electricity usage**
- Shifted `Global_active_power` by -1 to align for supervised learning

### 4. Model Training
- Split the data into:
  - **Training set**: All but the last 168 hours  
  - **Test set**: Last 168 hours (7 days)
- Trained three models:
  - ARIMA  
  - Prophet  
  - **XGBoost Regressor** ✅ *(Best-performing model)*
- Selected **XGBoost** for its superior accuracy and robustness

### 5. Evaluation Metrics
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**  
- Evaluated XGBoost predictions on the test set



 **Results & Findings**
- **XGBoost delivered the best performance** among all models
- Time-based features (`hour`, `dayofweek`, `is_weekend`) were effective in capturing energy usage patterns
- Model successfully forecasts energy use patterns across days and weekends
- Forecasts can assist users with **energy planning, cost-saving**, and **appliance scheduling**

•	 🖥️ **Interactive Streamlit Dashboard**  
- 🔗 [Live Dashboard Link](https://electricity-consumption-forecasting-dashboard-hc5f9prm2eflh9x3.streamlit.app/)

- Users can:
  - Select forecast period (24 hours, 3 days, 7 days, or custom)
  - View predicted energy usage as:
    - 📈 Line chart  
    - 📋 Data table  
- Easy to use and informative for both technical and non-technical users

 **Conclusion**
This task covered the complete machine learning pipeline:
- ✅ Time series data preprocessing  
- ✅ Feature and target engineering  
- ✅ Training and comparison of multiple forecasting models  
- ✅ Model evaluation and selection  
- ✅ Real-time dashboard deployment using Streamlit  

The final dashboard allows users to forecast electricity usage up to 7 days in advance, making it a valuable tool for **smart energy monitoring**.


🔗 **Useful Links**
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Matplotlib Docs](https://matplotlib.org/stable/)
- [Seaborn Docs](https://seaborn.pydata.org/)

---

🔖 Submitted as part of the **DeveloperHub Internship Program**

