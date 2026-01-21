import joblib
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Energy Consumption Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS for Professional Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-color: #0e1117;
        --card-background: #1e1e1e;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #333;
    }
    
    /* Author card */
    .author-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/xgboost_model.pkl")

model = load_model()

# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
st.sidebar.markdown("# ⚡ Energy Forecast Pro")
st.sidebar.markdown("---")

choice = st.sidebar.radio(
    "🧭 **Navigation**",
    ["🏠 Energy Forecast", "👤 About the Author"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.info("**Model Type:** XGBoost Regressor")
st.sidebar.success("**Accuracy:** Optimized for time-series")
st.sidebar.warning("**Features:** 3 (Time-based)")

# =====================================================
# 🔹 MAIN APP: ENERGY FORECAST
# =====================================================
if choice == "🏠 Energy Forecast":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Energy Consumption Forecasting</h1>
        <p>Predict household electricity usage using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🤖 Model</div>
            <div class="metric-value">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📈 Prediction Type</div>
            <div class="metric-value">Time-Series</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">⚙️ Features</div>
            <div class="metric-value">3 Variables</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it Works Section
    with st.expander("📘 **How Does This Model Work?**", expanded=False):
        st.markdown("""
        <div style='padding: 1rem;'>
        
        This application leverages a **machine learning model (XGBoost)** to predict **future electricity consumption**  
        based on **temporal patterns** learned from historical data.

        <div class="feature-card">
        <b>🎯 Key Input Features:</b><br>
        • 🕒 <b>Hour of the Day</b> - Captures daily consumption patterns<br>
        • 📅 <b>Day of the Week</b> - Identifies weekly trends<br>
        • 📌 <b>Weekend vs Weekday</b> - Distinguishes usage behavior
        </div>

        The model analyzes historical usage patterns to predict **how much power is likely to be consumed**  
        at any given time — even without direct power readings.
        
        </div>
        """, unsafe_allow_html=True)

    # Example Usage
    st.markdown("""
    <div class="info-box">
        💡 <b>Example Prediction:</b><br>
        <i>"Based on historical patterns, on Mondays at 2 PM, households typically consume around 0.73 kW."</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # Data Source Selection
    # ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>📂 Select Data Source</div>", unsafe_allow_html=True)
    
    data_option = st.radio(
        "**Choose your forecasting method:**",
        ["🎯 Use Default Data (Quick Start)", "📤 Upload Your Own Data"],
        horizontal=True
    )

    st.markdown("---")

    # =====================================================
    # 🔹 USER UPLOADED DATA
    # =====================================================
    if data_option == "📤 Upload Your Own Data":

        method = st.radio(
            "**Select input method:**",
            ["📝 Single Record (Manual Entry)", "📊 Multiple Records (CSV Upload)"],
            horizontal=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Single Record ─────────────────────────
        if method == "📝 Single Record (Manual Entry)":
            st.markdown("<div class='section-header'>📝 Single Record Prediction</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                date_input = st.date_input("📅 Select Date", datetime.now())
            with col2:
                time_input = st.time_input("🕐 Select Time", datetime.now().time())
            with col3:
                power_input = st.number_input(
                    "⚡ Active Power (kW)",
                    min_value=0.0,
                    value=0.0,
                    format="%.3f",
                    help="Historical power reading (optional)"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                predict_btn = st.button("🔮 **Generate Prediction**", use_container_width=True)

            if predict_btn:
                datetime_obj = datetime.combine(date_input, time_input)
                hour = datetime_obj.hour
                dayofweek = datetime_obj.weekday()
                is_weekend = 1 if dayofweek >= 5 else 0

                features = pd.DataFrame(
                    [[hour, dayofweek, is_weekend]],
                    columns=["hour", "dayofweek", "is_weekend"]
                )

                prediction = model.predict(features)[0]

                st.markdown("""
                <div class="success-box">
                    <h3 style='margin:0;'>✅ Prediction Complete!</h3>
                </div>
                """, unsafe_allow_html=True)

                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📅 Date & Time", datetime_obj.strftime("%Y-%m-%d %H:%M"))
                with col2:
                    st.metric("⚡ Predicted Power", f"{round(prediction, 3)} kW")
                with col3:
                    day_name = datetime_obj.strftime("%A")
                    period = "Weekend" if is_weekend else "Weekday"
                    st.metric("📊 Period Type", f"{day_name} ({period})")

                # Detailed results table
                st.markdown("<br>", unsafe_allow_html=True)
                result_df = pd.DataFrame({
                    "DateTime": [datetime_obj.strftime("%Y-%m-%d %H:%M:%S")],
                    "Hour": [hour],
                    "Day of Week": [day_name],
                    "Weekend": ["Yes" if is_weekend else "No"],
                    "Predicted Power (kW)": [round(prediction, 3)]
                })
                
                st.dataframe(result_df, use_container_width=True, hide_index=True)

        # ── CSV Upload ────────────────────────────
        else:
            st.markdown("<div class='section-header'>📊 Upload CSV File</div>", unsafe_allow_html=True)
            
            st.info("📋 **Required columns:** `Date`, `Time`, `Global_active_power`")

            uploaded_file = st.file_uploader(
                "**Choose your CSV file**",
                type=["csv"],
                help="Upload a CSV file with energy consumption data"
            )

            if uploaded_file is not None:
                with st.spinner("⏳ Processing your data..."):
                    user_df = pd.read_csv(uploaded_file, sep=';', low_memory=False)

                    user_df['Datetime'] = pd.to_datetime(
                        user_df['Date'] + ' ' + user_df['Time'],
                        errors='coerce'
                    )
                    user_df.dropna(subset=['Datetime'], inplace=True)
                    user_df.set_index('Datetime', inplace=True)
                    user_df['Global_active_power'] = pd.to_numeric(
                        user_df['Global_active_power'],
                        errors='coerce'
                    )
                    user_df.dropna(inplace=True)

                    df_hourly = user_df.resample('H').mean()
                    df_hourly['hour'] = df_hourly.index.hour
                    df_hourly['dayofweek'] = df_hourly.index.dayofweek
                    df_hourly['is_weekend'] = df_hourly['dayofweek'].isin([5, 6]).astype(int)

                st.success(f"✅ Successfully processed {len(df_hourly)} hourly records!")

                # Data preview
                with st.expander("👁️ **Preview Uploaded Data**"):
                    st.dataframe(df_hourly.head(10), use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Forecast settings
                st.markdown("<div class='section-header'>⚙️ Forecast Settings</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    horizon = st.slider(
                        "**Select forecast horizon (hours)**",
                        min_value=1,
                        max_value=168,
                        value=24,
                        help="Choose how many hours ahead to forecast"
                    )
                with col2:
                    st.metric("Forecast Period", f"{horizon} hrs")

                if st.button("🚀 **Generate Forecast**", use_container_width=True):
                    with st.spinner("🔮 Generating predictions..."):
                        future_times = [
                            df_hourly.index[-1] + timedelta(hours=i + 1)
                            for i in range(horizon)
                        ]

                        future_df = pd.DataFrame({
                            "hour": [t.hour for t in future_times],
                            "dayofweek": [t.weekday() for t in future_times],
                            "is_weekend": [1 if t.weekday() >= 5 else 0 for t in future_times],
                            "Datetime": future_times
                        })

                        future_df["Predicted Power (kW)"] = model.predict(
                            future_df[["hour", "dayofweek", "is_weekend"]]
                        )

                    st.markdown("<div class='section-header'>📈 Forecast Results</div>", unsafe_allow_html=True)

                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Average", f"{future_df['Predicted Power (kW)'].mean():.3f} kW")
                    with col2:
                        st.metric("📈 Maximum", f"{future_df['Predicted Power (kW)'].max():.3f} kW")
                    with col3:
                        st.metric("📉 Minimum", f"{future_df['Predicted Power (kW)'].min():.3f} kW")
                    with col4:
                        st.metric("📏 Range", f"{future_df['Predicted Power (kW)'].max() - future_df['Predicted Power (kW)'].min():.3f} kW")

                    # Interactive Plotly chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=future_df['Datetime'],
                        y=future_df['Predicted Power (kW)'],
                        mode='lines+markers',
                        name='Predicted Power',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=6, color='#764ba2'),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title="Energy Consumption Forecast",
                        xaxis_title="Date & Time",
                        yaxis_title="Power (kW)",
                        hovermode='x unified',
                        template='plotly_dark',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    with st.expander("📋 **View Detailed Forecast Table**"):
                        display_df = future_df.copy()
                        display_df['Datetime'] = display_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # =====================================================
    # 🔹 DEFAULT DATA FORECAST
    # =====================================================
    else:
        st.markdown("<div class='section-header'>⏳ Quick Forecast (Default Patterns)</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        
        with col1:
            option = st.selectbox(
                "**Select forecast horizon:**",
                ["Next 24 hours", "Next 3 days", "Next 7 days", "Custom"],
                help="Choose a pre-defined period or customize your own"
            )

        hours_map = {
            "Next 24 hours": 24,
            "Next 3 days": 72,
            "Next 7 days": 168
        }

        if option == "Custom":
            n_hours = st.slider("**Custom forecast hours:**", 1, 168, 24)
        else:
            n_hours = hours_map[option]
        
        with col2:
            st.metric("Period", f"{n_hours} hrs")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 **Generate Forecast**", use_container_width=True):
            with st.spinner("🔮 Creating your forecast..."):
                start_time = datetime.now()
                future_times = [
                    start_time + timedelta(hours=i)
                    for i in range(1, n_hours + 1)
                ]

                future_df = pd.DataFrame({
                    "hour": [t.hour for t in future_times],
                    "dayofweek": [t.weekday() for t in future_times],
                    "is_weekend": [1 if t.weekday() >= 5 else 0 for t in future_times],
                    "Datetime": future_times
                })

                future_df["Predicted Power (kW)"] = model.predict(
                    future_df[["hour", "dayofweek", "is_weekend"]]
                )

            st.markdown("<div class='section-header'>📈 Forecast Results</div>", unsafe_allow_html=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Average", f"{future_df['Predicted Power (kW)'].mean():.3f} kW", 
                         help="Mean predicted consumption")
            with col2:
                st.metric("📈 Peak", f"{future_df['Predicted Power (kW)'].max():.3f} kW",
                         help="Maximum predicted consumption")
            with col3:
                st.metric("📉 Minimum", f"{future_df['Predicted Power (kW)'].min():.3f} kW",
                         help="Minimum predicted consumption")
            with col4:
                total_kwh = future_df['Predicted Power (kW)'].sum()
                st.metric("⚡ Total Energy", f"{total_kwh:.2f} kWh",
                         help="Cumulative energy over period")

            # Interactive Plotly chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=future_df['Datetime'],
                y=future_df['Predicted Power (kW)'],
                mode='lines+markers',
                name='Predicted Power',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#764ba2'),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)',
                hovertemplate='<b>Time:</b> %{x}<br><b>Power:</b> %{y:.3f} kW<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f"Energy Consumption Forecast - Next {n_hours} Hours",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Date & Time",
                yaxis_title="Power Consumption (kW)",
                hovermode='x unified',
                template='plotly_dark',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Hourly breakdown by day
            future_df['Day'] = future_df['Datetime'].dt.day_name()
            daily_avg = future_df.groupby('Day')['Predicted Power (kW)'].mean().reset_index()
            
            if len(daily_avg) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    # Daily average chart
                    fig_daily = px.bar(
                        daily_avg,
                        x='Day',
                        y='Predicted Power (kW)',
                        title='Average Daily Consumption',
                        color='Predicted Power (kW)',
                        color_continuous_scale='Viridis'
                    )
                    fig_daily.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                with col2:
                    # Hourly pattern
                    hourly_avg = future_df.groupby('hour')['Predicted Power (kW)'].mean().reset_index()
                    fig_hourly = px.line(
                        hourly_avg,
                        x='hour',
                        y='Predicted Power (kW)',
                        title='Average Hourly Pattern',
                        markers=True
                    )
                    fig_hourly.update_layout(template='plotly_dark', height=400)
                    fig_hourly.update_traces(line_color='#667eea', marker=dict(size=8))
                    st.plotly_chart(fig_hourly, use_container_width=True)

            # Data table
            with st.expander("📋 **View Detailed Forecast Table**"):
                display_df = future_df.copy()
                display_df['Datetime'] = display_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['Day of Week'] = display_df['dayofweek'].map({
                    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                })
                display_df['Period'] = display_df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
                
                st.dataframe(
                    display_df[['Datetime', 'Day of Week', 'Period', 'Predicted Power (kW)']],
                    use_container_width=True,
                    hide_index=True
                )

# =====================================================
# 🔹 ABOUT THE AUTHOR
# =====================================================
if choice == "👤 About the Author":

    # Sidebar author info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Contact Information")
    
    try:
        st.sidebar.image("author_photo.png", width=150)
    except:
        st.sidebar.markdown("📷 *Photo not available*")
    
    st.sidebar.markdown("**Sara Arif**")
    st.sidebar.caption("CS Student | Data Analyst")

    # Main content
    st.markdown("""
    <div class="main-header">
        <h1>👤 About the Author</h1>
        <p>Meet the developer behind this application</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            st.image("author_photo.png", width=250)
        except:
            st.info("📷 Photo not available")
    
    # ❌ author-card REMOVED HERE
    with col2:
        st.markdown("""
            <h2 style='margin-top:0;'>Sara Arif</h2>
            <h4 style='opacity:0.9; margin-bottom:1.5rem;'>
                Computer Science Student | Aspiring Data Analyst
            </h4>
            <p style='line-height:1.8; font-size:1.1rem;'>
            A motivated 3rd-year Computer Science student with a passion for turning data into actionable insights.
            Specialized in data analysis, visualization, and machine learning with hands-on experience in building
            predictive models and interactive applications.
            </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Skills section (UNCHANGED)
    st.markdown("<div class='section-header'>💼 Technical Skills</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🐍 Programming</h4>
            • Python (Advanced)<br>
            • SQL<br>
            • Data Structures
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Data Analysis</h4>
            • Pandas & NumPy<br>
            • Matplotlib & Seaborn<br>
            • Power BI & Excel
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Machine Learning</h4>
            • Scikit-learn<br>
            • XGBoost<br>
            • Predictive Modeling
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Expertise areas (UNCHANGED)
    st.markdown("<div class='section-header'>🎯 Areas of Expertise</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📈 **Data Cleaning & Preprocessing**
        - 📊 **Statistical Analysis & Reporting**
        - 🎨 **Data Visualization & Dashboards**
        """)
    
    with col2:
        st.markdown("""
        - 🔮 **Predictive Analytics**
        - 🚀 **Web Application Development**
        - 📱 **Interactive Data Apps (Streamlit)**
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Connect section (UNCHANGED)
    st.markdown("<div class='section-header'>🌐 Connect With Me</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <a href="https://www.linkedin.com/in/sara-arif-7922642b8/" target="_blank">
            <div class="metric-card">
                <div style="font-size:2rem;">💼</div>
                <div style="margin-top:0.5rem;">LinkedIn</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="https://github.com/SaraArif6198" target="_blank">
            <div class="metric-card">
                <div style="font-size:2rem;">💻</div>
                <div style="margin-top:0.5rem;">GitHub</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <a href="https://share.streamlit.io/user/saraarif6198" target="_blank">
            <div class="metric-card">
                <div style="font-size:2rem;">🚀</div>
                <div style="margin-top:0.5rem;">Streamlit Apps</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>✨ Crafted with ❤️ using Streamlit & Python</p>
    <p>© 2026 <a href='https://www.linkedin.com/in/sara-arif-7922642b8/' target='_blank' style='color:#667eea;'>Sara Arif</a> | All Rights Reserved</p>
    <p style='font-size:0.8rem; margin-top:1rem;'>⚡ Powered by XGBoost • Built with Streamlit • Designed for Excellence</p>
</div>
""", unsafe_allow_html=True)
