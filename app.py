import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import plotly.express as px

# Data and backend function imports
from src.data_pipeline import load_and_prep_cargo_data
from src.model import train_baggage_risk_model, predict_risk
from src.optimizer import optimize_payload, calculate_delay_penalties

st.set_page_config(page_title="Cargo Payload & Baggage Intelligence System", layout="wide")

# --- Executive Dark Theme Styling ---
st.markdown("""
    <style>
    /* Main Background & Global Typography */
    .stApp {
        background-color: #0F0F12;
        color: #F5F5F7;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #16161C !important;
        border-right: 1px solid rgba(212, 175, 55, 0.4);
    }
    
    /* Headers & Accent Colors */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: 1.5px;
    }
    h2, h3 {
        color: #D4AF37 !important;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    /* KPI Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1C1C24, #121217);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    
    /* Label Size & Formatting to Prevent Ellipses (...) */
    div[data-testid="stMetric"] label {
        color: #D4AF37 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        white-space: nowrap !important;
    }
    
    /* Progress Bar Accent */
    .stProgress > div > div > div > div {
        background-color: #E50914 !important;
    }
    
    /* Slider Accent Styling (Clean, no red tooltip box) */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #E50914 !important;
        border-color: #E50914 !important;
    }
    div[data-baseweb="slider"] div[data-testid="stTickBar"] + div {
        background-color: #E50914 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("SkyCargo & Baggage Intelligence System")
st.caption("Flight Operations & Linear Payload Optimization Engine")

@st.cache_resource
def get_data_and_model():
    df = load_and_prep_cargo_data()
    model, features = train_baggage_risk_model(df)
    return df, model

try:
    df, model = get_data_and_model()
    
    # --- Sidebar Controls ---
    st.sidebar.header("Flight Operational Inputs")
    dep_delay = st.sidebar.slider("Departure Delay (mins)", 0, 360, 15)
    connect_time = st.sidebar.slider("Connecting Window (mins)", 15, 120, 35)
    passengers = st.sidebar.slider("Passenger Count", 50, 220, 150)
    bag_count = st.sidebar.slider("Checked Baggage Count", 50, 300, 180)
    max_payload = st.sidebar.number_input("Max Aircraft Payload Capacity (kg)", value=18000)

    # --- Cargo Weight Calculations ---
    avg_pass_weight = 80.0  # kg
    avg_bag_weight = 18.0   # kg
    
    total_pass_wt = passengers * avg_pass_weight
    total_bag_wt = bag_count * avg_bag_weight
    
    # --- Predictive Model Call ---
    input_features = {
        'MONTH': 6,
        'DAY_OF_WEEK': 3,
        'DISTANCE': 1200,
        'DEP_DELAY': dep_delay,
        'connect_time_mins': connect_time,
        'baggage_count': bag_count
    }
    risk_score = predict_risk(model, input_features)
    
    # --- Optimization Engine Calls ---
    opt_results = optimize_payload(max_payload, total_pass_wt, total_bag_wt)
    delay_metrics = calculate_delay_penalties(dep_delay, passengers)
    
    # --- Core Dashboard Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    risk_label = "High Risk" if risk_score > 0.5 else "Low Risk"
    
    cargo_cap = int(opt_results['available_cargo_capacity_kg'])
    cargo_rev = int(opt_results['cargo_revenue_est'])
    
    col1.metric(
        label="Baggage Risk", 
        value=f"{round(risk_score * 100, 1)}%", 
        delta=risk_label, 
        delta_color="inverse" if risk_score > 0.5 else "normal"
    )
    col2.metric(
        label="Pass & Bag Load", 
        value=f"{int(total_pass_wt + total_bag_wt):,} kg"
    )
    col3.metric(
        label="Opt. Cargo Capacity", 
        value=f"{cargo_cap:,} kg"
    )
    col4.metric(
        label="Est. Cargo Revenue", 
        value=f"${cargo_rev:,}"
    )

    st.markdown("---")

    # --- Operational Loss Liabilities Dashboard ---
    st.subheader("Financial Impact & Delay Liabilities")
    d_col1, d_col2, d_col3 = st.columns(3)

    total_loss = delay_metrics['total_delay_loss']
    food_loss = delay_metrics['food_voucher_cost']
    payout_loss = delay_metrics['hotel_cost'] + delay_metrics['compensation_cost']

    with d_col1:
        st.metric(
            label="Est. Delay Liabilities", 
            value=f"${total_loss:,.2f}",
            delta=f"+${total_loss:,.2f} Loss" if total_loss > 0 else "+$0.00 (No Loss)",
            delta_color="inverse" if total_loss > 0 else "normal"
        )

    with d_col2:
        st.metric(
            label="Food Vouchers", 
            value=f"${food_loss:,.2f}",
            delta=f"+${food_loss:,.2f} Cost" if food_loss > 0 else "+$0.00 (No Cost)",
            delta_color="inverse" if food_loss > 0 else "normal"
        )

    with d_col3:
        st.metric(
            label="Hotels & Payouts", 
            value=f"${payout_loss:,.2f}",
            delta=f"+${payout_loss:,.2f} Cost" if payout_loss > 0 else "+$0.00 (No Cost)",
            delta_color="inverse" if payout_loss > 0 else "normal"
        )

    st.markdown("---")
    
    # --- Structural Payload Bar ---
    st.subheader("Payload Weight Capacity Utilization")
    st.progress(min(opt_results['weight_status_pct'] / 100.0, 1.0))
    st.write(f"**Aircraft Payload Limit Used:** {opt_results['weight_status_pct']}%")

    # --- Operational Weight Breakdown Chart ---
    st.subheader("Operational Weight Breakdown")
    weight_data = pd.DataFrame({
        "Category": ["Passengers", "Baggage", "Available Cargo"],
        "Weight (kg)": [total_pass_wt, total_bag_wt, opt_results['available_cargo_capacity_kg']]
    })
    
    fig = px.bar(
        weight_data, 
        x="Category", 
        y="Weight (kg)", 
        color="Category",
        # Distinct high-contrast palette: Gold, Electric Cyan, Crimson
        color_discrete_map={
            "Passengers": "#D4AF37", 
            "Baggage": "#00E5FF", 
            "Available Cargo": "#E50914"
        },
        text_auto='.0f'
    )
    
    fig.update_traces(
        textfont_size=14,
        textfont_color="#FFFFFF",
        textposition="inside"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F5F5F7', size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#262630')
    )
    
    fig.update_yaxes(range=[0, max_payload * 1.1])
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error starting application: {e}")