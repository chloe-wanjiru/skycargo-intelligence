import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import plotly.express as px

# Data and backend function imports
from src.data_pipeline import load_and_prep_cargo_data
from src.model import train_baggage_risk_model, predict_risk
from src.optimizer import optimize_payload, calculate_delay_penalties

st.set_page_config(
    page_title="SkyCargo Intelligence",
    page_icon=":material/flight_takeoff:",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Create a top header row with a toggle button aligned to the right
header_col1, header_col2 = st.columns([6, 1])

with header_col1:
    st.markdown("### SKYCARGO INTELLIGENCE")

with header_col2:
    # Using a simple button or checkbox as a toggle switch
    is_dark = st.checkbox("Dark Theme", value=False)

# Apply styling rules based on `is_dark`
bg_color = "#121212" if is_dark else "#fcfbf9"
text_color = "#f5f5f5" if is_dark else "#1a1a1a"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Alpine airline editorial styling ---
st.markdown("""
    <style>
    :root {
        --ink: #171717;
        --muted: #73736f;
        --paper: #f4f1ea;
        --panel: #fffdf8;
        --panel-raised: #e8e4da;
        --line: rgba(23, 23, 23, .14);
        --cobalt: #2846d8;
        --lime: #d9f276;
        --orange: #f26b45;
    }

    .stApp {
        background:
            linear-gradient(rgba(23, 23, 23, .025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23, 23, 23, .025) 1px, transparent 1px),
            var(--paper);
        background-size: 32px 32px;
        color: var(--ink);
        font-family: 'Times New Roman', Times, serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding: 2.5rem clamp(1.25rem, 6vw, 6rem) 4rem;
    }

    .block-container {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--ink) !important;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1.25rem;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #fffdf8 !important;
        letter-spacing: -.03em;
    }

    section[data-testid="stSidebar"] label {
        color: #b8b8b1 !important;
        font-family: 'Times New Roman', Times, serif;
        font-size: .62rem !important;
        letter-spacing: .06em;
        text-transform: uppercase;
    }

    section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
        background: var(--lime) !important;
        border-color: var(--lime) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
        background: rgba(217, 242, 118, .25);
    }

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, .08) !important;
        border-color: var(--line) !important;
        color: #fffdf8 !important;
    }

    section[data-testid="stSidebar"] input[type="number"] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    .brand-mark {
        align-items: center;
        display: flex;
        gap: .75rem;
        margin-bottom: 3rem;
    }

    .brand-badge {
        align-items: center;
        background: var(--orange);
        border-radius: 50%;
        box-shadow: 5px 5px 0 var(--ink);
        color: #fffdf8;
        display: inline-flex;
        font-size: 1rem;
        font-weight: 700;
        height: 42px;
        justify-content: center;
        width: 42px;
    }

    .brand-name {
        color: #000000;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .eyebrow,
    .section-label {
        color: var(--cobalt);
        font-family: 'Times New Roman', Times, serif;
        font-size: .68rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .eyebrow {
        align-items: center;
        display: flex;
        gap: .65rem;
        margin: 0 0 1.35rem;
    }

    .eyebrow::before {
        background: var(--orange);
        border-radius: 50%;
        content: '';
        height: 8px;
        width: 8px;
    }

    h1 {
        color: var(--ink) !important;
        font-size: clamp(2.5rem, 5vw, 5.2rem) !important;
        font-weight: 700 !important;
        letter-spacing: -.075em !important;
        line-height: .92 !important;
        max-width: 900px;
    }

    h1 em, h2 em {
        color: var(--cobalt);
        font-style: normal;
    }

    h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -.045em;
    }

    h2 {
        font-size: clamp(1.5rem, 3vw, 2.4rem) !important;
        line-height: 1.05 !important;
    }

    .hero-copy {
        color: #565652;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 560px;
    }

    .section-rule {
        border-top: 1px solid var(--line);
        margin: 3rem 0 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        min-height: 116px;
        padding: 1rem 1.1rem;
        box-shadow: 6px 6px 0 rgba(23, 23, 23, .08);
        transition: transform .2s ease, border-color .2s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: var(--cobalt);
        transform: translateY(-2px);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-family: 'Times New Roman', Times, serif;
        font-size: .64rem !important;
        text-transform: uppercase;
        letter-spacing: .04em;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-size: 1.65rem !important;
        font-weight: 700;
        white-space: nowrap !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--cobalt) !important;
        font-family: 'Times New Roman', Times, serif;
        font-size: .68rem;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--cobalt), var(--orange)) !important;
    }

    div[data-testid="stProgress"] {
        background: rgba(23, 23, 23, .1);
        border-radius: 0;
        height: 10px;
    }

    .stButton > button {
        background: var(--cobalt);
        border: 1px solid var(--cobalt);
        border-radius: 3px;
        color: #fffdf8;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: var(--orange);
        border-color: var(--orange);
        color: #fffdf8;
    }

    [data-testid="stDataFrame"], .stPlotlyChart {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        border-top: 1px solid var(--line);
        padding: 1rem;
    }

    [data-testid="stDataFrame"] * {
        border-color: var(--line) !important;
    }

    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
    }

    [data-testid="stAlert"] {
        background: rgba(255, 253, 248, .9);
        border: 1px solid var(--line);
        color: var(--ink);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="brand-mark"><span class="brand-badge">S</span>'
    '<span class="brand-name">SkyCargo intelligence</span></div>',
    unsafe_allow_html=True,
)

st.markdown('<p class="eyebrow">Live flight operations / predictive analytics</p>', unsafe_allow_html=True)
st.markdown(
   '<p class="hero-copy">A real-time command center for baggage risk, payload capacity, '
   'and the financial impact of disruption.</p>',
   unsafe_allow_html=True,
)

@st.cache_resource
def get_data_and_model():
    df = load_and_prep_cargo_data()
    model, features = train_baggage_risk_model(df)
    return df, model

try:
    df, model = get_data_and_model()
    
    # --- Sidebar Controls ---
    st.sidebar.markdown('<p class="section-label">Control room / inputs</p>', unsafe_allow_html=True)
    st.sidebar.header("Set the Flight Envelope")
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

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    # --- Operational Loss Liabilities Dashboard ---
    st.markdown('<p class="section-label">01 / Cost of Disruption</p>', unsafe_allow_html=True)
    st.subheader("Delay Chain")
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

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    
    # --- Structural Payload Bar ---
    st.markdown('<p class="section-label">02 / Weight Profile & Limits</p>', unsafe_allow_html=True)
    st.subheader("Weight Stack")
    st.progress(min(opt_results['weight_status_pct'] / 100.0, 1.0))
    st.write(f"**Aircraft Payload Limit Used:** {opt_results['weight_status_pct']}%")

    # --- Operational Weight Breakdown Chart ---
    st.subheader("Weight Allocation")
    weight_data = pd.DataFrame({
        "Category": ["Passengers", "Baggage", "Available Cargo"],
        "Weight (kg)": [total_pass_wt, total_bag_wt, opt_results['available_cargo_capacity_kg']]
    })
    
    fig = px.bar(
        weight_data, 
        x="Category", 
        y="Weight (kg)", 
        color="Category",
        color_discrete_map={
            "Passengers": "#2846d8",
            "Baggage": "#d9f276",
            "Available Cargo": "#f26b45"
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
        font=dict(color='#171717', family='Times New Roman, Times, serif', size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(160,190,220,.14)')
    )
    
    fig.update_yaxes(range=[0, max_payload * 1.1])
    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.error(f"Error starting application: {e}")