import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import time

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide"
)

# ================== GLOBAL CSS ==================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 24px;
    font-weight: 900;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar Navigation Bigger + Spacious */
section[data-testid="stSidebar"] label {
    font-size: 20px !important;
    font-weight: 650 !important;
    padding: 8px 0px !important;
}

/* Space between options */
div[role="radiogroup"] label {
    margin-bottom: 12px !important;
}

/* Purple Radio Bullet */
div[role="radiogroup"] input[type="radio"] {
    accent-color: #7c3aed !important;
}

/* Selected Page Gradient Text */
div[role="radiogroup"] input:checked + div {
    font-weight: 800 !important;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Gradient Buttons */
div.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    transition: 0.25s;
}

div.stButton > button:hover {
    background: linear-gradient(135deg,#4338ca,#6d28d9);
    transform: scale(1.02);
}

/* Capability Cards */
.cap-card {
    background: rgba(255,255,255,0.06);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(150,150,150,0.25);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    transition: 0.3s ease-in-out;
    height: 100%;
}

.cap-card:hover {
    border: 1px solid #7c3aed;
    box-shadow: 0px 6px 18px rgba(124,58,237,0.25);
    transform: translateY(-6px);
}

/* Metric Cards Tint */
div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(79,70,229,0.12),
        rgba(124,58,237,0.10)
    );
    border: 1px solid rgba(124,58,237,0.25);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}

/* Metric Label Gradient */
div[data-testid="stMetricLabel"] > div {
    font-size: 18px !important;
    font-weight: 750 !important;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
with open("theft_prediction_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# ================== LOAD LOCATION CSV ==================
location_df = pd.read_csv("meter_locations.csv")

# ================== SESSION STATE ==================
if "meters" not in st.session_state:
    st.session_state.meters = []

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ================== FEATURE COLUMNS ==================
FEATURE_COLUMNS = [
    "Usage (kWh)",
    "TimeOfDay",
    "VoltageFluctuations",
    "NumberOfResidents",
    "ApplianceCount",
    "IndustrialAreaNearby",
    "PreviousTheftHistory",
    "AverageDailyUsage",
    "BillPaymentDelay (days)",
    "UnusualUsageSpike"
]

# ================== SIDEBAR ==================
st.sidebar.markdown(
    "<div class='sidebar-title'>⚡ Theft Detection System</div>",
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Add Meter", "Analysis", "About"],
    index=["Home", "Add Meter", "Analysis", "About"].index(st.session_state.page)
)

st.session_state.page = menu

st.sidebar.markdown("---")

if st.sidebar.button("Clear All Records"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# ========================== HOME ==========================
# =========================================================
if st.session_state.page == "Home":

    st.markdown("""
    <div style="
        padding:50px;
        border-radius:18px;
        background: linear-gradient(135deg,#4f46e5,#7c3aed);
        color:white;
        text-align:center;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.25);
    ">
        <h1>Electricity Theft Detection System</h1>
        <p style="font-size:18px; margin-top:10px;">
            Machine learning platform for theft risk prediction and inspection prioritization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(" ")

    # Live System Overview
    st.subheader(" Live System Overview")

    total_meters = len(st.session_state.meters)

    if total_meters > 0:
        df_temp = pd.DataFrame(st.session_state.meters)
        probs = model.predict_proba(df_temp[FEATURE_COLUMNS])[:, 1]
        high_risk_count = sum(probs * 100 > 60)
        avg_risk = round((probs * 100).mean(), 2)
    else:
        high_risk_count = 0
        avg_risk = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Meters Analyzed", total_meters)
    c2.metric("High Risk Alerts", high_risk_count)
    c3.metric("Average Risk Score", f"{avg_risk}%")

    st.markdown("---")

    # Key Capabilities
    st.subheader("Key Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="cap-card">
            <h3>📊 Smart Consumption Analysis</h3>
            <ul>
                <li>Detects abnormal usage patterns</li>
                <li>Flags unusual spikes automatically</li>
                <li>Monitors voltage irregularities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="cap-card">
            <h3>🧠 Machine Learning Engine</h3>
            <ul>
                <li>Random Forest Classifier</li>
                <li>SMOTE balancing for theft cases</li>
                <li>Probability-based risk scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="cap-card">
            <h3>🗺 Location-Based Insights</h3>
            <ul>
                <li>Area-wise theft visualization</li>
                <li>Interactive risk location map</li>
                <li>Inspection priority support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How System Works
    st.subheader(" How the System Works")

    st.markdown("""
    1. Meter data is collected (usage, voltage, payment delay, history).  
    2. The ML model analyzes consumption behavior.  
    3. Theft risk probability score is generated.  
    4. High-risk meters are highlighted in dashboard and map.  
    5. Authorities can prioritize inspections efficiently.  
    """)

    st.markdown(" ")

    if st.button("Proceed to Meter Data Entry"):
        st.session_state.page = "Add Meter"
        st.rerun()

# =========================================================
# ======================= ADD METER ========================
# =========================================================
if st.session_state.page == "Add Meter":

    st.title("Add Meter Record")

    with st.form("meter_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            meter_id = st.text_input("Meter ID")
            usage = st.number_input("Usage (kWh)", min_value=0.0)
            voltage = st.number_input("Voltage Fluctuations", min_value=0.0)
            residents = st.number_input("Residents", min_value=1)

        with c2:
            appliances = st.number_input("Appliance Count", min_value=1)
            avg_usage = st.number_input("Average Daily Usage", min_value=0.0)
            delay = st.number_input("Bill Payment Delay (Days)", min_value=0)

        with c3:
            time_day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])
            industrial = st.selectbox("Industrial Area Nearby", ["No","Yes"])
            history = st.selectbox("Previous Theft History", ["No","Yes"])
            spike = st.selectbox("Unusual Usage Spike", ["No","Yes"])

        submit = st.form_submit_button("Add Meter")

    if submit and meter_id:

        row = location_df[location_df["Meter ID"] == meter_id]

        if not row.empty:
            area = row.iloc[0]["Area"]
            lat = row.iloc[0]["Latitude"]
            lon = row.iloc[0]["Longitude"]
        else:
            area = "Unknown"
            lat = 18.5204
            lon = 73.8567

        st.session_state.meters.append({
            "Meter ID": meter_id,
            "Area": area,
            "Usage (kWh)": usage,
            "TimeOfDay": ["Morning","Afternoon","Evening","Night"].index(time_day),
            "VoltageFluctuations": voltage,
            "NumberOfResidents": residents,
            "ApplianceCount": appliances,
            "IndustrialAreaNearby": ["No","Yes"].index(industrial),
            "PreviousTheftHistory": ["No","Yes"].index(history),
            "AverageDailyUsage": avg_usage,
            "BillPaymentDelay (days)": delay,
            "UnusualUsageSpike": ["No","Yes"].index(spike),
            "Latitude": lat,
            "Longitude": lon
        })

        st.success(f"Meter added successfully (Area: {area})")

    if st.button("Go to Risk Analysis Dashboard"):
        st.session_state.page = "Analysis"
        st.rerun()

# =========================================================
# ======================== ANALYSIS ========================
# =========================================================
if st.session_state.page == "Analysis":

    st.title("Risk Analysis Dashboard")
    st.markdown(" ")
    if not st.session_state.meters:
        st.warning("Please add meter records first.")
    else:

        df = pd.DataFrame(st.session_state.meters)

        with st.spinner("Analyzing theft risk..."):
            time.sleep(1)

        probs = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        df["Risk %"] = (probs * 100).round(2)

        df["Risk Category"] = df["Risk %"].apply(
            lambda x: "High Risk" if x > 60 else "Low Risk"
        )

        df["Inspection Recommendation"] = df["Risk %"].apply(
            lambda x: "Immediate Inspection Required"
            if x > 60 else "Monitoring Recommended"
        )

        st.markdown("### 📋 Risk Summary Table")
        st.markdown(" ")
        # Dark-mode friendly highlight
        def highlight_high_risk(row):
            if row["Risk Category"] == "High Risk":
                return ["background-color: #dc2626; color: white; font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df = df[[
            "Meter ID", "Area", "Risk %", "Risk Category",
            "Inspection Recommendation"
        ]].style.apply(highlight_high_risk, axis=1)

        st.dataframe(styled_df, use_container_width=True)
        st.markdown(" ")

        st.markdown("### 🗺 Theft Risk Location Map")
        st.markdown(" ")

        map_fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Risk %",
            size="Risk %",
            hover_name="Meter ID",
            hover_data=["Area", "Risk %", "Risk Category"],
            zoom=9,
            height=430,
            color_continuous_scale="Reds"
        )

        map_fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(map_fig, use_container_width=True)
        st.markdown(" ")

    if st.button("View Project Information"):
        st.session_state.page = "About"
        st.rerun()

# =========================================================
# ========================== ABOUT =========================
# =========================================================
if st.session_state.page == "About":

    st.title("About This Project")

    st.markdown("""
    This application assists electricity authorities in detecting potential electricity theft
    using machine learning based risk prediction.

    **Core Features:**
    - Theft risk prediction using Random Forest  
    - Class imbalance correction using SMOTE  
    - Location-aware visualization for inspection planning  
    - Dashboard-based monitoring and reporting  
    """)

    st.success("Developed as an academic machine learning project.")
