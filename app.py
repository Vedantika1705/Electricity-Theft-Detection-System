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

# ================== LOAD MODEL ==================
with open("theft_prediction_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# ================== LOAD LOCATION CSV ==================
location_df = pd.read_csv("meter_locations.csv")

# ================== SESSION STATE ==================
if "meters" not in st.session_state:
    st.session_state.meters = []

# ================== FEATURES ==================
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
st.sidebar.title("⚡ Theft Detection")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "➕ Add Meter", "📊 Analysis", "ℹ️ About"]
)

# ================== GLOBAL CSS ==================
st.markdown("""
<style>
.fade-in {
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from {opacity:0; transform: translateY(20px);}
    to {opacity:1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ================== HOME ==================
# ================== HOME ==================
if menu == "🏠 Home":

    st.markdown("""
    <div class="fade-in" style="
        padding:60px;
        border-radius:25px;
        background: linear-gradient(135deg,#4f46e5,#9333ea);
        color:white;
        text-align:center;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
    ">
        <h1 style="font-size:48px;">⚡ Electricity Theft Detection System</h1>
        <p style="font-size:22px; margin-top:10px;">
             platform for detecting electricity theft & prioritizing inspections
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================== LIVE SYSTEM STATS ==================
    st.markdown("## 📊 Live System Overview")

    c1, c2, c3 = st.columns(3)

    total_meters = len(st.session_state.meters)

    if total_meters > 0:
        df_temp = pd.DataFrame(st.session_state.meters)
        probs = model.predict_proba(df_temp[FEATURE_COLUMNS])[:, 1]
        high_risk_count = sum(probs * 100 > 60)
        avg_risk = round((probs * 100).mean(), 2)
    else:
        high_risk_count = 0
        avg_risk = 0

    c1.metric("📟 Meters Analyzed", total_meters)
    c2.metric("🚨 High Risk Meters", high_risk_count)
    c3.metric("⚡ Average Theft Risk", f"{avg_risk}%")


    # ================== FEATURES ==================
    st.markdown("## 🔍 Key Capabilities")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        ### 📊 Smart Consumption Analysis
        - Detects abnormal usage patterns  
        - Identifies voltage irregularities  
        - Flags unusual spikes automatically
        """)

    with f2:
        st.markdown("""
        ### 🧠 Machine Learning Engine
        - Random Forest Classifier  
        - SMOTE for class imbalance  
        - Probability-based risk scoring
        """)

    with f3:
        st.markdown("""
        ### 🗺 Location-Aware Detection
        - Area-wise theft visualization  
        - Interactive risk maps  
        - Inspection priority ranking
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================== HOW IT WORKS ==================
    st.markdown("## ⚙️ How the System Works")

    st.markdown("""
    ➡️ **Step 1:** Meter data is collected (usage, voltage, payment delay, history)  
    ➡️ **Step 2:** Machine learning model analyzes consumption behavior  
    ➡️ **Step 3:** Theft risk percentage is generated  
    ➡️ **Step 4:** High-risk meters are highlighted on map & table  
    ➡️ **Step 5:** Inspection recommendations are provided
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================== CTA ==================
    st.info("👉 Use the **Add Meter** tab to input meter data and analyze theft risk in real-time.")


# ================== ADD METER ==================
if menu == "➕ Add Meter":
    st.markdown("## ➕ Add Meter")

    with st.form("meter_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            meter_id = st.text_input("Meter ID")
            usage = st.number_input("Usage (kWh)", 0.0)
            voltage = st.number_input("Voltage Fluctuations", 0.0)
            residents = st.number_input("Residents", 1)

        with c2:
            appliances = st.number_input("Appliance Count", 1)
            avg_usage = st.number_input("Average Daily Usage", 0.0)
            delay = st.number_input("Bill Payment Delay (Days)", 0)

        with c3:
            time_day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])
            industrial = st.selectbox("Industrial Area Nearby", ["No","Yes"])
            history = st.selectbox("Previous Theft History", ["No","Yes"])
            spike = st.selectbox("Unusual Usage Spike", ["No","Yes"])

        submit = st.form_submit_button("➕ Add Meter")

    if submit and meter_id:
        # ===== Fetch location from CSV =====
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

        st.success(f"✅ Meter added successfully (Area: {area})")

# ================== ANALYSIS ==================
if menu == "📊 Analysis":
    st.markdown("## 📊 Risk Analysis Dashboard")

    if not st.session_state.meters:
        st.warning("Please add meters first.")
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
            lambda x: "🚨 Immediate Inspection" if x > 60 else "🟢 Monitor"
        )

        high_risk = df[df["Risk %"] > 60]
        if not high_risk.empty:
            st.error(f"🚨 ALERT: {len(high_risk)} meter(s) require immediate inspection!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Meters", len(df))
        c2.metric("High Risk", len(high_risk))
        c3.metric("Average Risk", f"{df['Risk %'].mean():.2f}%")

        # ================== MAP ==================
        st.markdown("### 🗺 Theft Location Map")
        map_fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Risk %",
            size="Risk %",
            hover_name="Meter ID",
            hover_data=["Area","Risk %","Risk Category"],
            zoom=9,
            height=450,
            color_continuous_scale="Reds"
        )
        map_fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(map_fig, use_container_width=True)

        # ================== TABLE (FIXED VISIBILITY) ==================
        def color_rows(row):
            if row["Risk Category"] == "High Risk":
                return ["background-color: #fecaca; color: black; font-weight: bold;"] * len(row)
            else:
                return ["background-color: #bbf7d0; color: black;"] * len(row)

        st.markdown("### 📋 Detailed Risk Table")
        st.dataframe(
            df[[
                "Meter ID",
                "Area",
                "Risk %",
                "Risk Category",
                "Inspection Recommendation"
            ]].style.apply(color_rows, axis=1),
            use_container_width=True
        )

# ================== ABOUT ==================
if menu == "ℹ️ About":
    st.markdown("""
    <div style="
        padding:30px;
        border-radius:15px;
        background: linear-gradient(135deg,#111827,#1f2933);
        color:white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    ">
        <h2>ℹ️ About</h2>
        <h4>Electricity Theft Detection System</h4>
        <p style="font-size:16px; line-height:1.6;">
            This application is designed to assist electricity authorities in identifying
            potential electricity theft using machine learning techniques.
        </p>
        <ul style="font-size:16px; line-height:1.8;">
            <li>📍 Meter ID to Area mapping using CSV-based location data</li>
            <li>🧠 Theft risk prediction based on consumption behavior</li>
            <li>🗺️ Location-aware visualization of high-risk meters</li>
            <li>🚨 Inspection prioritization using ML-based risk scoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ================== RESET ==================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Data"):
    st.session_state.clear()
    st.rerun()
