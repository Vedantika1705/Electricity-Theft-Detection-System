
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
@st.cache_resource
def load_model():
    with open("theft_prediction_pipeline.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()


# ================== LOAD LOCATION CSV ==================
@st.cache_data
def load_locations():
    locations = pd.read_csv("meter_locations.csv")

    locations.columns = locations.columns.str.strip()

    required_columns = [
        "Meter ID", "Area", "Latitude", "Longitude"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in locations.columns
    ]

    if missing_columns:
        st.error(
            "Missing columns in meter_locations.csv: "
            + ", ".join(missing_columns)
        )
        st.stop()

    locations["Meter ID"] = (
        locations["Meter ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    locations["Latitude"] = pd.to_numeric(
        locations["Latitude"], errors="coerce"
    )

    locations["Longitude"] = pd.to_numeric(
        locations["Longitude"], errors="coerce"
    )

    return locations


location_df = load_locations()


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


# ================== THEFT PROBABILITY ==================
def get_theft_probabilities(features):
    """
    Return the predicted probability of class 1 (Theft).
    This avoids assuming that class 1 is always at index 1.
    """
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The loaded model does not contain class 1 (Theft). "
            "Please check the trained model and target labels."
        )

    theft_index = classes.index(1)

    return model.predict_proba(features)[:, theft_index]


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
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# ================== HOME ==================
if menu == "🏠 Home":

    st.markdown("""
    <div class="fade-in" style="
        padding: 45px;
        border-radius: 25px;
        background: linear-gradient(135deg,#4f46e5,#9333ea);
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <h1 style="font-size:42px;">
            ⚡ Electricity Theft Detection System
        </h1>
        <p style="font-size:20px;">
            Detecting potential electricity theft and prioritizing inspections
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📊 Live System Overview")

    c1, c2, c3 = st.columns(3)

    total_meters = len(st.session_state.meters)

    if total_meters:
        df_temp = pd.DataFrame(st.session_state.meters)

        probs = get_theft_probabilities(
            df_temp[FEATURE_COLUMNS]
        )

        model_scores = probs * 100

        high_risk_count = int((model_scores > 60).sum())
        avg_risk = round(model_scores.mean(), 2)

    else:
        high_risk_count = 0
        avg_risk = 0

    c1.metric("📟 Meters Analyzed", total_meters)
    c2.metric("🚨 High Risk Meters", high_risk_count)
    c3.metric("⚡ Average Model Risk Score", f"{avg_risk}%")

    st.caption(
        "Model risk scores support inspection prioritization; "
        "they do not confirm electricity theft."
    )

    st.markdown("## 🔍 Key Capabilities")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        ### 📊 Smart Consumption Analysis
        - Detects abnormal usage patterns
        - Identifies voltage irregularities
        - Flags unusual usage spikes
        """)

    with f2:
        st.markdown("""
        ### 🧠 Machine Learning Engine
        - Random Forest classifier
        - SMOTE-based class balancing during training
        - Model-based risk scoring
        """)

    with f3:
        st.markdown("""
        ### 🗺 Location-Aware Detection
        - Area-wise meter visualization
        - Interactive location map
        - Inspection prioritization
        """)

    st.markdown("## ⚙️ How the System Works")

    st.markdown("""
    1. Meter consumption and related data are entered.
    2. The trained machine-learning model analyzes the data.
    3. A model risk score is generated.
    4. Meter locations and risk scores are displayed.
    5. Higher-scoring records are prioritized for inspection.
    """)

    st.info(
        "Use the Add Meter tab to enter meter data and calculate a model risk score."
    )


# ================== ADD METER ==================
elif menu == "➕ Add Meter":

    st.markdown("## ➕ Add Meter")

    with st.form("meter_form", clear_on_submit=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            meter_id = st.text_input("Meter ID").strip().upper()

            usage = st.number_input(
                "Usage (kWh)", min_value=0.0
            )

            voltage = st.number_input(
                "Voltage Fluctuations", min_value=0.0
            )

            residents = st.number_input(
                "Number of Residents",
                min_value=1,
                step=1
            )

        with c2:
            appliances = st.number_input(
                "Appliance Count",
                min_value=1,
                step=1
            )

            avg_usage = st.number_input(
                "Average Daily Usage",
                min_value=0.0
            )

            delay = st.number_input(
                "Bill Payment Delay (Days)",
                min_value=0,
                step=1
            )

        with c3:
            time_day = st.selectbox(
                "Time of Day",
                ["Morning", "Afternoon", "Evening", "Night"]
            )

            industrial = st.selectbox(
                "Industrial Area Nearby",
                ["No", "Yes"]
            )

            history = st.selectbox(
                "Previous Theft History",
                ["No", "Yes"]
            )

            spike = st.selectbox(
                "Unusual Usage Spike",
                ["No", "Yes"]
            )

        submit = st.form_submit_button("➕ Add Meter")

    if submit:

        if not meter_id:
            st.error("Please enter a Meter ID.")

        elif meter_id in [
            m["Meter ID"] for m in st.session_state.meters
        ]:
            st.warning("This Meter ID has already been added.")

        else:
            row = location_df[
                location_df["Meter ID"] == meter_id
            ]

            if row.empty:
                st.error(
                    f"Meter ID '{meter_id}' was not found in "
                    "meter_locations.csv. Add this ID and its coordinates "
                    "to the CSV, then try again."
                )

            else:
                loc = row.iloc[0]
                lat = loc["Latitude"]
                lon = loc["Longitude"]
                area = str(loc["Area"])

                if (
                    pd.isna(lat)
                    or pd.isna(lon)
                    or not (-90 <= lat <= 90)
                    or not (-180 <= lon <= 180)
                ):
                    st.error(
                        "This meter has invalid latitude or longitude "
                        "in meter_locations.csv. Please correct the CSV."
                    )

                else:
                    meter = {
                        "Meter ID": meter_id,
                        "Area": area,
                        "Usage (kWh)": usage,
                        "TimeOfDay": [
                            "Morning",
                            "Afternoon",
                            "Evening",
                            "Night"
                        ].index(time_day),
                        "VoltageFluctuations": voltage,
                        "NumberOfResidents": residents,
                        "ApplianceCount": appliances,
                        "IndustrialAreaNearby": [
                            "No", "Yes"
                        ].index(industrial),
                        "PreviousTheftHistory": [
                            "No", "Yes"
                        ].index(history),
                        "AverageDailyUsage": avg_usage,
                        "BillPaymentDelay (days)": delay,
                        "UnusualUsageSpike": [
                            "No", "Yes"
                        ].index(spike),
                        "Latitude": float(lat),
                        "Longitude": float(lon)
                    }

                    st.session_state.meters.append(meter)

                    st.success(
                        f"Meter {meter_id} added successfully. "
                        f"Area: {area}"
                    )


# ================== ANALYSIS ==================
elif menu == "📊 Analysis":

    st.markdown("## 📊 Risk Analysis Dashboard")

    if not st.session_state.meters:
        st.warning("Please add meters first.")

    else:
        df = pd.DataFrame(st.session_state.meters)

        with st.spinner("Analyzing theft risk..."):
            time.sleep(0.5)

            probs = get_theft_probabilities(
                df[FEATURE_COLUMNS]
            )

        df["Model Risk Score (%)"] = (
            probs * 100
        ).round(2)

        df["Risk Category"] = df[
            "Model Risk Score (%)"
        ].apply(
            lambda score: (
                "High Risk" if score > 60
                else "Low Risk"
            )
        )

        df["Inspection Recommendation"] = df[
            "Model Risk Score (%)"
        ].apply(
            lambda score: (
                "🚨 Prioritize Inspection"
                if score > 60
                else "🟢 Monitor"
            )
        )

        # ================== SUMMARY METRICS ==================
        high_risk = df[
            df["Model Risk Score (%)"] > 60
        ]

        if not high_risk.empty:
            st.warning(
                f"{len(high_risk)} meter(s) are flagged for "
                "inspection based on their model-estimated scores."
            )

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Meters", len(df))
        c2.metric("High Risk", len(high_risk))
        c3.metric(
            "Average Model Risk Score",
            f"{df['Model Risk Score (%)'].mean():.2f}%"
        )

        # ================== LOCATION MAP ==================
        st.markdown("### 🗺 Theft Location Map")

        map_df = df.copy()

        map_df["Latitude"] = pd.to_numeric(
            map_df["Latitude"],
            errors="coerce"
        )

        map_df["Longitude"] = pd.to_numeric(
            map_df["Longitude"],
            errors="coerce"
        )

        map_df = map_df.dropna(
            subset=["Latitude", "Longitude"]
        )

        map_df = map_df[
            map_df["Latitude"].between(-90, 90)
            & map_df["Longitude"].between(-180, 180)
        ].copy()

        if map_df.empty:
            st.warning(
                "No valid coordinates are available. Check "
                "meter_locations.csv and the meter records."
            )

        else:
            map_df["Marker Size"] = (
                map_df["Model Risk Score (%)"].clip(lower=1)
            )

            map_fig = px.scatter_map(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Model Risk Score (%)",
                size="Marker Size",
                size_max=20,
                hover_name="Meter ID",
                hover_data={
                    "Area": True,
                    "Model Risk Score (%)": True,
                    "Risk Category": True,
                    "Latitude": True,
                    "Longitude": True,
                    "Marker Size": False
                },
                zoom=10,
                height=500,
                color_continuous_scale="Reds",
                range_color=[0, 100]
            )

            map_fig.update_layout(
                map_style="open-street-map",
                map_center={
                    "lat": float(map_df["Latitude"].mean()),
                    "lon": float(map_df["Longitude"].mean())
                },
                margin={
                    "r": 0,
                    "t": 0,
                    "l": 0,
                    "b": 0
                }
            )

            st.plotly_chart(
                map_fig,
                use_container_width=True
            )

            st.caption(
                "Hover over a marker to inspect its meter ID, area, "
                "coordinates, and model risk score. The score is an "
                "indicator for prioritization, not proof of theft."
            )

        # ================== DETAILED RISK TABLE ==================
        st.markdown("### 📋 Detailed Risk Table")

        display_columns = [
            "Meter ID",
            "Area",
            "Model Risk Score (%)",
            "Risk Category",
            "Inspection Recommendation"
        ]

        def color_rows(row):
            if row["Risk Category"] == "High Risk":
                style = (
                    "background-color: #fecaca; "
                    "color: black; font-weight: bold;"
                )
            else:
                style = (
                    "background-color: #bbf7d0; "
                    "color: black;"
                )

            return [style] * len(row)

        st.dataframe(
            df[display_columns].style.apply(
                color_rows,
                axis=1
            ),
            use_container_width=True
        )

        st.caption(
            "Risk categories use a 60% model-score threshold. "
            "These predictions should be verified through appropriate "
            "inspection and should not be treated as confirmed theft."
        )


# ================== ABOUT ==================
elif menu == "ℹ️ About":

    st.markdown("""
    <div style="
        padding:30px;
        border-radius:15px;
        background:linear-gradient(135deg,#111827,#1f2933);
        color:white;
    ">
        <h2>ℹ️ About</h2>
        <h4>Electricity Theft Detection System</h4>
        <p style="font-size:16px; line-height:1.6;">
            This application helps identify potentially unusual electricity
            consumption patterns using a trained machine-learning model.
            Its results are intended to support inspection prioritization.
        </p>
        <ul style="font-size:16px; line-height:1.8;">
            <li>Meter ID and area mapping using CSV location data</li>
            <li>Model risk scores based on meter-related features</li>
            <li>Interactive map with meter coordinates</li>
            <li>Inspection prioritization based on estimated risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ================== RESET ==================
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Data"):
    st.session_state.meters = []
    st.rerun()
