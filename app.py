import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD TRAINED MODEL ----------------
with open("theft_prediction_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- SESSION STATE ----------------
if "meters" not in st.session_state:
    st.session_state.meters = []

# ---------------- FEATURE NAMES ----------------
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

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(
        page_title="Electricity Theft Detection System",
        layout="wide"
    )

    # ---------------- STYLING ----------------
    st.markdown("""
    <style>
        .section-gap { margin-top: 60px; }

        .gradient {
        
           background-image: linear-gradient(to top, #5f72bd 0%, #9b23ea 100%);
        }

        .header-banner {
            padding: 36px;
            border-radius: 10px;
            text-align: center;
        }

        .action-card {
            padding: 40px;
            border-radius: 12px;
        }

        .action-card h3 {
            font-size: 30px;
            margin-bottom: 14px;
            color: white;
        }

        .action-card p, .action-card li {
            font-size: 18px;
            color: #e5e7eb;
        }

        .count-badge {
            display: inline-block;
            padding: 18px 32px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: 600;
            margin-right: 20px;
        }

        .stButton > button {
            width: 100%;
            height: 48px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            color: white;
           
            background-image: linear-gradient(to top, #5f72bd 0%, #9b23ea 100%);
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- HEADER ----------------
    st.markdown("""
    <div class="gradient header-banner">
        <h1>Electricity Theft Detection System</h1>
        <p style="font-size:20px; margin-top:12px;">
            Circuit-Level Meter Risk Analysis Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- INPUT SECTION ----------------
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("## Meter Data Entry")

    with st.form("meter_form"):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            meter_id = st.text_input("Meter Identifier")
            usage = st.number_input("Energy Usage (kWh)", min_value=0.0, step=0.1)
            voltage = st.number_input("Voltage Fluctuations", min_value=0.0, step=0.1)
            residents = st.number_input("Number of Residents", min_value=1, step=1)

        with col2:
            appliances = st.number_input("Appliance Count", min_value=1, step=1)
            avg_usage = st.number_input("Average Daily Usage", min_value=0.0, step=0.1)
            payment_delay = st.number_input("Bill Payment Delay (Days)", min_value=0, step=1)

        with col3:
            time_of_day = st.selectbox(
                "Time of Day",
                ["Morning", "Afternoon", "Evening", "Night"]
            )
            industrial = st.selectbox("Industrial Area Nearby", ["No", "Yes"])
            theft_history = st.selectbox("Previous Theft History", ["No", "Yes"])
            usage_spike = st.selectbox("Unusual Usage Spike", ["No", "Yes"])

        add_meter = st.form_submit_button("Add Meter to Circuit")

    if add_meter and meter_id.strip():
        st.session_state.meters.append({
            "Meter ID": meter_id,
            "Usage (kWh)": usage,
            "TimeOfDay": ["Morning", "Afternoon", "Evening", "Night"].index(time_of_day),
            "VoltageFluctuations": voltage,
            "NumberOfResidents": residents,
            "ApplianceCount": appliances,
            "IndustrialAreaNearby": ["No", "Yes"].index(industrial),
            "PreviousTheftHistory": ["No", "Yes"].index(theft_history),
            "AverageDailyUsage": avg_usage,
            "BillPaymentDelay (days)": payment_delay,
            "UnusualUsageSpike": ["No", "Yes"].index(usage_spike)
        })
        st.success(f"Meter '{meter_id}' added to circuit.")

    # ---------------- METER OVERVIEW ----------------
    if st.session_state.meters:
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown("## Circuit Meter Overview")

        overview_df = pd.DataFrame(st.session_state.meters)
        overview_df.index = overview_df.index + 1
        overview_df.index.name = "Sr. No."

        st.dataframe(overview_df, use_container_width=True)

    # ---------------- ANALYSIS ----------------
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    if st.button("Run Circuit Risk Analysis"):
        if not st.session_state.meters:
            st.warning("No meter data available.")
            return

        df = pd.DataFrame(st.session_state.meters)
        probs = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        df["Risk Percentage"] = (probs * 100).round(2)

        # ---- ONLY HIGH / LOW ----
        df["Risk Category"] = df["Risk Percentage"].apply(
            lambda x: "High Risk" if x > 60 else "Low Risk"
        )

        df = df.sort_values("Risk Percentage", ascending=False)

        # ---------------- COUNT BADGES ----------------
        high = (df["Risk Category"] == "High Risk").sum()
        low = (df["Risk Category"] == "Low Risk").sum()

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div>
            <span class="gradient count-badge">High Risk: {high}</span>
            <span class="gradient count-badge">Low Risk: {low}</span>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- RESULTS TABLE (NO %) ----------------
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown("## Circuit Risk Assessment Results")

        result_df = df[["Meter ID", "Risk Category"]].copy()
        result_df.index = range(1, len(result_df) + 1)
        result_df.index.name = "Sr. No."

        st.dataframe(result_df, use_container_width=True)

        # ---------------- HIGH-RISK DETAILS ----------------
        high_df = df[df["Risk Category"] == "High Risk"]

        if not high_df.empty:
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="gradient action-card">
                <h3>High-Risk Meters – Detailed View</h3>
                <p><b>Total High-Risk Meters:</b> {len(high_df)}</p>
                <ul>
                    {''.join(f"<li>{row['Meter ID']} – {row['Risk Percentage']}%</li>" for _, row in high_df.iterrows())}
                </ul>
                <p style="margin-top:16px;">
                    <b>Recommended Action:</b><br>
                    Immediate physical inspection is strongly recommended for all listed meters.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("No high-risk meters detected in this circuit.")

    # ---------------- RESET ----------------
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    if st.button("Reset Circuit Data"):
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()
