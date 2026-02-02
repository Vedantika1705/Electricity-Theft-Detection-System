import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("theft_prediction_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

def main():
    st.set_page_config(
        page_title="Electricity Theft Prediction System",
        layout="wide"
    )

    # ---------- TITLE ----------
    st.markdown("""
        <h1 style="text-align:center; color:#ff4b4b;">
        🚨 Electricity Theft Prediction System
        </h1>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:gray;'>Predict electricity theft using consumption behavior</p>",
        unsafe_allow_html=True
    )

    # ---------- INPUT FORM ----------
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            usage = st.number_input("Usage (kWh)", min_value=0.0, step=0.1)
            voltage = st.number_input("Voltage Fluctuations", min_value=0.0, step=0.1)
            residents = st.number_input("Number of Residents", min_value=1, step=1)
            appliances = st.number_input("Appliance Count", min_value=1, step=1)

        with col2:
            time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            industrial = st.selectbox("Industrial Area Nearby", ["No", "Yes"])
            theft_history = st.selectbox("Previous Theft History", ["No", "Yes"])

        with col3:
            avg_usage = st.number_input("Average Daily Usage", min_value=0.0, step=0.1)
            payment_delay = st.number_input("Bill Payment Delay (days)", min_value=0, step=1)
            usage_spike = st.selectbox("Unusual Usage Spike", ["No", "Yes"])

        submit = st.form_submit_button("🔍 Predict")

    # ---------- PREDICTION ----------
    if submit:
        # Encode categorical inputs
        time_of_day = ["Morning", "Afternoon", "Evening", "Night"].index(time_of_day)
        industrial = ["No", "Yes"].index(industrial)
        theft_history = ["No", "Yes"].index(theft_history)
        usage_spike = ["No", "Yes"].index(usage_spike)

        input_data = np.array([[
            usage,
            time_of_day,
            voltage,
            residents,
            appliances,
            industrial,
            theft_history,
            avg_usage,
            payment_delay,
            usage_spike
        ]])

        prediction = model.predict(input_data)[0]

        # ---------- RISK SCORE ----------
        risk_score = int(
            10
            + usage_spike * 25
            + theft_history * 25
            + industrial * 15
            + min(payment_delay, 60) * 0.4
            + min(voltage, 5) * 4
        )
        risk_score = min(risk_score, 100)

        # ---------- RESULT ----------
        if prediction == 1:
            st.markdown(f"""
                <div style="background-color:#f8d7da;padding:25px;border-radius:10px;text-align:center;">
                    <h2 style="color:#721c24;">🚨 Theft Detected</h2>
                    <h3 style="color:#721c24;">Risk Level: {risk_score}%</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#d4edda;padding:25px;border-radius:10px;text-align:center;">
                    <h2 style="color:#155724;">✅ No Theft Detected</h2>
                    <h3 style="color:#155724;">Risk Level: {risk_score}%</h3>
                </div>
            """, unsafe_allow_html=True)

        # ---------- RISK BAR ----------
        st.markdown("### ⚡ Theft Risk Meter")
        st.progress(risk_score / 100)

        # ---------- EXPLANATION ----------
        st.markdown("### 🔍 Key Factors")
        reasons = []
        if usage_spike:
            reasons.append("• Unusual usage spike detected")
        if theft_history:
            reasons.append("• Previous theft history")
        if payment_delay > 30:
            reasons.append("• Long bill payment delay")
        if industrial:
            reasons.append("• Industrial area nearby")
        if voltage > 2:
            reasons.append("• High voltage fluctuations")

        if reasons:
            st.markdown("\n".join(reasons))
        else:
            st.markdown("• Normal electricity consumption pattern")

        # ---------- ACTION ----------
        st.markdown("### 📌 Recommendation")
        if risk_score >= 70:
            st.error("Immediate inspection recommended")
        elif risk_score >= 40:
            st.warning("Monitor consumption closely")
        else:
            st.success("No action required")

if __name__ == "__main__":
    main()
