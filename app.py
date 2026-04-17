import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- LOAD FILES ----------------
model = pickle.load(open("model/model.pkl", "rb"))
symptoms = pickle.load(open("model/encoder.pkl", "rb"))

# Safe loading (prevents crash if file issue)
try:
    desc_df = pd.read_csv("data/symptom_Description.csv")
except:
    desc_df = pd.DataFrame()

try:
    prec_df = pd.read_csv("data/symptom_precaution.csv")
except:
    prec_df = pd.DataFrame()

try:
    sev_df = pd.read_csv("data/Symptom-severity.csv")
except:
    sev_df = pd.DataFrame()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Disease Predictor", layout="centered")

st.markdown("<h1 style='text-align:center;'>🩺 Disease Prediction System</h1>", unsafe_allow_html=True)

# ---------------- USER DETAILS FORM ----------------
st.subheader("👤 Enter Patient Details")

with st.form("patient_form"):

    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=20)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    blood_group = st.selectbox(
        "Blood Group",
        ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-", "Not Sure"]
    )

    hereditary = st.text_input("Hereditary Diseases (if any)")

    days = st.slider("Days since symptoms started", 1, 30, 3)

    submitted = st.form_submit_button("Save Details")

# Save to session
if submitted:
    st.session_state["user"] = {
        "name": name,
        "age": age,
        "gender": gender,
        "blood_group": blood_group,
        "hereditary": hereditary,
        "days": days
    }

# ---------------- SIDEBAR DISPLAY ----------------
if "user" in st.session_state:

    user = st.session_state["user"]

    st.sidebar.title("👤 Patient Summary")
    st.sidebar.write(f"**Name:** {user['name']}")
    st.sidebar.write(f"**Age:** {user['age']}")
    st.sidebar.write(f"**Gender:** {user['gender']}")
    st.sidebar.write(f"**Blood Group:** {user['blood_group']}")
    st.sidebar.write(f"**Hereditary:** {user['hereditary']}")
    st.sidebar.write(f"**Days since symptoms:** {user['days']}")

    # ---------------- SYMPTOM INPUT ----------------
    st.subheader("🔍 Select Symptoms")

    selected = st.multiselect("Choose symptoms:", symptoms)

    input_data = [1 if s in selected else 0 for s in symptoms]
    input_data = np.array(input_data).reshape(1, -1)

    # ---------------- PREDICTION ----------------
    if st.button("🚀 Predict"):

        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        confidence = max(probs)

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"🧾 Prediction: {prediction}")

        with col2:
            st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # ---------------- TOP 3 ----------------
        st.subheader("🔝 Top 3 Possible Diseases")
        top3 = probs.argsort()[-3:][::-1]

        for i in top3:
            st.write(f"{model.classes_[i]} — {probs[i]*100:.2f}%")

        # ---------------- DESCRIPTION ----------------
        if not desc_df.empty:
            desc = desc_df[desc_df['Disease'].str.lower() == prediction.lower()]['Description'].values

            if len(desc) > 0:
                st.subheader("📖 Description")
                st.write(desc[0])

        # ---------------- PRECAUTIONS ----------------
        if not prec_df.empty:
            precautions = prec_df[prec_df['Disease'].str.lower() == prediction.lower()].values

            if len(precautions) > 0:
                st.subheader("🛡️ Precautions")
                for i in precautions[0][1:]:
                    if pd.notna(i):
                        st.write(f"✔ {i}")

        # ---------------- SEVERITY SCORE ----------------
        if not sev_df.empty:
            severity_score = 0

            for symptom in selected:
                val = sev_df[sev_df['Symptom'].str.lower() == symptom.lower()]['weight']
                if not val.empty:
                    severity_score += int(val.values[0])

            st.subheader("⚠️ Severity Level")

            if severity_score < 10:
                st.success(f"Low ({severity_score})")
            elif severity_score < 20:
                st.warning(f"Medium ({severity_score})")
            else:
                st.error(f"High ({severity_score})")

        # ---------------- RISK WARNINGS ----------------
        if user["age"] > 60:
            st.warning("⚠️ Higher risk due to age")

        if user["hereditary"]:
            st.warning("⚠️ Monitor condition due to hereditary factors")

# ---------------- DISCLAIMER ----------------
st.warning("⚠️ This is an AI-based prediction and not a medical diagnosis. Please consult a doctor.")
