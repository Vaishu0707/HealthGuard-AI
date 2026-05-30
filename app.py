import streamlit as st
import sqlite3
import pandas as pd
import joblib
import re
from datetime import date

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Load ML Model
# -----------------------------
try:
    model = joblib.load("health_model.pkl")
    encoder = joblib.load("encoder.pkl")
except:
    st.error("Model files not found! Run train_model.py first.")
    st.stop()

# -----------------------------
# Database Connection
# -----------------------------
conn = sqlite3.connect(
    "patients.db",
    check_same_thread=False
)

cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    dob TEXT,
    email TEXT,
    glucose REAL,
    haemoglobin REAL,
    cholesterol REAL,
    remarks TEXT
)
""")

conn.commit()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_health(glucose, haemoglobin, cholesterol):

    input_data = pd.DataFrame({
        "glucose": [glucose],
        "haemoglobin": [haemoglobin],
        "cholesterol": [cholesterol]
    })

    prediction = model.predict(input_data)

    result = encoder.inverse_transform(prediction)

    return result[0]


# -----------------------------
# Email Validation
# -----------------------------
def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)


# -----------------------------
# App Title
# -----------------------------
st.title("🏥 HealthGuard AI")
st.markdown("### Smart Health Risk Predictor")

menu = st.sidebar.selectbox(
    "Choose Option",
    [
        "Add Patient",
        "View Patients",
        "Update Patient",
        "Delete Patient"
    ]
)

# ==================================
# ADD PATIENT
# ==================================
if menu == "Add Patient":

    st.subheader("Add Patient Details")

    full_name = st.text_input("Full Name")

    dob = st.date_input(
        "Date of Birth",
        min_value=date(1950, 1, 1),
        max_value=date.today()
    )

    email = st.text_input("Email Address")

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        step=1.0
    )

    haemoglobin = st.number_input(
        "Haemoglobin",
        min_value=0.0,
        step=0.1
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0.0,
        step=1.0
    )

    if st.button("Predict & Save"):

        if full_name == "":
            st.error("Please enter patient name")

        elif not validate_email(email):
            st.error("Enter valid email address")

        else:

            prediction = predict_health(
                glucose,
                haemoglobin,
                cholesterol
            )

            cursor.execute("""
            INSERT INTO patients (
                full_name,
                dob,
                email,
                glucose,
                haemoglobin,
                cholesterol,
                remarks
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                full_name,
                str(dob),
                email,
                glucose,
                haemoglobin,
                cholesterol,
                prediction
            ))

            conn.commit()

            st.success(
                "Patient Added Successfully!"
            )

            st.info(
                f"AI Prediction: {prediction}"
            )

# ==================================
# VIEW PATIENTS
# ==================================
elif menu == "View Patients":

    st.subheader("Patient Records")

    data = pd.read_sql_query(
        "SELECT * FROM patients",
        conn
    )

    st.dataframe(
        data,
        use_container_width=True
    )

# ==================================
# UPDATE PATIENT
# ==================================
elif menu == "Update Patient":

    st.subheader("Update Patient")

    patient_id = st.number_input(
        "Enter Patient ID",
        min_value=1,
        step=1
    )

    new_name = st.text_input(
        "New Full Name"
    )

    new_email = st.text_input(
        "New Email Address"
    )

    if st.button("Update Record"):

        cursor.execute("""
        UPDATE patients
        SET full_name=?,
            email=?
        WHERE id=?
        """, (
            new_name,
            new_email,
            patient_id
        ))

        conn.commit()

        st.success(
            "Patient Updated Successfully!"
        )

# ==================================
# DELETE PATIENT
# ==================================
elif menu == "Delete Patient":

    st.subheader("Delete Patient")

    patient_id = st.number_input(
        "Enter Patient ID",
        min_value=1,
        step=1
    )

    if st.button("Delete Record"):

        cursor.execute(
            "DELETE FROM patients WHERE id=?",
            (patient_id,)
        )

        conn.commit()

        st.success(
            "Patient Deleted Successfully!"
        )