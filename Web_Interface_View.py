import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("Employee Promotion Prediction")

# --------------------------------------------------
# Load Training Data (for dropdown values)
# --------------------------------------------------
df = pd.read_csv("train.csv")

# --------------------------------------------------
# Load Model (Pipeline)
# --------------------------------------------------
artifact = joblib.load("Emp_Promotion_model.pkl")
model = artifact["model"]
THRESHOLD = artifact["threshold"]

# --------------------------------------------------
# UI Inputs (Driven from CSV)
# --------------------------------------------------

department = st.selectbox(
    "Department",
    pd.unique(df['department'])
)

region = st.selectbox(
    "Region",
    pd.unique(df['region'])
)

education = st.selectbox(
    "Education",
    pd.unique(df['education'].dropna())
)

gender = st.selectbox(
    "Gender",
    pd.unique(df['gender'])
)

recruitment_channel = st.selectbox(
    "Recruitment Channel",
    pd.unique(df['recruitment_channel'])
)

no_of_trainings = st.number_input(
    "Number of Trainings", min_value=0, step=1
)

age = st.number_input(
    "Age", min_value=18, max_value=65
)

previous_year_rating = st.number_input(
    "Previous Year Rating (0–5)", min_value=0, max_value=5
)

length_of_service = st.number_input(
    "Length of Service (Years)", min_value=0
)

kpis_met = st.selectbox(
    "KPIs Met > 80%",
    pd.unique(df['KPIs_met >80%'])
)

awards_won = st.selectbox(
    "Awards Won",
    pd.unique(df['awards_won?'])
)

avg_training_score = st.number_input(
    "Average Training Score",
    min_value=40,
    max_value=100
)

# --------------------------------------------------
# Build Input Dictionary (Same Pattern as Your Example)
# --------------------------------------------------

inputs = {
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel,
    "no_of_trainings": no_of_trainings,
    "age": age,
    "previous_year_rating": previous_year_rating,
    "length_of_service": length_of_service,
    "KPIs_met >80%": int(kpis_met),
    "awards_won?": int(awards_won),
    "avg_training_score": avg_training_score
}

# --------------------------------------------------
# Predict Button Logic (Simple & Clean)
# --------------------------------------------------

if st.button("Predict"):
    x_input = pd.DataFrame([inputs])

    # Feature engineering (must match training)
    x_input['training_efficiency'] = (
        x_input['avg_training_score'] / (x_input['length_of_service'] + 1)
    )

    x_input['performance_index'] = (
        x_input['avg_training_score']
        * (x_input['KPIs_met >80%'] + 1)
        * (x_input['awards_won?'] + 1)
    )

    x_input['kpi_per_training'] = (
        x_input['KPIs_met >80%'] / (x_input['no_of_trainings'] + 1)
    )

    x_input['awards_per_service'] = (
        x_input['awards_won?'] / (x_input['length_of_service'] + 1)
    )

    x_input['score_times_awards'] = (
        x_input['avg_training_score'] * (x_input['awards_won?'] + 1)
    )

    x_input['score_times_kpi'] = (
        x_input['avg_training_score'] * (x_input['KPIs_met >80%'] + 1)
    )

    prob = model.predict_proba(x_input)[0, 1]
    prediction = int(prob >= THRESHOLD)

    st.subheader("Prediction Result")
    st.write(f"Promotion Probability: **{prob:.3f}**")
    st.write(f"Threshold Used: **{THRESHOLD:.2f}**")

    if prediction == 1:
        st.success("Employee is likely to be PROMOTED")
    else:
        st.warning("Employee is NOT likely to be promoted")
