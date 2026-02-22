import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Define the custom transformer EXACTLY as it is in the notebook
# Joblib needs this class definition to successfully unpickle the pipeline
class BloodPressureSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Split 'Blood Pressure' into 'Systolic' and 'Diastolic'
        bp_split = X_copy['Blood Pressure'].str.split('/', expand=True)
        X_copy['Systolic'] = bp_split[0].astype(int)
        X_copy['Diastolic'] = bp_split[1].astype(int)
        # Drop the original 'Blood Pressure' column
        X_copy = X_copy.drop('Blood Pressure', axis=1)
        return X_copy

# 2. Load the model safely using Streamlit's cache to prevent reloading on every interaction
@st.cache_resource
def load_model():
    # Make sure this file is in the same directory as your streamlit_app.py
    return joblib.load('heart_attack_prediction_pipeline.joblib')

pipeline = load_model()

# 3. Application Title & Description
st.title("🫀 Heart Attack Risk Predictor")
st.markdown("Enter the patient's vitals and lifestyle metrics below to assess their heart attack risk.")

# 4. User Input Widgets
# Grouping inputs logically into columns for a cleaner UI
st.subheader("Patient Vitals")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=45)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])
    # Text input for Blood Pressure to match the notebook's string split logic
    blood_pressure = st.text_input("Blood Pressure (Systolic/Diastolic)", value="120/80")
    triglycerides = st.number_input("Triglycerides", min_value=30, max_value=1000, value=150)

with col3:
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)

st.subheader("Lifestyle & Medical History")
col4, col5, col6 = st.columns(3)

with col4:
    diet = st.selectbox("Diet", options=["Healthy", "Average", "Unhealthy"])
    exercise_hours = st.number_input("Exercise (Hours/Week)", min_value=0.0, max_value=40.0, value=3.0)
    physical_activity_days = st.slider("Physical Activity (Days/Week)", 0, 7, 3)

with col5:
    sleep_hours = st.number_input("Sleep (Hours/Day)", min_value=2, max_value=16, value=7)
    sedentary_hours = st.number_input("Sedentary (Hours/Day)", min_value=0.0, max_value=24.0, value=6.0)
    smoking = st.selectbox("Smoking?", options=["No", "Yes"])

with col6:
    alcohol = st.selectbox("Alcohol Consumption?", options=["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", options=["No", "Yes"])
    obesity = st.selectbox("Obesity?", options=["No", "Yes"])

st.subheader("Family History & Geography")
col7, col8 = st.columns(2)

with col7:
    family_history = st.selectbox("Family History of Heart Problems?", options=["No", "Yes"])
    prev_heart_problems = st.selectbox("Previous Heart Problems?", options=["No", "Yes"])
    medication_use = st.selectbox("Currently on Medication?", options=["No", "Yes"])

with col8:
    country = st.text_input("Country", value="United States")
    continent = st.selectbox("Continent", options=["North America", "South America", "Europe", "Asia", "Africa", "Australia"])
    hemisphere = st.selectbox("Hemisphere", options=["Northern Hemisphere", "Southern Hemisphere"])

# Helper function to map Yes/No to 1/0
def map_binary(val):
    return 1 if val == "Yes" else 0

# 5. Prediction Logic
if st.button("Predict Heart Attack Risk", type="primary"):
    
    # Construct the dictionary exactly matching the notebook's X_raw columns
    input_data = {
        'Age': age,
        'Sex': sex,
        'Cholesterol': cholesterol,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Diabetes': map_binary(diabetes),
        'Family History': map_binary(family_history),
        'Smoking': map_binary(smoking),
        'Obesity': map_binary(obesity),
        'Alcohol Consumption': map_binary(alcohol),
        'Exercise Hours Per Week': exercise_hours,
        'Diet': diet,
        'Previous Heart Problems': map_binary(prev_heart_problems),
        'Medication Use': map_binary(medication_use),
        'Stress Level': stress_level,
        'Sedentary Hours Per Day': sedentary_hours,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Physical Activity Days Per Week': physical_activity_days,
        'Sleep Hours Per Day': sleep_hours,
        'Country': country,
        'Continent': continent,
        'Hemisphere': hemisphere
    }
    
    # Convert to a single-row DataFrame
    input_df = pd.DataFrame([input_data])
    
    try:
        # 6. Display Results
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        risk_probability = probabilities[1] * 100
        
        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ High Risk Detected")
            st.write(f"The model predicts a high likelihood of a heart attack.")
        else:
            st.success(f"### ✅ Low Risk Detected")
            st.write(f"The model predicts a low likelihood of a heart attack.")
            
        st.info(f"**Confidence / Probability of Risk:** {risk_probability:.2f}%")
        
    except Exception as e:
        st.error(f"An error occurred during prediction. Please verify your inputs (e.g., Blood Pressure format). Error details: {e}")