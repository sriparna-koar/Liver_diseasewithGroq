import streamlit as st
import pandas as pd
import pickle

# Load the dataset
df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')

# Compute min and max values for each column
min_max_values = df.describe().to_dict()

# Load the trained model
with open('liver_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Liver Disease Prediction")

st.sidebar.header("Enter Patient Details")

# User Input
age = st.sidebar.number_input("Age", min_value=int(min_max_values['age']['min']), max_value=int(min_max_values['age']['max']), value=int(min_max_values['age']['mean']))
gender = st.sidebar.radio("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0
tot_bilirubin = st.sidebar.number_input("Total Bilirubin", min_value=min_max_values['tot_bilirubin']['min'], max_value=min_max_values['tot_bilirubin']['max'], value=min_max_values['tot_bilirubin']['mean'])
direct_bilirubin = st.sidebar.number_input("Direct Bilirubin", min_value=min_max_values['direct_bilirubin']['min'], max_value=min_max_values['direct_bilirubin']['max'], value=min_max_values['direct_bilirubin']['mean'])
tot_proteins = st.sidebar.number_input("Total Proteins", min_value=min_max_values['tot_proteins']['min'], max_value=min_max_values['tot_proteins']['max'], value=min_max_values['tot_proteins']['mean'])
albumin = st.sidebar.number_input("Albumin", min_value=min_max_values['albumin']['min'], max_value=min_max_values['albumin']['max'], value=min_max_values['albumin']['mean'])
ag_ratio = st.sidebar.number_input("A/G Ratio", min_value=min_max_values['ag_ratio']['min'], max_value=min_max_values['ag_ratio']['max'], value=min_max_values['ag_ratio']['mean'])
sgpt = st.sidebar.number_input("SGPT", min_value=min_max_values['sgpt']['min'], max_value=min_max_values['sgpt']['max'], value=min_max_values['sgpt']['mean'])
sgot = st.sidebar.number_input("SGOT", min_value=min_max_values['sgot']['min'], max_value=min_max_values['sgot']['max'], value=min_max_values['sgot']['mean'])
alkphos = st.sidebar.number_input("Alkaline Phosphatase", min_value=min_max_values['alkphos']['min'], max_value=min_max_values['alkphos']['max'], value=min_max_values['alkphos']['mean'])

# Convert to DataFrame
patient_data = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'tot_bilirubin': tot_bilirubin,
    'direct_bilirubin': direct_bilirubin,
    'tot_proteins': tot_proteins,
    'albumin': albumin,
    'ag_ratio': ag_ratio,
    'sgpt': sgpt,
    'sgot': sgot,
    'alkphos': alkphos
}])

# Scale input data
patient_scaled = scaler.transform(patient_data)

# Make prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)[0][1]

    result = "Liver Patient" if prediction[0] == 1 else "Not a Liver Patient"
    st.write(f"### Prediction: {result}")
    st.write(f"### Probability: {probability:.2f}")
