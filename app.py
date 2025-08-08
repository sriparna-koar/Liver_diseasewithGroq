
import streamlit as st
import pandas as pd
import pickle
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- File Loading Helper Functions ---
def load_file(filepath, description):
    """Loads a file and handles FileNotFoundError."""
    try:
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            st.error(f"Unsupported file type for {filepath}")
            st.stop()
    except FileNotFoundError:
        st.error(f"{description} '{filepath}' not found. Please ensure it's in the same directory as the script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {description} from '{filepath}': {e}")
        st.stop()

# --- Load Model, Scaler, and Dataset ---
# Assuming 'Indian Liver Patient Dataset (ILPD).csv', 'liver_disease_model.pkl', and 'scaler.pkl' are available
# For demonstration, we'll create dummy files if they don't exist for the purpose of running this code.
# In a real scenario, these files would be pre-trained and saved.

# Dummy data and model/scaler creation for demonstration if files are missing
if not os.path.exists('Indian Liver Patient Dataset (ILPD).csv'):
    dummy_data = {
        'age': [45, 60, 30, 55, 70],
        'gender': [1, 0, 1, 0, 1],
        'tot_bilirubin': [1.0, 2.5, 0.8, 3.2, 1.1],
        'direct_bilirubin': [0.3, 1.2, 0.2, 1.5, 0.4],
        'tot_proteins': [7.0, 5.5, 6.8, 6.0, 7.2],
        'albumin': [3.5, 2.8, 4.0, 3.0, 3.8],
        'ag_ratio': [1.0, 0.8, 1.2, 0.9, 1.1],
        'sgpt': [40, 120, 25, 150, 30],
        'sgot': [50, 180, 30, 200, 45],
        'alkphos': [100, 300, 80, 400, 120],
        'target': [0, 1, 0, 1, 0] # 0: Not liver patient, 1: Liver patient
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv('Indian Liver Patient Dataset (ILPD).csv', index=False)
    st.info("Created a dummy 'Indian Liver Patient Dataset (ILPD).csv' for demonstration.")
else:
    df = load_file('Indian Liver Patient Dataset (ILPD).csv', "Dataset")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if not os.path.exists('liver_disease_model.pkl') or not os.path.exists('scaler.pkl'):
    st.info("Creating dummy model and scaler for demonstration.")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    with open('liver_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
else:
    model = load_file('liver_disease_model.pkl', "Trained Model")
    scaler = load_file('scaler.pkl', "Scaler")


# Compute min and max values for each column from the loaded dataframe
if df is not None:
    min_max_values = df.describe().to_dict()
else:
    st.error("Dataframe could not be loaded. Cannot compute min/max values.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Liver Disease Prediction", page_icon="🔬", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
        padding: 10px;
        border-radius: 5px;
    }
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
        padding: 10px;
        border-radius: 5px;
    }
    .recommendation-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        background-color: #e8f5e9;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: black; /* Make text color black */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Liver Disease Prediction 🧐")
st.markdown("---")
st.markdown("Enter the patient's details in the sidebar to predict the likelihood of liver disease and receive AI-powered general recommendations.")

st.sidebar.header("Enter Patient Details 📋")

# User Input
# Using .get() with default values to handle potential missing keys or NaN from describe()
age = st.sidebar.slider("Age", min_value=int(min_max_values.get('age', {}).get('min', 1)), max_value=int(min_max_values.get('age', {}).get('max', 100)), value=int(min_max_values.get('age', {}).get('mean', 45)))
gender = st.sidebar.radio("Gender", ["Male", "Female"])
gender_encoded = 1 if gender == "Male" else 0 # Encode gender for the model

tot_bilirubin = st.sidebar.number_input("Total Bilirubin (mg/dL)", min_value=float(min_max_values.get('tot_bilirubin', {}).get('min', 0.1)), max_value=float(min_max_values.get('tot_bilirubin', {}).get('max', 50.0)), value=float(min_max_values.get('tot_bilirubin', {}).get('mean', 1.5)), format="%.2f")
direct_bilirubin = st.sidebar.number_input("Direct Bilirubin (mg/dL)", min_value=float(min_max_values.get('direct_bilirubin', {}).get('min', 0.0)), max_value=float(min_max_values.get('direct_bilirubin', {}).get('max', 25.0)), value=float(min_max_values.get('direct_bilirubin', {}).get('mean', 0.5)), format="%.2f")
tot_proteins = st.sidebar.number_input("Total Proteins (g/dL)", min_value=float(min_max_values.get('tot_proteins', {}).get('min', 0.0)), max_value=float(min_max_values.get('tot_proteins', {}).get('max', 10.0)), value=float(min_max_values.get('tot_proteins', {}).get('mean', 6.5)), format="%.2f")
albumin = st.sidebar.number_input("Albumin (g/dL)", min_value=float(min_max_values.get('albumin', {}).get('min', 0.0)), max_value=float(min_max_values.get('albumin', {}).get('max', 6.0)), value=float(min_max_values.get('albumin', {}).get('mean', 3.5)), format="%.2f")

# Handle potential NaN for ag_ratio min/max/mean gracefully
ag_ratio_min = float(min_max_values.get('ag_ratio', {}).get('min', 0.0))
ag_ratio_max = float(min_max_values.get('ag_ratio', {}).get('max', 5.0))
ag_ratio_mean = float(min_max_values.get('ag_ratio', {}).get('mean', 1.0))
ag_ratio = st.sidebar.number_input("Albumin and Globulin Ratio (A/G Ratio)", min_value=ag_ratio_min, max_value=ag_ratio_max, value=ag_ratio_mean, format="%.2f")

sgpt = st.sidebar.number_input("SGPT (ALT) (IU/L)", min_value=float(min_max_values.get('sgpt', {}).get('min', 0.0)), max_value=float(min_max_values.get('sgpt', {}).get('max', 2000.0)), value=float(min_max_values.get('sgpt', {}).get('mean', 50.0)), format="%.0f")
sgot = st.sidebar.number_input("SGOT (AST) (IU/L)", min_value=float(min_max_values.get('sgot', {}).get('min', 0.0)), max_value=float(min_max_values.get('sgot', {}).get('max', 2000.0)), value=float(min_max_values.get('sgot', {}).get('mean', 70.0)), format="%.0f")
alkphos = st.sidebar.number_input("Alkaline Phosphatase (ALP) (IU/L)", min_value=float(min_max_values.get('alkphos', {}).get('min', 0.0)), max_value=float(min_max_values.get('alkphos', {}).get('max', 1500.0)), value=float(min_max_values.get('alkphos', {}).get('mean', 200.0)), format="%.0f")


# Convert to DataFrame
patient_data = pd.DataFrame([{
    'age': age,
    'gender': gender_encoded,
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
try:
    patient_scaled = scaler.transform(patient_data)
except Exception as e:
    st.error(f"Error scaling patient data: {e}. Please ensure all input fields are correctly filled.")
    st.stop()

# Initialize Groq AI and LangChain components
# IMPORTANT: In a real application, retrieve API key securely, e.g., from environment variables
# For demonstration purposes, hardcoding it here as per user's prompt.
groq_api_key = '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' 

llm = None
if not groq_api_key:
    st.sidebar.warning("🤖 AI recommendations are disabled. Please set the **GROQ_API_KEY** environment variable to enable this feature.")
    st.sidebar.info("You can get a free API key from [console.groq.com](https://console.groq.com/docs/api).")
else:
    try:
        llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        # Define LangChain prompt templates for different scenarios
        liver_patient_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant providing general, non-medical health and lifestyle advice based on liver disease prediction. Always encourage professional medical consultation for diagnosis and treatment. Do not provide specific medical advice. Be empathetic and clear."),
            ("user", "A patient has been predicted to be a **Liver Patient** with a probability of {probability:.2f}. What general advice should be given regarding immediate next steps, lifestyle changes, and dietary considerations? Focus on general, non-medical advice that encourages professional consultation.")
        ])

        not_liver_patient_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant providing general health and lifestyle advice to maintain well-being. Always encourage regular health check-ups. Do not provide specific medical advice. Be encouraging and clear."),
            ("user", "A patient has been predicted to **NOT** be a liver patient with a probability of {probability:.2f}. What general health advice should be given to maintain liver health and overall well-being? Focus on preventive measures and healthy habits.")
        ])

        # Create LangChain chains
        output_parser = StrOutputParser()
        liver_patient_chain = liver_patient_prompt | llm | output_parser
        not_liver_patient_chain = not_liver_patient_prompt | llm | output_parser
    except Exception as e:
        st.error(f"Failed to initialize Groq AI or LangChain components: {e}")
        llm = None # Disable AI features if initialization fails


# Make prediction
if st.sidebar.button("Predict Liver Disease 🚀"):
    st.markdown("---")
    st.header("Prediction Results 📈")
    with st.spinner("Analyzing data and predicting..."):
        try:
            prediction = model.predict(patient_scaled)
            probability_liver_patient = model.predict_proba(patient_scaled)[0][1] # Probability of being a liver patient (class 1)
            probability_not_liver_patient = model.predict_proba(patient_scaled)[0][0] # Probability of not being a liver patient (class 0)

            if prediction[0] == 1:
                result_text = "Likely a **Liver Patient** ⚠️"
                st.error(f"### Prediction: {result_text}")
                # st.info(f"### Probability of being a Liver Patient: **{probability_liver_patient:.2f}**")
            else:
                result_text = "Likely **Not** a Liver Patient ✅"
                st.success(f"### Prediction: {result_text}")
                # st.info(f"### Probability of being a Liver Patient: **{probability_liver_patient:.2f}**")


            st.markdown("---")
            st.header("AI-Powered Recommendations 🤖💡")

            if llm:
                with st.spinner("Generating AI recommendations..."):
                    try:
                        if prediction[0] == 1: # Liver Patient
                            ai_response = liver_patient_chain.invoke({"probability": probability_liver_patient})
                        else: # Not a Liver Patient
                            ai_response = not_liver_patient_chain.invoke({"probability": probability_not_liver_patient})

                        st.markdown(f'<div class="recommendation-box">{ai_response}</div>', unsafe_allow_html=True)

                    except Exception as ai_e:
                        st.error(f"Error fetching AI recommendations: {ai_e}. Please check your internet connection or Groq API key.")
            else:
                st.info("AI recommendations are not available. Please set the GROQ_API_KEY environment variable.")

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

st.markdown("---")
st.markdown("""
### Important Disclaimer and Usage Information ⚠️

This **Liver Disease Prediction Tool** is designed for **informational and educational purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Here are some key points to remember:

* **Not a Medical Diagnosis:** The predictions generated by this tool are based on a machine learning model and are **not** a definitive medical diagnosis. Only a qualified healthcare professional can diagnose medical conditions.
* **Always Consult a Doctor:** If you have any health concerns, symptoms, or questions about your liver health, please **immediately consult with a licensed physician or healthcare provider**. Do not delay or disregard professional medical advice because of information obtained from this tool.
* **AI Recommendations are General:** Any AI-generated recommendations are for general guidance on healthy lifestyle and preventive measures. They are **not personalized medical advice** and should not be followed without professional medical consultation, especially if you have existing health conditions.
* **Data Accuracy:** The accuracy of the prediction depends on the accuracy and completeness of the input data. Incorrect or incomplete information may lead to inaccurate results.
* **No Doctor-Patient Relationship:** Use of this tool does not establish a doctor-patient relationship between you and the developers or the AI system.
* **Continuous Monitoring:** For individuals with existing liver conditions or high risk factors, regular medical check-ups and adherence to a doctor's treatment plan are crucial.

By using this application, you acknowledge and agree to this disclaimer. Your health is paramount; always seek the advice of a medical professional.
""")
