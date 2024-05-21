import streamlit as st
import pandas as pd
import numpy as np
from scikit-learn.neighbors import KNeighborsClassifier


#Load the K-Nearest Neighbour Classifier model
def load_model():
   try:
       model = KNeighborsClassifier(n_neighbors=5)
       #Training Data  
       X_train = np.random.rand(100, 13)
       Y_train = np.random.randint(2, size=100)
       model.fit(X_train, Y_train)
       return model
   except Exception as e:
       st.error(f"Error loading model: {e}")
       return None
       
# Load the model
model = load_model()

# Streamlit application
st.title('Heart Disease Prediction')
st.write("Welcome to the Heart Disease Risk Prediction app!\n "
        "Please enter the patient's details in the sidebar to get the patient's heart disease prediction.")

# Sidebar for data capturing
st.sidebar.header('Patient Details')
st.sidebar.write("Please fill in the following patient details: ")
def user_input_features():
   try:
       age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=50, help="Patient's age in years.")
       sex = st.sidebar.selectbox('Sex', ('Male', 'Female'), help="Select the patient's sex.")
       cp = st.sidebar.selectbox('Chest Pain Type', (1, 2, 3, 4), help="Type of chest pain experienced: 1 - Typical Angina, 2 - Atypical Angina, 3 - Non-angina pain, 4 - Asymptomatic.")
       trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120, help="Patient's resting blood pressure in mm Hg on admission to the hospital.")
       chol = st.sidebar.number_input('Serum Cholestoral (mg/dl)', min_value=100, max_value=600, value=200, help="Patient's serum cholestoral level in mg/dl.")
       fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1), help="Is the patient's fasting blood sugar > 120 mg/dl? (1 = True, 0 = False)")
       restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', (0, 1, 2), help="Results of the patient's resting electrocardiogram (Normal, Abnormal, Ventricula hypertrophy).")
       thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150, help="Maximum heart rate achieved by the patient.")
       exang = st.sidebar.selectbox('Exercise Induced Angina', (0, 1), help="Does the patient experience exercise-induced angina? (1 = yes, 0 = no)")
       oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=1.0, help="Enter the ST depression induced by exercise relative to rest.")
       slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', (1, 2, 3), help="Slope of the peak exercise ST segment (1 - Upsloping, 2 - Flat, 3 - Downsloping).")
       ca = st.sidebar.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, value=0, help="Number of major vessels (0 - Mild, 1 - Moderate, 3 - Severe) colored by fluoroscopy.")
       thal = st.sidebar.selectbox('Thalassemia', (1, 2, 3), help="Patient's thalassemia level (1 - Normal, 2 - Fixed Defect, 3 - Reversable defect).")
       # Convert sex to binary
       sex = 1 if sex == 'Male' else 0
       data = {'age': age,
               'sex': sex,
               'cp': cp,
               'trestbps': trestbps,
               'chol': chol,
               'fbs': fbs,
               'restecg': restecg,
               'thalach': thalach,
               'exang': exang,
               'oldpeak': oldpeak,
               'slope': slope,
               'ca': ca,
               'thal': thal}
       features = pd.DataFrame(data, index=[0])
       return features
   except Exception as e:
       st.error(f"Error in user input features: {e}")
       return pd.DataFrame()
input_df = user_input_features()
st.subheader('Patient Input Features')
if not input_df.empty:
   st.write(input_df)
else:
   st.error("Please enter valid patient details to proceed.")
    
# Prediction
try:
   if model and not input_df.empty:
       prediction = model.predict(input_df)
       prediction_proba = model.predict_proba(input_df)
       st.subheader('Prediction')
       heart_disease = np.array(['No', 'Yes'])
       st.write(f"The model predicts that the patient {'has' if heart_disease[prediction][0] == 'Yes' else 'does not have'} heart disease.")
       st.subheader('Prediction Probability')
       st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")
       st.write(f"Probability of not having heart disease: {prediction_proba[0][0]:.2f}")
   else:
       st.error("Input data is not available for prediction.")
except NotFittedError as e:
   st.error(f"Model not fitted: {e}")
except ValueError as e:
   st.error(f"Value error in prediction: {e}")
except Exception as e:
   st.error(f"Unexpected error in prediction: {e}")
    
# Documentation
st.sidebar.header('Documentation')
st.sidebar.write("This application allows doctors to enter the details of patients to determine whether the patient likely suffers from heart disease. \nThis helps doctors decide whether to send the patient for further tests or treatment."
                "The model considers various features including age, sex, chest pain type, resting blood pressure, serum cholestoral, and more.")
