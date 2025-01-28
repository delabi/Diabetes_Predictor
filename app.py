import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Diabetes Prediction App')

# Collect user input
st.write("""
### Input Parameters
Please provide the following information:
- **Pregnancies**: Number of times pregnant (0-20)
- **Glucose**: Plasma glucose concentration (0-200)
- **Blood Pressure**: Diastolic blood pressure (mm Hg) (0-150)
- **Skin Thickness**: Triceps skin fold thickness (mm) (0-100)
- **Insulin**: 2-Hour serum insulin (mu U/ml) (0-900)
- **BMI**: Body mass index (weight in kg/(height in m)^2) (0.0-70.0)
- **Diabetes Pedigree Function**: Diabetes pedigree function (0.0-3.0)
- **Age**: Age (years) (0-120)
""")

pregnancies = st.number_input('Pregnancies (0-20)', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose (0-200)', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure (0-150)', min_value=0, max_value=150, value=0)
skin_thickness = st.number_input('Skin Thickness (0-100)', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin (0-900)', min_value=0, max_value=900, value=0)
bmi = st.number_input('BMI (0.0-70.0)', min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input('Diabetes Pedigree Function (0.0-3.0)', min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input('Age (0-120)', min_value=0, max_value=120, value=0)

st.write("### Important Features")
st.image("important_features.png", caption="Important Features in the Model")

# Predict diabetes
if st.button('Predict'):
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[0][1]
    if prediction[0] == 1:
        st.write(f'The model predicts that you have diabetes with a probability of {probability:.2f}.')
    else:
        st.write(f'The model predicts that you do not have diabetes with a probability of {probability:.2f}.')
