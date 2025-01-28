import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data = pd.read_csv('dataset/diabetes.csv')
X = data.drop(columns='Outcome')

# Streamlit app
st.title('Diabetes Prediction App')

# User input
st.sidebar.header('User Input Features')
def user_input_features():
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    age = st.sidebar.slider('Age', 21, 81, 29)
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.3725)
    data = {'Glucose': glucose,
            'BMI': bmi,
            'Age': age,
            'Pregnancies': pregnancies,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'DiabetesPedigreeFunction': dpf}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure the feature order matches the training data
df = input_df[X.columns]

# Predict
prediction = model.predict(df)

# Display prediction with probability
st.subheader('Prediction')
probability = model.predict_proba(df)[0][1]
st.write(f'Diabetic (Probability: {probability:.2f})' if prediction[0] == 1 else f'Not Diabetic (Probability: {probability:.2f})')

# Display feature importance
st.subheader('Feature Importance')
importance_df = pd.read_csv('important_features.csv').sort_values(by='Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))
