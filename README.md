# Diabetes Prediction Project

## Overview
This project aims to predict the likelihood of diabetes in patients using machine learning models. The project includes data preprocessing, model training, evaluation, and feature importance analysis.

## Project Structure
- `app.py`: Main application script.
- `diabetes_model.pkl`: Trained machine learning model.
- `diabetes_prediction.py`: Script for making predictions using the trained model.
- `generate_feature_importance.py`: Script to generate feature importance plot.
- `important_features.png`: Image showing the important features.
- `test_diabetes_prediction.py`: Unit tests for the prediction script.
- `dataset/diabetes.csv`: Dataset used for training and evaluation.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/delabi/Diabetes_Predictor.git
## Usage
1. To run the main application:
   ```bash
   python app.py
   ```

2. To run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. To generate feature importance plot:
   ```bash
   python generate_feature_importance.py
   ```

4. To run the unit tests:
   ```bash
   python -m unittest test_diabetes_prediction.py
   ```
