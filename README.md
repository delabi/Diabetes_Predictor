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
   git clone <repository_url>
   cd Diabetes
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. To run the main application:
   ```bash
   python app.py
   ```

2. To generate feature importance plot:
   ```bash
   python generate_feature_importance.py
   ```

3. To run the unit tests:
   ```bash
   python -m unittest test_diabetes_prediction.py
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.
