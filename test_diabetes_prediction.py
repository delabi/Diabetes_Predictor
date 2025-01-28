import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TestDiabetesPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset
        cls.data = pd.read_csv('dataset/diabetes.csv')
        # Preprocess the data
        cls.X = cls.data.drop(columns='Outcome')
        cls.y = cls.data['Outcome']
        # Split the data into training and testing sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)
        # Standardize the features
        cls.scaler = StandardScaler()
        cls.X_train = cls.scaler.fit_transform(cls.X_train)
        cls.X_test = cls.scaler.transform(cls.X_test)
        # Train a logistic regression model
        cls.model = LogisticRegression()
        cls.model.fit(cls.X_train, cls.y_train)

    def test_model_accuracy(self):
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.7, "Accuracy should be at least 70%")

    def test_model_prediction(self):
        # Make a single prediction
        sample = self.X_test[0].reshape(1, -1)
        prediction = self.model.predict(sample)
        self.assertIn(prediction[0], [0, 1], "Prediction should be either 0 or 1")

if __name__ == '__main__':
    unittest.main()
