# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
data = pd.read_csv('dataset/diabetes.csv')

# Preprocess the data
# Assuming the target column is named 'Outcome' and features are the rest
X = data.drop(columns='Outcome')
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Example prediction
example_data = X_test[0].reshape(1, -1)
example_prediction = model.predict(example_data)
print(f'Example data for first test sample: {example_data}')
if example_prediction[0] == 1:
    print('Example prediction for first test sample: The person is predicted to have diabetes.')
else:
    print('Example prediction for first test sample: The person is predicted not to have diabetes.')

# Save the model as a pickle file
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)
