import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data = pd.read_csv('dataset/diabetes.csv')
X = data.drop(columns='Outcome')

# Get feature importance
importance = model.coef_[0]
features = X.columns

# Create a DataFrame for visualization
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.savefig('important_features.png')
plt.show()
