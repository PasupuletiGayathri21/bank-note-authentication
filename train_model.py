import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
# You can download this dataset from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
data = pd.read_csv('data_banknote_authentication.txt', header=None, 
                   names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

# Split features and target
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Print the model's accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'banknote_auth_model.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')

print("Model saved as 'banknote_auth_model.joblib'")
print("Scaler saved as 'feature_scaler.joblib'")
