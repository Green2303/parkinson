import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv('parkinsons.csv')

# Check the data structure (optional)
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Get info about the dataset, including datatypes

# Features and target variable
X = data.drop(columns=['name', 'status'])  # Exclude non-numeric columns and the target variable
y = data['status']   # Target variable (the last column)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')

# Optionally, you can evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Optionally, save the model accuracy to a text file
with open('model_accuracy.txt', 'w') as f:
    f.write(f'Model Accuracy: {accuracy * 100:.2f}%\n')
