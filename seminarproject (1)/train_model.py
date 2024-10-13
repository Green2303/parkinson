import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load your training data
data = pd.read_csv('parkinsons.csv')

# Drop the 'name' column as it's not needed for training
data = data.drop(columns=['name'])

# Define your feature matrix (X) and target vector (y)
X = data.drop(columns=['status'])  # Features
y = data['status']  # Target

# Check for any missing values (optional but recommended)
if X.isnull().any().any():
    print("Warning: There are missing values in the dataset.")
    # You can handle missing values here if needed, e.g., fill or drop

# Split the dataset into training and testing sets to evaluate performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction
# Retain 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the Random Forest model on the reduced feature set
model = RandomForestClassifier(random_state=42)
model.fit(X_train_pca, y_train)

# Evaluate the model on the test set to check for overfitting
y_pred_train = model.predict(X_train_pca)
y_pred_test = model.predict(X_test_pca)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Save the model
with open('model_pca.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model with PCA trained and saved successfully.")
