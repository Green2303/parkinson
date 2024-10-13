import pandas as pd
from sklearn.decomposition import PCA
import joblib
import numpy as np

# Load your dataset
data = pd.read_csv('parkinsons.csv')  # Adjust the path as needed

# Drop the 'name' column and any other non-numeric columns
data = data.drop(columns=['name'])

# Check data types
print(data.dtypes)

# Ensure all feature columns are numeric
X = data.drop(columns=['status'])  # 'status' is the target variable
y = data['status']

# Check for non-numeric values in features
if not np.issubdtype(X.values.dtype, np.number):
    print("Non-numeric values found in the feature set.")

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Adjust the variance retention as needed

# Fit PCA only if X is numeric
if X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]:
    X_pca = pca.fit_transform(X)
    # Save the PCA model
    joblib.dump(pca, 'pca_model.pkl')
    print("PCA model saved successfully.")
else:
    print("Error: Feature set contains non-numeric values.")
