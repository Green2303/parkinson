from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return "Welcome to the Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Convert JSON data to a NumPy array
    input_data = np.array([
        data["MDVP:Fo(Hz)"],
        data["MDVP:Fhi(Hz)"],
        data["MDVP:Flo(Hz)"],
        data["MDVP:Jitter(%)"],
        data["MDVP:Jitter(Abs)"],
        data["MDVP:RAP"],
        data["MDVP:PPQ"],
        data["Jitter:DDP"],
        data["MDVP:Shimmer"],
        data["MDVP:Shimmer(dB)"],
        data["Shimmer:APQ3"],
        data["Shimmer:APQ5"],
        data["MDVP:APQ"],
        data["Shimmer:DDA"],
        data["NHR"],
        data["HNR"],
        data["RPDE"],
        data["DFA"],
        data["spread1"],
        data["spread2"],
        data["D2"],
        data["PPE"]
    ]).reshape(1, -1)  # Reshape for a single sample

    # Make a prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
