from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model and scaler
try:
    model = joblib.load('model/one.pkl')  # Replace 'one.pkl' with your actual model filename
    scaler = joblib.load('model/two.pkl')  # Replace 'two.pkl' with your actual scaler filename
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs and ensure they are valid
        age = int(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        chest_pain = int(request.form.get('chest_pain', 0))
        resting_bp = int(request.form.get('resting_bp', 0))
        cholesterol = int(request.form.get('cholesterol', 0))
        fasting_bs = int(request.form.get('fasting_bs', 0))
        rest_ecg = int(request.form.get('rest_ecg', 0))
        max_hr = int(request.form.get('max_hr', 0))
        exercise_angina = int(request.form.get('exercise_angina', 0))
        oldpeak = float(request.form.get('oldpeak', 0.0))
        slope = int(request.form.get('slope', 0))
        ca = int(request.form.get('ca', 0))
        thal = int(request.form.get('thal', 0))

        # Create a numpy array for prediction
        input_data = np.array([[age, gender, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr,
                                exercise_angina, oldpeak, slope, ca, thal]])

        # Debugging: Print the raw input data
        print(f"Raw input data: {input_data}")

        # Scale input data
        scaled_data = scaler.transform(input_data)
        
        # Debugging: Print the scaled input data
        print(f"Scaled input data: {scaled_data}")

        # Get prediction
        prediction = model.predict(scaled_data)[0]
        
        # Debugging: Print the raw prediction
        print(f"Raw prediction: {prediction}")

        # Ensure the prediction is within the expected range
        if prediction not in range(5):  # Assuming your model outputs values from 0 to 4
            raise ValueError("Unexpected prediction value.")

        # Mapping of predictions to stages of heart disease
        outcome_map = {
            0: "No Disease",
            1: "Mild Disease",
            2: "Moderate Disease",
            3: "Severe Disease",
            4: "Critical Disease"
        }
        outcome = outcome_map.get(prediction, "Unknown outcome")

        return render_template('index.html', prediction_text=f"Heart Disease Stage: {outcome}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
