from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model 
lin_reg = joblib.load('lin_reg_model.pkl')
rf_reg = joblib.load('rf_reg_model.pkl')
gb_reg = joblib.load('gb_reg_model.pkl')

# Home Endpoint
@app.route('/')
def home():
    return "Boston House Price Prediction API Running!!"

#  Predict With One Model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()


# Convert data into numpy array
    features = np.array(data["features"]).reshape(1, -1)

# Default model selection to RF if not specified
    model_choice = data.get("model", "rf") 

# Make predictions based on model choice
    if model_choice == "lin":
        prediction = lin_reg.predict(features) [0]
        model_used = "Linear Regression"
    elif model_choice == "rf":
        prediction = rf_reg.predict(features) [0]
        model_used = "Random Forest"
    else:
        prediction = gb_reg.predict(features) [0]
        model_used = "Gradient Boosting"
                                      
    # Return Prediction in JSON format
    return jsonify({
        "model_used": model_used,
        "prediction": float(prediction)
    })

# Compare with all Models
@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)

    results = {
        "Linear Regression": float(lin_reg.predict(features)[0]),
        "Random Forest": float(rf_reg.predict(features)[0]),
        "Gradient Boosting": float(gb_reg.predict(features)[0])
    }

    # Return Prediction in JSON FORMAT
    return jsonify({
        "All Model Predictions": results
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
