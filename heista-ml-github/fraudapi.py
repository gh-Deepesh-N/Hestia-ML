import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load trained models
with open("new_random_forest_fraud_model.pkl", "rb") as f:
    rf_price_model = pickle.load(f)

with open("new_log_reg_text_fraud_model.pkl", "rb") as f:
    log_reg_text_model = pickle.load(f)

with open("new_text_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("new_label_enc_type.pkl", "rb") as f:
    label_enc_type = pickle.load(f)

with open("new_label_enc_loc.pkl", "rb") as f:
    label_enc_loc = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract property details
    property_type = label_enc_type.transform([data["PropertyType"]])[0]
    location = label_enc_loc.transform([data["Location"]])[0]
    sqft = data["SquareFeet"]
    price = data["PropertyPrice"]
    price_per_sqft = price / sqft

    # Prepare numerical features
    X_num = np.array([[property_type, location, sqft, price, price_per_sqft]])
    price_fraud_prob = rf_price_model.predict_proba(X_num)[:, 1][0]

    # Prepare text features
    X_text = vectorizer.transform([data["Description"]])
    text_fraud_prob = log_reg_text_model.predict_proba(X_text)[:, 1][0]

    # Compute Final Fraud Score (Weighted Combination)
    fraud_score = (0.6 * price_fraud_prob) + (0.4 * text_fraud_prob)
    final_decision = "Fraudulent" if fraud_score > 0.5 else "Legit"

    return jsonify({
        "FraudScore": fraud_score,
        "FinalDecision": final_decision
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
