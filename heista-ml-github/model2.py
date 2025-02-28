import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(r"enter file path")

# Encode categorical variables
label_enc_type = LabelEncoder()
label_enc_loc = LabelEncoder()

df["PropertyType"] = label_enc_type.fit_transform(df["PropertyType"])
df["Location"] = label_enc_loc.fit_transform(df["Location"])

# Feature Engineering
df["PricePerSqFt"] = df["PropertyPrice"] / df["SquareFeet"]

# Select Features and Labels
X = df[["PropertyType", "Location", "SquareFeet", "PropertyPrice", "PricePerSqFt"]]
y = df["Fraudulent"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier for Price-Based Fraud Detection
rf_price_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_price_model.fit(X_train, y_train)

# Evaluate Price Fraud Detection Model
y_pred_rf = rf_price_model.predict(X_test)
rf_fraud_classification_report = classification_report(y_test, y_pred_rf, target_names=["Legit", "Fraudulent"])
print("🔹 Price-Based Fraud Detection Model (Random Forest)\n", rf_fraud_classification_report)

# Train TF-IDF Vectorizer for Text-Based Fraud Detection
vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(df["Description"].fillna(""))

# Train Logistic Regression Model for Text Fraud Detection
log_reg_text_model = LogisticRegression()
log_reg_text_model.fit(X_text, y)

# Evaluate Text Fraud Detection Model
y_text_pred_log_reg = log_reg_text_model.predict(X_text)
log_reg_text_classification_report = classification_report(y, y_text_pred_log_reg, target_names=["Legit", "Fraudulent"])
print("🔹 Text-Based Fraud Detection Model (Logistic Regression)\n", log_reg_text_classification_report)

# Compute Fraud Probability Scores
price_fraud_prob = rf_price_model.predict_proba(X)[:, 1]  # Probability of fraud from price model
text_fraud_prob = log_reg_text_model.predict_proba(X_text)[:, 1]  # Probability of fraud from text model

# Compute Final Fraud Score (Weighted Combination)
df["FraudScore"] = (0.6 * price_fraud_prob) + (0.4 * text_fraud_prob)
df["FinalFraudDecision"] = df["FraudScore"].apply(lambda x: "Fraudulent" if x > 0.512 else "Legit")

# Save models for future use
with open("new_random_forest_fraud_model.pkl", "wb") as f:
    pickle.dump(rf_price_model, f)

with open("new_log_reg_text_fraud_model.pkl", "wb") as f:
    pickle.dump(log_reg_text_model, f)

with open("new_text_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("new_label_enc_type.pkl", "wb") as f:
    pickle.dump(label_enc_type, f)

with open("new_label_enc_loc.pkl", "wb") as f:
    pickle.dump(label_enc_loc, f)

# Save updated dataset
df.to_csv("processed_fraudulent_property_data.csv", index=False)
print("✅ ML Pipeline Completed: Fraud Detection & Analysis Results Saved.")
