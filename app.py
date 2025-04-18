from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Load model
model = pickle.load(open("prediction_model.pkl", "rb"))

# Validate model
if not isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
    raise TypeError("Loaded model is not a valid RandomForestRegressor or DecisionTreeRegressor.")

# Load training columns
training_columns = list(pd.read_csv("encoded_feature_columns.csv").columns)

# Load APY dataset
apy_df = pd.read_csv("APY.csv", encoding="utf-8")
apy_df.rename(columns=lambda x: x.strip(), inplace=True)

# Extract dropdown options
states = sorted(apy_df['State'].dropna().unique().tolist())
districts = sorted(apy_df['District'].dropna().unique().tolist())
crops = sorted(apy_df['Crop'].dropna().unique().tolist())
seasons = sorted(apy_df['Season'].dropna().unique().tolist())

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the Crop Production Prediction API",
        "endpoints": {
            "/predict": "POST with JSON {state, district, crop, season, area, year}",
            "/options": "GET available dropdown values"
        }
    })

@app.route("/options", methods=["GET"])
def get_options():
    return jsonify({
        "states": states,
        "districts": districts,
        "crops": crops,
        "seasons": seasons
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        state = data.get("state")
        district = data.get("district")
        crop = data.get("crop")
        season = data.get("season")
        area = float(data.get("area"))
        year = int(data.get("year"))

        # Input validation
        if area <= 0:
            return jsonify({"error": "Area must be greater than zero."}), 400
        if year < 2000 or year > 2050:
            return jsonify({"error": "Year should be within a reasonable range."}), 400

        # Create input dataframe
        row = pd.DataFrame({
            'State': [state],
            'District': [district],
            'Crop_Year': [year],
            'Season': [season],
            'Crop': [crop],
            'Area': [area]
        })

        # Encode and align with training columns
        row_encoded = pd.get_dummies(row)
        for col in training_columns:
            if col not in row_encoded.columns:
                row_encoded[col] = 0
        row_encoded = row_encoded[training_columns]

        # Predict
        prediction = model.predict(row_encoded)[0]
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
