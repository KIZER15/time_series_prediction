from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Global caches
_model = None
_training_columns = None
_apy_df = None

# Google Drive File ID for the model file
FILE_ID = '1VMQYo9_QdWav5jyyU-DN8YpYfoZiUJRB'
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def download_model():
    """Download the model from Google Drive"""
    response = requests.get(DOWNLOAD_URL)
    if response.status_code == 200:
        with open("prediction_model.pkl", "wb") as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download the model from Google Drive.")

def get_model():
    """Lazy-load model"""
    global _model
    if _model is None:
        # Check if the model is already cached locally
        if not os.path.exists("prediction_model.pkl"):
            download_model()  # Download if not available locally
        with open("prediction_model.pkl", "rb") as f:
            _model = pickle.load(f)
        if not isinstance(_model, (RandomForestRegressor, DecisionTreeRegressor)):
            raise TypeError("Loaded model is not a valid RandomForestRegressor or DecisionTreeRegressor.")
    return _model

# Lazy-load training columns
def get_training_columns():
    global _training_columns
    if _training_columns is None:
        _training_columns = list(pd.read_csv("encoded_feature_columns.csv").columns)
    return _training_columns

# Lazy-load APY data
def get_apy_df():
    global _apy_df
    if _apy_df is None:
        _apy_df = pd.read_csv("APY.csv", encoding="utf-8")
        _apy_df.rename(columns=lambda x: x.strip(), inplace=True)
    return _apy_df

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
    df = get_apy_df()
    return jsonify({
        "states": sorted(df['State'].dropna().unique().tolist()),
        "districts": sorted(df['District'].dropna().unique().tolist()),
        "crops": sorted(df['Crop'].dropna().unique().tolist()),
        "seasons": sorted(df['Season'].dropna().unique().tolist())
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

        # Prepare input row
        row = pd.DataFrame({
            'State': [state],
            'District': [district],
            'Crop_Year': [year],
            'Season': [season],
            'Crop': [crop],
            'Area': [area]
        })

        # Encoding with pd.concat to avoid fragmentation
        training_columns = get_training_columns()
        row_encoded = pd.get_dummies(row)
        missing_cols = [col for col in training_columns if col not in row_encoded.columns]
        missing_df = pd.DataFrame(0, index=row_encoded.index, columns=missing_cols)
        row_encoded = pd.concat([row_encoded, missing_df], axis=1)
        row_encoded = row_encoded[training_columns]

        # Predict
        model = get_model()
        prediction = model.predict(row_encoded)[0]
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
