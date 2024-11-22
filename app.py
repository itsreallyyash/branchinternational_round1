# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(title="Loan Default Prediction API")

# Load the trained SVM pipeline
try:
    model = joblib.load('svm_pipeline.joblib')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise e

# Define the input data model using Pydantic
class LoanData(BaseModel):
    age: float
    cash_incoming_30days: float
    cash_incoming_per_day: float
    distance_traveled: float
    mean_distance: float
    max_distance: float
    latitude: float
    longitude: float
    altitude: float
    application_hour: float
    application_dayofweek: float
    accuracy: float
    bearing: float
    gps_upload_delay: float
    total_distance: float

# Define the correct feature order as per model training
FEATURE_ORDER = [
    "age",
    "cash_incoming_30days",
    "cash_incoming_per_day",
    "distance_traveled",
    "mean_distance",
    "max_distance",
    "latitude",
    "longitude",
    "altitude",
    "application_hour",
    "application_dayofweek",
    "accuracy",
    "bearing",
    "gps_upload_delay",
    "total_distance"
]

@app.post("/predict")
def predict_default(data: LoanData):
    logging.info("Received prediction request.")
    try:
        # Convert input data to DataFrame
        input_dict = data.dict()
        logging.debug(f"Input Data Dictionary: {input_dict}")
        
        input_df = pd.DataFrame([input_dict])
        logging.debug(f"Input DataFrame before reordering:\n{input_df}")
        
        # Reorder the DataFrame columns to match the training order
        input_df = input_df[FEATURE_ORDER]
        logging.debug(f"Input DataFrame after reordering:\n{input_df}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of defaulting
        logging.info(f"Prediction: {prediction}, Probability of default: {prediction_proba:.4f}")
        
        # Interpret the prediction
        result = "Defaulted" if prediction == 1 else "Not Defaulted"
        
        return {
            "prediction": result,
            "probability_of_default": round(prediction_proba, 4)
        }
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
