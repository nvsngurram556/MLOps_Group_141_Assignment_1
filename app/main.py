from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Load model
model_path = os.path.join("app", "model", "model.joblib")
model = joblib.load(model_path)

# Input Schema
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {"message": "California Housing Price Prediction API"}

@app.post("/predict")
def predict(features: HousingFeatures):
    data = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population, features.AveOccup,
        features.Latitude, features.Longitude
    ]])
    prediction = model.predict(data)[0]
    return {"predicted_price": round(prediction, 2)}