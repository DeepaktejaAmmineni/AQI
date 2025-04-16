from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import pickle
import os

# Define app
app = FastAPI()

# Load only city_day model at startup
with open("city_day_rf_model.pkl", "rb") as f:
    city_model = pickle.load(f)

# Pydantic model for input
class AQIInput(BaseModel):
    City: int
    PM2_5: float
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float
    Benzene: float
    Toluene: float
    Xylene: float
    AQI_Bucket: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Air Quality Prediction API!"}

# Prediction endpoint: city_day
@app.post("/predict_city_day/")
def predict_city_day(data: AQIInput):
    try:
        features = np.array([[
            data.City, data.PM2_5, data.PM10, data.NO, data.NO2, data.NOx,
            data.NH3, data.CO, data.SO2, data.O3,
            data.Benzene, data.Toluene, data.Xylene, data.AQI_Bucket
        ]])
        prediction = city_model.predict(features)
        return {"predicted_AQI": round(float(prediction[0]), 2)}
    except Exception as e:
        return {"error": str(e)}

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html from static folder
@app.get("/web", response_class=FileResponse)
def get_webpage():
    file_path = os.path.join("static", "index.html")
    return FileResponse(file_path)
