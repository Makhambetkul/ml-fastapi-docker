"""
FastAPI application that serves predictions of the best model from MLflow.

The model file (model.joblib) and its info (model_info.json) are created by train.py:
the best run is registered in the MLflow Model Registry and exported for the API.

Endpoints:
    GET  /            -> checks that the API is running
    GET  /model-info  -> registered model name, version, metrics
    POST /predict     -> takes 4 flower measurements and returns the predicted species
    GET  /docs        -> interactive Swagger documentation
"""

import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
INFO_PATH = BASE_DIR / "model_info.json"

app = FastAPI(
    title="Iris Classifier API",
    description="ML API: predicts the iris species from flower measurements. "
                "The model is tracked and registered with MLflow.",
    version="2.0.0",
)

# Load the trained model once when the app starts
if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Run `python train.py` first.")
model = joblib.load(MODEL_PATH)
model_info = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}


class IrisFeatures(BaseModel):
    """Input features. All values are in centimeters and must be positive."""

    sepal_length: float = Field(..., gt=0, le=20, description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, le=20, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, le=20, description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, le=20, description="Petal width (cm)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_name: str | None = None
    model_version: str | None = None


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info")
def get_model_info():
    if not model_info:
        raise HTTPException(status_code=404, detail="model_info.json not found")
    return model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    # Same feature order as in training
    x = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    try:
        prediction = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(cls): round(float(p), 4) for cls, p in zip(model.classes_, proba)
        },
        "model_name": model_info.get("model_name"),
        "model_version": model_info.get("model_version"),
    }
