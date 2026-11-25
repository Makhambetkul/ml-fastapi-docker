from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(
    title="Iris Classifier API",
    description="Simple ML API: predicts the iris species from flower measurements.",
    version="1.0.0",
)

if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Run `python train.py` first.")
model = joblib.load(MODEL_PATH)


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


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
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
    }
