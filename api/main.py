from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Load model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "student_performance_model.joblib"
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict whether a student is Good or At Risk.",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class StudentData(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "student_performance_model"
    }

@app.get("/model-info")
def model_info():
    return {
        "model": "Tuned XGBoost",
        "target": "student performance",
        "classes": {
            "0": "Good",
            "1": "At Risk"
        },
        "accuracy": 0.759494,
        "precision": 0.684211,
        "recall": 0.500000,
        "f1": 0.577778,
        "roc_auc": 0.743832
    }
# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(student: StudentData):

    student_df = pd.DataFrame(
        [student.model_dump()]
    )

    prediction = model.predict(student_df)[0]

    at_risk_probability = (
        model.predict_proba(student_df)[0][1]
    )

    label = (
        "At Risk"
        if prediction == 1
        else "Good"
    )

    return {
        "prediction": int(prediction),
        "label": label,
        "at_risk_probability": round(
            float(at_risk_probability),
            4
        )
    }