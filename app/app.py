from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("churn_pipeline.pkl")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: Customer):

    data = customer.model_dump()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }