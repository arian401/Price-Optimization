"""
FastAPI endpoint for *price‑sensitivity* prediction.
Expects 9 raw features (7 numeric + 2 categorical).  The loaded
model (`model.pkl`) already contains all preprocessing steps
(scaling + one‑hot encoding), so we simply convert the incoming
JSON to a DataFrame and call `predict` / `predict_proba`.
"""

from typing import Literal
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ──────────────────────── Load artefacts ────────────────────────
model = joblib.load("models/model.pkl")  # full sklearn pipeline

# ──────────────────────── FastAPI app ───────────────────────────
app = FastAPI(
    title="Price‑Sensitivity Prediction API",
    description="Predicts whether a customer will still buy after a price increase",
    version="1.0",
)

# ──────────────────────── Input schema ──────────────────────────
class CustomerData(BaseModel):
    total_spent:               float
    avg_order_value:           float
    avg_purchase_frequency:    float
    days_since_last_purchase:  float
    discount_behavior:         float
    loyalty_program_member:    int                # 0 = No, 1 = Yes
    days_in_advance:           int
    flight_type:               Literal["domestic", "international"]
    cabin_class:               Literal["economy", "business"]

# ──────────────────────── Prediction endpoint ───────────────────
@app.post("/predict/")
def predict_price_sensitivity(data: CustomerData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Predict (pipeline handles preprocessing internally)
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0, 1])

        return {
            "will_buy_after_price_increase": pred,
            "probability": round(proba, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
