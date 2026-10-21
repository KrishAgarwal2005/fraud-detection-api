"""REST API that scores card transactions for fraud and stores every prediction.

Run from the project folder (after training):
    uvicorn fraud.api:app --reload
Then open http://127.0.0.1:8000/docs to try it.

create_app() takes the model path and database path as arguments, so tests can
pass a small test model and a temporary database instead of the real ones.
"""

import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from fraud import db
from fraud.features import add_features


class Transaction(BaseModel):
    """One card transaction, in the same shape as the Kaggle dataset."""
    time: float = Field(ge=0, description="Seconds since the first transaction")
    amount: float = Field(ge=0, description="Transaction amount")
    v: List[float] = Field(min_length=28, max_length=28,
                           description="The 28 anonymised features V1..V28, in order")


class PredictionOut(BaseModel):
    prediction_id: int
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str


def create_app(model_path: str, db_path: str) -> FastAPI:
    bundle = joblib.load(model_path)          # load once at startup, not per request
    model = bundle["model"]
    threshold = bundle["threshold"]
    version = f'{bundle["model_name"]} / {bundle["version"]}'
    db.init_db(db_path)

    app = FastAPI(title="Fraud Detection API", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": version, "threshold": threshold}

    @app.post("/predict", response_model=PredictionOut)
    def predict(tx: Transaction):
        row = {"Time": tx.time, "Amount": tx.amount}
        row.update({f"V{i}": value for i, value in enumerate(tx.v, start=1)})
        X = add_features(pd.DataFrame([row]))          # same features as training
        probability = float(model.predict_proba(X)[0, 1])
        is_fraud = probability >= threshold
        new_id = db.save_prediction(db_path, tx.amount, int(X["hour"].iloc[0]),
                                    probability, is_fraud, threshold, version)
        return PredictionOut(prediction_id=new_id, fraud_probability=round(probability, 6),
                             is_fraud=is_fraud, threshold=threshold, model_version=version)

    @app.get("/predictions")
    def list_predictions(limit: int = Query(20, ge=1, le=500), only_fraud: bool = False):
        return db.recent_predictions(db_path, limit, only_fraud)

    @app.get("/predictions/{prediction_id}")
    def get_prediction(prediction_id: int):
        found = db.get_prediction(db_path, prediction_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return found

    @app.get("/stats")
    def get_stats():
        return db.stats(db_path)

    return app


# The app uvicorn runs. Paths can be changed with environment variables.
if os.path.exists(os.getenv("MODEL_PATH", "models/model.joblib")):
    app = create_app(os.getenv("MODEL_PATH", "models/model.joblib"),
                     os.getenv("DB_PATH", "predictions.db"))
