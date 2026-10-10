from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# 1. PATHS AND APP CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "plateiq-ml"

MODEL_PATH = ML_DIR / "plateiq_demand_model.joblib"
MEAL_PATH = ML_DIR / "meal_info.csv"
CENTER_PATH = ML_DIR / "fulfilment_center_info.csv"

app = FastAPI(
    title="PlateIQ Demand Forecast API",
    description="ML-powered weekly meal demand forecasting",
    version="1.0.0",
)
app.include_router(analytics_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. LOAD MODEL AND DATA
# =========================================================

model = None
meals = None
centers = None
model_load_error = None

# Stores the latest successful batch in memory.
# It resets whenever the backend restarts.
last_batch_forecast = None

try:
    model = joblib.load(MODEL_PATH)
    meals = pd.read_csv(MEAL_PATH)
    centers = pd.read_csv(CENTER_PATH)

    print("PlateIQ model loaded successfully.")
    print(f"Loaded {len(meals)} meals.")
    print(f"Loaded {len(centers)} fulfillment centers.")

except Exception as exc:
    model_load_error = str(exc)
    print(f"Failed to load PlateIQ model or data: {exc}")


# =========================================================
# 3. REQUEST MODELS
# =========================================================

class PredictionRequest(BaseModel):
    week: int = Field(ge=1)
    meal_id: int
    center_id: int
    checkout_price: float = Field(gt=0)
    base_price: float = Field(gt=0)
    emailer_for_promotion: int = Field(default=0, ge=0, le=1)
    homepage_featured: int = Field(default=0, ge=0, le=1)


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(
        min_length=1,
        max_length=100,
    )
class PreparationPlanRequest(BaseModel):
    projected_demand: int = Field(ge=0)
    recommended_prep: int = Field(ge=0)
    demand_change: float = Field(default=0)  


# =========================================================
# 4. HELPER FUNCTION: PREDICT ONE MEAL
# =========================================================

def generate_prediction(request: PredictionRequest) -> dict:
    if model is None or meals is None or centers is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model or metadata is unavailable: {model_load_error}",
        )

    meal_rows = meals.loc[meals["meal_id"] == request.meal_id]
    center_rows = centers.loc[centers["center_id"] == request.center_id]

    if meal_rows.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown meal_id: {request.meal_id}",
        )

    if center_rows.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown center_id: {request.center_id}",
        )

    meal = meal_rows.iloc[0]
    center = center_rows.iloc[0]

    # These feature names and values must match train_model.py.
    features = pd.DataFrame([{
        "center_id": request.center_id,
        "meal_id": request.meal_id,
        "city_code": center["city_code"],
        "region_code": center["region_code"],
        "center_type": center["center_type"],
        "category": meal["category"],
        "cuisine": meal["cuisine"],
        "week": request.week,
        "checkout_price": request.checkout_price,
        "base_price": request.base_price,
        "emailer_for_promotion": request.emailer_for_promotion,
        "homepage_featured": request.homepage_featured,
        "op_area": center["op_area"],
    }])

    try:
        prediction_log = float(model.predict(features)[0])

        # The model was trained on log1p(num_orders).
        predicted_orders = max(
            0,
            int(round(np.expm1(prediction_log))),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Check model features: {exc}",
        )

    return {
        "predicted_orders": predicted_orders,
        "week": request.week,
        "meal_id": request.meal_id,
        "center_id": request.center_id,
        "unit": "orders",
        "forecast_type": "weekly",
    }


# =========================================================
# 5. ROOT ENDPOINT
# =========================================================

@app.get("/")
def home():
    return {
        "message": "PlateIQ Demand Forecast API is running",
        "docs": "/docs",
        "endpoints": [
            "GET /health",
            "POST /predict",
            "POST /forecast/batch",
            "GET /forecast/summary",
            "GET /meals",
            "GET /centers",
        ],
    }


# =========================================================
# 6. HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    ready = (
        model is not None
        and meals is not None
        and centers is not None
    )

    return {
        "status": "healthy" if ready else "model_not_ready",
        "model_loaded": model is not None,
        "metadata_loaded": meals is not None and centers is not None,
        "error": model_load_error,
    }


# =========================================================
# 7. SINGLE MEAL PREDICTION
# =========================================================

@app.post("/predict")
def predict_demand(request: PredictionRequest):
    return generate_prediction(request)


# =========================================================
# 8. BATCH FORECASTING
# =========================================================

@app.post("/forecast/batch")
def forecast_batch(request: BatchPredictionRequest):
    global last_batch_forecast

    results = []

    # Generate each prediction.
    for item in request.predictions:
        result = generate_prediction(item)
        results.append(result)

    total_orders = sum(
        item["predicted_orders"] for item in results
    )

    response = {
        "forecast_type": "weekly",
        "number_of_meals": len(results),
        "total_predicted_orders": total_orders,
        "predictions": results,
    }

    # Save only after every prediction succeeds.
    last_batch_forecast = response

    return response


# =========================================================
# 9. FORECAST SUMMARY
# =========================================================

@app.get("/forecast/summary")
def forecast_summary():
    if last_batch_forecast is None:
        raise HTTPException(
            status_code=404,
            detail="No batch forecast found. Run POST /forecast/batch first.",
        )

    predictions = last_batch_forecast["predictions"]

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail="The latest batch contains no predictions.",
        )

    total = last_batch_forecast["total_predicted_orders"]
    count = len(predictions)

    highest_demand = max(
        predictions,
        key=lambda item: item["predicted_orders"],
    )

    return {
        "forecast_type": "weekly",
        "number_of_meals": count,
        "total_predicted_orders": total,
        "average_predicted_orders_per_meal": round(
            total / count, 2
        ),
        "highest_demand_meal": {
            "meal_id": highest_demand["meal_id"],
            "center_id": highest_demand["center_id"],
            "predicted_orders": highest_demand["predicted_orders"],
        },
        "summary_source": "most_recent_batch_forecast",
    }


# =========================================================
# 10. MEAL LIST
# =========================================================

@app.get("/meals")
def get_meals():
    if meals is None:
        raise HTTPException(
            status_code=503,
            detail="Meal data is not loaded.",
        )

    return meals.sort_values("meal_id").to_dict(orient="records")


# =========================================================
# 11. FULFILLMENT CENTER LIST
# =========================================================

@app.get("/centers")
def get_centers():
    if centers is None:
        raise HTTPException(
            status_code=503,
            detail="Center data is not loaded.",
        )

    return centers.sort_values("center_id").to_dict(orient="records")
# =========================================================
# 12. APPLY PREPARATION PLAN
# =========================================================

last_applied_plan = None


@app.post("/preparation-plan/apply")
def apply_preparation_plan(request: PreparationPlanRequest):
    global last_applied_plan

    last_applied_plan = {
        "projected_demand": request.projected_demand,
        "recommended_prep": request.recommended_prep,
        "demand_change": request.demand_change,
    }

    return {
        "success": True,
        "message": "Preparation plan applied successfully",
        "plan": last_applied_plan,
    }