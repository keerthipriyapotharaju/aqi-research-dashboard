
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import json
import joblib
import pandas as pd

from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AQI Research Dashboard API",
    description="Multi-horizon AQI forecasting and experimental "
                "5-class AQI classification API",
    version="1.1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS AND DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = BASE_DIR / "air_quality_daily_final_with_infrastructure.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["city", "date"]).reset_index(drop=True)

print("Dataset loaded successfully!")


# ============================================================
# LOAD NUMERICAL AQI MODELS
# ============================================================

model_1day = XGBRegressor()
model_1day.load_model(str(BASE_DIR / "xgb_1day.json"))

model_3day = XGBRegressor()
model_3day.load_model(str(BASE_DIR / "xgb_3day.json"))

model_7day = XGBRegressor()
model_7day.load_model(str(BASE_DIR / "xgb_7day.json"))

print("Numerical XGBoost models loaded successfully!")


# ============================================================
# LOAD INFRASTRUCTURE ENCODER
# ============================================================

infrastructure_encoder = joblib.load(
    BASE_DIR / "infrastructure_encoder.pkl"
)


# ============================================================
# LOAD EXPERIMENTAL 5-CLASS CLASSIFIER
# ============================================================

category_model = XGBClassifier()
category_model.load_model(
    str(MODEL_DIR / "aqi_category_5class.json")
)

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    category_feature_columns = json.load(f)

with open(MODEL_DIR / "classes.json", "r") as f:
    category_class_info = json.load(f)

with open(MODEL_DIR / "metadata.json", "r") as f:
    category_metadata = json.load(f)

# Exported mapping: category label -> integer class ID
id_to_category = {
    int(class_id): label
    for label, class_id
    in category_class_info["class_to_id"].items()
}

if category_model.n_features_in_ != len(category_feature_columns):
    raise RuntimeError(
        "Classifier feature mismatch: model expects "
        f"{category_model.n_features_in_} features but "
        f"feature_columns.json contains "
        f"{len(category_feature_columns)}."
    )

print("Experimental 5-class classifier loaded successfully!")
print("Classifier feature count:", len(category_feature_columns))
print("Classifier class mapping:", id_to_category)


# ============================================================
# SHARED FEATURE DEFINITIONS
# ============================================================

BASE_FEATURE_COLUMNS = [
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_7",
    "aqi_roll_3_mean",
    "aqi_roll_7_mean",
    "aqi_roll_14_mean",
    "month",
    "day_of_week",
    "day_of_year",
    "pm2_5_mean_lag_1",
    "pm2_5_mean_lag_3",
    "pm2_5_mean_lag_7",
    "pm2_5_max_lag_1",
    "pm2_5_max_lag_3",
    "pm2_5_max_lag_7",
    "pm10_mean_lag_1",
    "pm10_mean_lag_3",
    "pm10_mean_lag_7",
    "pm10_max_lag_1",
    "pm10_max_lag_3",
    "pm10_max_lag_7",
    "no2_mean_lag_1",
    "no2_mean_lag_3",
    "no2_mean_lag_7",
    "so2_mean_lag_1",
    "so2_mean_lag_3",
    "so2_mean_lag_7",
    "o3_8h_max_lag_1",
    "o3_8h_max_lag_3",
    "o3_8h_max_lag_7",
    "co_8h_max_lag_1",
    "co_8h_max_lag_3",
    "co_8h_max_lag_7",
    "dust_mean_lag_1",
    "dust_mean_lag_3",
    "dust_mean_lag_7",
    "aod_mean_lag_1",
    "aod_mean_lag_3",
    "aod_mean_lag_7",
]

# Ensure the classifier JSON has the expected feature names and order.
if category_feature_columns != BASE_FEATURE_COLUMNS:
    raise RuntimeError(
        "The classifier feature list differs from the backend's "
        "39-feature construction. Check feature_columns.json and "
        "the classifier training notebook before deployment."
    )

POLLUTANTS = [
    "pm2_5_mean",
    "pm2_5_max",
    "pm10_mean",
    "pm10_max",
    "no2_mean",
    "so2_mean",
    "o3_8h_max",
    "co_8h_max",
    "dust_mean",
    "aod_mean",
]


# ============================================================
# SHARED HELPERS
# ============================================================

def classify_infrastructure(station_count: int) -> str:
    if station_count <= 1:
        return "Low"
    elif station_count <= 3:
        return "Medium"
    return "High"


def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


def get_city_dataframe(city: str) -> pd.DataFrame:
    city_df = df[df["city"].str.lower() == city.lower()].copy()

    if city_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found."
        )

    return city_df.sort_values("date").reset_index(drop=True)


def add_prediction_features(city_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the same 39 lag, rolling and calendar features."""

    city_df = city_df.copy()

    city_df["aqi_lag_1"] = city_df["india_aqi"].shift(1)
    city_df["aqi_lag_3"] = city_df["india_aqi"].shift(3)
    city_df["aqi_lag_7"] = city_df["india_aqi"].shift(7)

    city_df["aqi_roll_3_mean"] = (
        city_df["india_aqi"].shift(1).rolling(3).mean()
    )
    city_df["aqi_roll_7_mean"] = (
        city_df["india_aqi"].shift(1).rolling(7).mean()
    )
    city_df["aqi_roll_14_mean"] = (
        city_df["india_aqi"].shift(1).rolling(14).mean()
    )

    city_df["month"] = city_df["date"].dt.month
    city_df["day_of_week"] = city_df["date"].dt.dayofweek
    city_df["day_of_year"] = city_df["date"].dt.dayofyear

    for col in POLLUTANTS:
        if col not in city_df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Required dataset column is missing: {col}"
            )

        for lag in (1, 3, 7):
            city_df[f"{col}_lag_{lag}"] = city_df[col].shift(lag)

    return city_df


def get_latest_feature_row(city: str):
    city_df = get_city_dataframe(city)
    featured_df = add_prediction_features(city_df)

    missing_columns = [
        col for col in BASE_FEATURE_COLUMNS
        if col not in featured_df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Missing prediction features: {missing_columns}"
        )

    valid_rows = featured_df.dropna(subset=BASE_FEATURE_COLUMNS)

    if valid_rows.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough historical data for prediction."
        )

    latest_row = valid_rows.iloc[-1]
    X = latest_row[BASE_FEATURE_COLUMNS].to_frame().T
    X = X.astype(float)

    return city_df, latest_row, X


# ============================================================
# HOME
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AQI Research Dashboard API",
        "status": "running",
        "version": "1.1.0",
        "numerical_forecasts": True,
        "experimental_category_classifier": True
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "numerical_models": "loaded",
        "category_classifier": "loaded"
    }


# ============================================================
# SUMMARY
# ============================================================

@app.get("/api/summary")
def get_summary():
    return {
        "total_cities": int(df["city"].nunique()),
        "total_records": int(len(df)),
        "date_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_end": df["date"].max().strftime("%Y-%m-%d"),
        "average_aqi": round(float(df["india_aqi"].mean()), 2),
        "maximum_aqi": int(df["india_aqi"].max())
    }


# ============================================================
# CITIES
# ============================================================

@app.get("/api/cities")
def get_cities():
    return {
        "cities": sorted(df["city"].dropna().unique().tolist())
    }


# ============================================================
# CITY DATA
# ============================================================

@app.get("/api/city/{city}")
def get_city_data(city: str):
    city_df = get_city_dataframe(city)

    station_count = int(
        city_df["number_of_monitoring_stations"].iloc[0]
    )
    infrastructure_level = classify_infrastructure(station_count)
    latest = city_df.iloc[-1]

    historical_data = (
        city_df[["date", "india_aqi"]]
        .rename(columns={"india_aqi": "aqi"})
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )

    records = (
        city_df[
            ["date", "india_aqi", "india_aqi_category", "dominant_pollutant"]
        ]
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )

    return {
        "city": city,
        "monitoring_stations": station_count,
        "infrastructure_level": infrastructure_level,
        "latest_aqi": int(latest["india_aqi"]),
        "latest_category": str(latest["india_aqi_category"]),
        "latest_available_date": latest["date"].strftime("%Y-%m-%d"),
        "historical_data": historical_data,
        "records": records
    }


# ============================================================
# NUMERICAL AQI FORECAST
# Existing endpoint: /api/predict/{city}?horizon=1|3|7
# ============================================================

@app.get("/api/predict/{city}")
def predict_aqi(city: str, horizon: int = 1):
    if horizon not in (1, 3, 7):
        raise HTTPException(
            status_code=400,
            detail="Horizon must be 1, 3, or 7 days."
        )

    city_df, latest_row, X_base = get_latest_feature_row(city)

    if horizon == 1:
        model = model_1day
    elif horizon == 3:
        model = model_3day
    else:
        model = model_7day

    station_count = int(latest_row["number_of_monitoring_stations"])
    infrastructure_level = classify_infrastructure(station_count)

    infra_encoded = infrastructure_encoder.transform(
        [[infrastructure_level]]
    )

    infra_df = pd.DataFrame(
        infra_encoded,
        columns=[
            "infrastructure_level_Low",
            "infrastructure_level_Medium"
        ]
    )

    X_final = pd.concat(
        [X_base.reset_index(drop=True), infra_df.reset_index(drop=True)],
        axis=1
    )

    prediction = float(model.predict(X_final)[0])

    latest_date = city_df["date"].max()
    prediction_date = latest_date + pd.Timedelta(days=horizon)

    return {
        "city": city,
        "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        "horizon_days": horizon,
        "predicted_aqi": round(prediction, 2),
        "category": get_aqi_category(prediction),
        "monitoring_stations": station_count,
        "infrastructure_level": infrastructure_level,
        "latest_available_date": latest_date.strftime("%Y-%m-%d")
    }


# ============================================================
# EXPERIMENTAL 5-CLASS AQI CATEGORY CLASSIFIER
# Endpoint: /api/predict-category/{city}
# ============================================================

@app.get("/api/predict-category/{city}")
def predict_aqi_category(city: str):
    city_df, latest_row, X = get_latest_feature_row(city)

    # Validate features immediately before inference.
    if list(X.columns) != category_feature_columns:
        raise HTTPException(
            status_code=500,
            detail="Classifier input feature names/order do not match "
                   "feature_columns.json."
        )

    predicted_id = int(category_model.predict(X)[0])

    if predicted_id not in id_to_category:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown predicted class ID: {predicted_id}"
        )

    predicted_category = id_to_category[predicted_id]

    probabilities = category_model.predict_proba(X)[0]
    probability_by_class = {}

    for class_id, probability in zip(category_model.classes_, probabilities):
        class_id = int(class_id)
        label = id_to_category.get(class_id, str(class_id))
        probability_by_class[label] = round(float(probability), 4)

    latest_date = city_df["date"].max()
    prediction_date = latest_date + pd.Timedelta(days=1)

    return {
        "city": city,
        "prediction_type": "experimental_next_day_category_classification",
        "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        "predicted_category": predicted_category,
        "predicted_class_id": predicted_id,
        "class_probabilities": probability_by_class,
        "latest_observed_aqi": round(float(latest_row["india_aqi"]), 2),
        "latest_available_date": latest_date.strftime("%Y-%m-%d"),
        "model_type": "XGBClassifier",
        "number_of_features": len(category_feature_columns),
        "note": (
            "Experimental category classification. "
            "This is separate from the numerical AQI regression forecast."
        )
    }
