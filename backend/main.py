from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AQI Research Dashboard API",
    description="Backend API for multi-city AQI forecasting research",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "air_quality_daily_final_with_infrastructure.csv"

MODEL_1DAY_PATH = BASE_DIR / "xgb_1day.json"
MODEL_3DAY_PATH = BASE_DIR / "xgb_3day.json"
MODEL_7DAY_PATH = BASE_DIR / "xgb_7day.json"

ENCODER_PATH = BASE_DIR / "infrastructure_encoder.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["city", "date"]
).reset_index(drop=True)


# ============================================================
# LOAD XGBOOST MODELS
# ============================================================

model_1day = XGBRegressor()
model_1day.load_model(MODEL_1DAY_PATH)

model_3day = XGBRegressor()
model_3day.load_model(MODEL_3DAY_PATH)

model_7day = XGBRegressor()
model_7day.load_model(MODEL_7DAY_PATH)


# ============================================================
# LOAD INFRASTRUCTURE ENCODER
# ============================================================

infrastructure_encoder = joblib.load(ENCODER_PATH)


print("Dataset loaded successfully!")
print("XGBoost models loaded successfully!")
print("Infrastructure encoder loaded successfully!")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AQI Research Dashboard API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# SUMMARY
# ============================================================

@app.get("/api/summary")
def summary():

    return {
        "rows": len(df),
        "cities": int(df["city"].nunique()),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "avg_aqi": round(
            float(df["india_aqi"].mean()),
            2
        )
    }


# ============================================================
# CITY LIST
# ============================================================

@app.get("/api/cities")
def get_cities():

    cities = sorted(
        df["city"].unique().tolist()
    )

    return {
        "count": len(cities),
        "cities": cities
    }


# ============================================================
# CITY HISTORICAL DATA
# ============================================================

@app.get("/api/city/{city}")
def get_city_data(city: str):

    city_df = df[
        df["city"].str.lower() == city.lower()
    ].copy()

    if city_df.empty:
        return {
            "error": "City not found"
        }

    city_df = city_df.sort_values("date")

    station_count = int(
        city_df["number_of_monitoring_stations"].iloc[0]
    )

    # Infrastructure classification
    if station_count <= 1:
        infrastructure_level = "Low"

    elif station_count <= 3:
        infrastructure_level = "Medium"

    else:
        infrastructure_level = "High"

    return {

        "city": city,

        "monitoring_stations": station_count,

        "infrastructure_level": infrastructure_level,

        "records": (
            city_df[
                [
                    "date",
                    "india_aqi",
                    "india_aqi_category",
                    "dominant_pollutant"
                ]
            ]
            .assign(
                date=lambda x:
                x["date"].dt.strftime("%Y-%m-%d")
            )
            .to_dict(
                orient="records"
            )
        )
    }


# ============================================================
# AQI PREDICTION
# ============================================================

@app.get("/api/predict/{city}")
def predict_aqi(
    city: str,
    horizon: int = 1
):

    # --------------------------------------------------------
    # Validate horizon
    # --------------------------------------------------------

    if horizon not in [1, 3, 7]:

        return {
            "error": "Horizon must be 1, 3, or 7 days"
        }


    # --------------------------------------------------------
    # Get city data
    # --------------------------------------------------------

    city_df = df[
        df["city"].str.lower() == city.lower()
    ].copy()

    if city_df.empty:

        return {
            "error": "City not found"
        }


    city_df = city_df.sort_values(
        "date"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if horizon == 1:

        model = model_1day

    elif horizon == 3:

        model = model_3day

    else:

        model = model_7day


    # --------------------------------------------------------
    # AQI lag features
    # --------------------------------------------------------

    city_df["aqi_lag_1"] = (
        city_df["india_aqi"].shift(1)
    )

    city_df["aqi_lag_3"] = (
        city_df["india_aqi"].shift(3)
    )

    city_df["aqi_lag_7"] = (
        city_df["india_aqi"].shift(7)
    )


    # --------------------------------------------------------
    # Rolling AQI features
    # --------------------------------------------------------

    city_df["aqi_roll_3_mean"] = (
        city_df["india_aqi"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    city_df["aqi_roll_7_mean"] = (
        city_df["india_aqi"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    city_df["aqi_roll_14_mean"] = (
        city_df["india_aqi"]
        .shift(1)
        .rolling(14)
        .mean()
    )


    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    city_df["month"] = (
        city_df["date"].dt.month
    )

    city_df["day_of_week"] = (
        city_df["date"].dt.dayofweek
    )

    city_df["day_of_year"] = (
        city_df["date"].dt.dayofyear
    )


    # --------------------------------------------------------
    # Pollutant columns
    # --------------------------------------------------------

    pollutants = [

        "pm2_5_mean",
        "pm2_5_max",

        "pm10_mean",
        "pm10_max",

        "no2_mean",

        "so2_mean",

        "o3_8h_max",

        "co_8h_max",

        "dust_mean",

        "aod_mean"
    ]


    # --------------------------------------------------------
    # Pollutant lag features
    # --------------------------------------------------------

    for col in pollutants:

        city_df[f"{col}_lag_1"] = (
            city_df[col].shift(1)
        )

        city_df[f"{col}_lag_3"] = (
            city_df[col].shift(3)
        )

        city_df[f"{col}_lag_7"] = (
            city_df[col].shift(7)
        )


    # ========================================================
    # EXACT 39 BASE FEATURES
    # ========================================================

    feature_columns = [

        # AQI history
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_7",

        # Rolling AQI
        "aqi_roll_3_mean",
        "aqi_roll_7_mean",
        "aqi_roll_14_mean",

        # Calendar
        "month",
        "day_of_week",
        "day_of_year",

        # PM2.5
        "pm2_5_mean_lag_1",
        "pm2_5_mean_lag_3",
        "pm2_5_mean_lag_7",

        "pm2_5_max_lag_1",
        "pm2_5_max_lag_3",
        "pm2_5_max_lag_7",

        # PM10
        "pm10_mean_lag_1",
        "pm10_mean_lag_3",
        "pm10_mean_lag_7",

        "pm10_max_lag_1",
        "pm10_max_lag_3",
        "pm10_max_lag_7",

        # NO2
        "no2_mean_lag_1",
        "no2_mean_lag_3",
        "no2_mean_lag_7",

        # SO2
        "so2_mean_lag_1",
        "so2_mean_lag_3",
        "so2_mean_lag_7",

        # O3
        "o3_8h_max_lag_1",
        "o3_8h_max_lag_3",
        "o3_8h_max_lag_7",

        # CO
        "co_8h_max_lag_1",
        "co_8h_max_lag_3",
        "co_8h_max_lag_7",

        # Dust
        "dust_mean_lag_1",
        "dust_mean_lag_3",
        "dust_mean_lag_7",

        # AOD
        "aod_mean_lag_1",
        "aod_mean_lag_3",
        "aod_mean_lag_7"
    ]


    # --------------------------------------------------------
    # Latest valid historical row
    # --------------------------------------------------------

    valid_rows = city_df.dropna(
        subset=feature_columns
    )

    if valid_rows.empty:

        return {
            "error": "Not enough historical data for prediction"
        }


    latest_row = valid_rows.iloc[-1]


    # --------------------------------------------------------
    # Base feature dataframe
    # --------------------------------------------------------

    X_base = pd.DataFrame(
        [latest_row[feature_columns].values],
        columns=feature_columns
    )


    # --------------------------------------------------------
    # Infrastructure classification
    # --------------------------------------------------------

    station_count = int(
        latest_row[
            "number_of_monitoring_stations"
        ]
    )


    if station_count <= 1:

        infrastructure_level = "Low"

    elif station_count <= 3:

        infrastructure_level = "Medium"

    else:

        infrastructure_level = "High"


    # --------------------------------------------------------
    # Encode infrastructure
    #
    # Encoder categories:
    # High, Low, Medium
    #
    # High is dropped because drop="first"
    # --------------------------------------------------------

    infra_encoded = (
        infrastructure_encoder.transform(
            [[infrastructure_level]]
        )
    )


    infra_df = pd.DataFrame(
        infra_encoded,

        columns=[
            "infrastructure_level_Low",
            "infrastructure_level_Medium"
        ]
    )


    # --------------------------------------------------------
    # Combine 39 + 2 = 41 features
    # --------------------------------------------------------

    X_final = pd.concat(
        [
            X_base.reset_index(drop=True),
            infra_df.reset_index(drop=True)
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = float(
        model.predict(X_final)[0]
    )


    # --------------------------------------------------------
    # AQI category
    # --------------------------------------------------------

    if prediction <= 50:

        category = "Good"

    elif prediction <= 100:

        category = "Satisfactory"

    elif prediction <= 200:

        category = "Moderate"

    elif prediction <= 300:

        category = "Poor"

    elif prediction <= 400:

        category = "Very Poor"

    else:

        category = "Severe"


    # --------------------------------------------------------
    # Prediction date
    # --------------------------------------------------------

    latest_date = city_df["date"].max()

    prediction_date = (
        latest_date
        + pd.Timedelta(days=horizon)
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "city": city,

        "prediction_date":
            prediction_date.strftime("%Y-%m-%d"),

        "horizon_days": horizon,

        "predicted_aqi":
            round(prediction, 2),

        "category":
            category,

        "monitoring_stations":
            station_count,

        "infrastructure_level":
            infrastructure_level,

        "latest_available_date":
            latest_date.strftime("%Y-%m-%d")
    }