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
    description="Multi-horizon AQI forecasting API",
    version="1.0.0"
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
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = (
    BASE_DIR /
    "air_quality_daily_final_with_infrastructure.csv"
)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df.sort_values(
        ["city", "date"]
    )
    .reset_index(drop=True)
)

print("Dataset loaded successfully!")


# ============================================================
# LOAD XGBOOST MODELS
# ============================================================

model_1day = XGBRegressor()
model_1day.load_model(
    str(BASE_DIR / "xgb_1day.json")
)

model_3day = XGBRegressor()
model_3day.load_model(
    str(BASE_DIR / "xgb_3day.json")
)

model_7day = XGBRegressor()
model_7day.load_model(
    str(BASE_DIR / "xgb_7day.json")
)

print("XGBoost models loaded successfully!")


# ============================================================
# LOAD INFRASTRUCTURE ENCODER
# ============================================================

infrastructure_encoder = joblib.load(
    BASE_DIR / "infrastructure_encoder.pkl"
)

print(
    "Infrastructure encoder loaded successfully!"
)


# ============================================================
# INFRASTRUCTURE CLASSIFICATION
# ============================================================

def classify_infrastructure(
    station_count: int
):

    if station_count <= 1:
        return "Low"

    elif station_count <= 3:
        return "Medium"

    else:
        return "High"


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(
    aqi: float
):

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

    else:
        return "Severe"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AQI Research Dashboard API",
        "status": "running"
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
def get_summary():

    return {

        "total_cities":
            int(df["city"].nunique()),

        "total_records":
            int(len(df)),

        "date_start":
            df["date"].min().strftime("%Y-%m-%d"),

        "date_end":
            df["date"].max().strftime("%Y-%m-%d"),

        "average_aqi":
            round(
                float(df["india_aqi"].mean()),
                2
            ),

        "maximum_aqi":
            int(
                df["india_aqi"].max()
            )
    }


# ============================================================
# CITIES
# ============================================================

@app.get("/api/cities")
def get_cities():

    cities = sorted(
        df["city"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "cities": cities
    }


# ============================================================
# CITY DATA
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

    city_df = city_df.sort_values(
        "date"
    )

    # --------------------------------------------------------
    # Monitoring stations
    # --------------------------------------------------------

    station_count = int(
        city_df[
            "number_of_monitoring_stations"
        ].iloc[0]
    )


    # --------------------------------------------------------
    # Infrastructure classification
    # --------------------------------------------------------

    infrastructure_level = (
        classify_infrastructure(
            station_count
        )
    )


    # --------------------------------------------------------
    # Latest record
    # --------------------------------------------------------

    latest = city_df.iloc[-1]


    # --------------------------------------------------------
    # Historical data for charts
    # --------------------------------------------------------

    historical_data = (

        city_df[
            [
                "date",
                "india_aqi"
            ]
        ]

        .assign(
            date=lambda x:
            x["date"].dt.strftime(
                "%Y-%m-%d"
            )
        )

        .rename(
            columns={
                "india_aqi": "aqi"
            }
        )

        .to_dict(
            orient="records"
        )
    )


    # --------------------------------------------------------
    # Detailed records
    # --------------------------------------------------------

    records = (

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
            x["date"].dt.strftime(
                "%Y-%m-%d"
            )
        )

        .to_dict(
            orient="records"
        )
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "city":
            city,

        "monitoring_stations":
            station_count,

        "infrastructure_level":
            infrastructure_level,

        "latest_aqi":
            int(
                latest["india_aqi"]
            ),

        "latest_category":
            str(
                latest["india_aqi_category"]
            ),

        "latest_available_date":
            latest["date"].strftime(
                "%Y-%m-%d"
            ),

        "historical_data":
            historical_data,

        "records":
            records
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
            "error":
            "Horizon must be 1, 3, or 7 days"
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


    city_df = (
        city_df
        .sort_values("date")
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if horizon == 1:

        model = model_1day

    elif horizon == 3:

        model = model_3day

    else:

        model = model_7day


    # ========================================================
    # AQI LAG FEATURES
    # ========================================================

    city_df["aqi_lag_1"] = (
        city_df["india_aqi"].shift(1)
    )

    city_df["aqi_lag_3"] = (
        city_df["india_aqi"].shift(3)
    )

    city_df["aqi_lag_7"] = (
        city_df["india_aqi"].shift(7)
    )


    # ========================================================
    # ROLLING AQI FEATURES
    # ========================================================

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


    # ========================================================
    # DATE FEATURES
    # ========================================================

    city_df["month"] = (
        city_df["date"].dt.month
    )

    city_df["day_of_week"] = (
        city_df["date"].dt.dayofweek
    )

    city_df["day_of_year"] = (
        city_df["date"].dt.dayofyear
    )


    # ========================================================
    # POLLUTANT COLUMNS
    # ========================================================

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


    # ========================================================
    # POLLUTANT LAG FEATURES
    # ========================================================

    for col in pollutants:

        city_df[
            f"{col}_lag_1"
        ] = city_df[col].shift(1)

        city_df[
            f"{col}_lag_3"
        ] = city_df[col].shift(3)

        city_df[
            f"{col}_lag_7"
        ] = city_df[col].shift(7)


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


    # ========================================================
    # LATEST VALID HISTORICAL ROW
    # ========================================================

    valid_rows = city_df.dropna(
        subset=feature_columns
    )

    if valid_rows.empty:

        return {
            "error":
            "Not enough historical data for prediction"
        }


    latest_row = valid_rows.iloc[-1]


    # ========================================================
    # BASE FEATURE DATAFRAME
    # ========================================================

    X_base = pd.DataFrame(
        [
            latest_row[
                feature_columns
            ].values
        ],
        columns=feature_columns
    )


    # ========================================================
    # INFRASTRUCTURE
    # ========================================================

    station_count = int(
        latest_row[
            "number_of_monitoring_stations"
        ]
    )

    infrastructure_level = (
        classify_infrastructure(
            station_count
        )
    )


    # ========================================================
    # INFRASTRUCTURE ENCODING
    #
    # Encoder categories:
    # High, Low, Medium
    #
    # High is dropped because
    # drop="first"
    # ========================================================

    infra_encoded = (
        infrastructure_encoder.transform(
            [[
                infrastructure_level
            ]]
        )
    )


    infra_df = pd.DataFrame(

        infra_encoded,

        columns=[
            "infrastructure_level_Low",
            "infrastructure_level_Medium"
        ]
    )


    # ========================================================
    # COMBINE 39 + 2 = 41 FEATURES
    # ========================================================

    X_final = pd.concat(

        [
            X_base.reset_index(
                drop=True
            ),

            infra_df.reset_index(
                drop=True
            )
        ],

        axis=1
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = float(
        model.predict(
            X_final
        )[0]
    )


    # ========================================================
    # AQI CATEGORY
    # ========================================================

    category = get_aqi_category(
        prediction
    )


    # ========================================================
    # PREDICTION DATE
    # ========================================================

    latest_date = (
        city_df["date"].max()
    )

    prediction_date = (

        latest_date
        +
        pd.Timedelta(
            days=horizon
        )
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "city":
            city,

        "prediction_date":
            prediction_date.strftime(
                "%Y-%m-%d"
            ),

        "horizon_days":
            horizon,

        "predicted_aqi":
            round(
                prediction,
                2
            ),

        "category":
            category,

        "monitoring_stations":
            station_count,

        "infrastructure_level":
            infrastructure_level,

        "latest_available_date":
            latest_date.strftime(
                "%Y-%m-%d"
            )
    }