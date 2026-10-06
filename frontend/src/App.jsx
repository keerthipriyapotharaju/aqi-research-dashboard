import { useEffect, useState } from "react";
import axios from "axios";
import ReactECharts from "echarts-for-react";
import "./index.css";

function App() {
  const [cities, setCities] = useState([]);
  const [selectedCity, setSelectedCity] = useState("Hyderabad");
  const [cityData, setCityData] = useState(null);

  const [horizon, setHorizon] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [forecastComparison, setForecastComparison] = useState([]);

  const [loading, setLoading] = useState(false);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  // =========================================================
  // RESEARCH INFRASTRUCTURE RESULTS
  // =========================================================

  const infrastructureAnalysis = [
    {
      level: "Low",
      cities: 36,
      observations: 9936,
      mae: 26.239,
    },
    {
      level: "Medium",
      cities: 12,
      observations: 3312,
      mae: 25.302,
    },
    {
      level: "High",
      cities: 2,
      observations: 552,
      mae: 61.653,
    },
  ];

  // =========================================================
  // LOAD CITIES
  // =========================================================

  useEffect(() => {
    axios
      .get("http://127.0.0.1:8000/api/cities")
      .then((response) => {
        setCities(response.data.cities);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []);

  // =========================================================
  // LOAD CITY DATA
  // =========================================================

  useEffect(() => {
    setLoading(true);
    setPrediction(null);
    setForecastComparison([]);

    axios
      .get(`http://127.0.0.1:8000/api/city/${selectedCity}`)
      .then((response) => {
        setCityData(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error(error);
        setLoading(false);
      });
  }, [selectedCity]);

  // =========================================================
  // SINGLE HORIZON PREDICTION
  // =========================================================

  const handlePrediction = () => {
    setPredictionLoading(true);
    setPrediction(null);

    axios
      .get(
        `http://127.0.0.1:8000/api/predict/${selectedCity}?horizon=${horizon}`
      )
      .then((response) => {
        setPrediction(response.data);
        setPredictionLoading(false);
      })
      .catch((error) => {
        console.error(error);

        setPrediction({
          error: "Prediction failed. Please try again.",
        });

        setPredictionLoading(false);
      });
  };

  // =========================================================
  // ALL THREE FORECASTS
  // =========================================================

  const handleForecastComparison = async () => {
    setComparisonLoading(true);
    setForecastComparison([]);

    try {
      const horizons = [1, 3, 7];

      const results = await Promise.all(
        horizons.map((h) =>
          axios.get(
            `http://127.0.0.1:8000/api/predict/${selectedCity}?horizon=${h}`
          )
        )
      );

      const comparison = results.map(
        (response) => response.data
      );

      setForecastComparison(comparison);
    } catch (error) {
      console.error(error);
    }

    setComparisonLoading(false);
  };

  // =========================================================
  // HISTORICAL AQI CHART
  // =========================================================

  const chartOption = cityData
    ? {
        tooltip: {
          trigger: "axis",
        },

        grid: {
          left: "5%",
          right: "4%",
          bottom: "12%",
          top: "8%",
          containLabel: true,
        },

        xAxis: {
          type: "category",

          data: cityData.records.map(
            (item) => item.date
          ),

          axisLabel: {
            color: "#667085",
            fontSize: 10,
          },
        },

        yAxis: {
          type: "value",

          name: "AQI",

          nameTextStyle: {
            color: "#667085",
          },

          axisLabel: {
            color: "#667085",
          },
        },

        series: [
          {
            name: "AQI",

            type: "line",

            data: cityData.records.map(
              (item) => item.india_aqi
            ),

            smooth: true,

            showSymbol: false,

            lineStyle: {
              width: 2,
            },

            areaStyle: {
              opacity: 0.08,
            },
          },
        ],
      }
    : {};

  // =========================================================
  // FORECAST COMPARISON CHART
  // =========================================================

  const forecastChartOption =
    forecastComparison.length > 0
      ? {
          tooltip: {
            trigger: "axis",
          },

          grid: {
            left: "5%",
            right: "5%",
            bottom: "12%",
            top: "15%",
            containLabel: true,
          },

          xAxis: {
            type: "category",

            data: forecastComparison.map(
              (item) => `${item.horizon_days}D`
            ),

            axisLabel: {
              color: "#667085",
              fontSize: 12,
            },
          },

          yAxis: {
            type: "value",

            name: "Predicted AQI",

            nameTextStyle: {
              color: "#667085",
            },

            axisLabel: {
              color: "#667085",
            },
          },

          series: [
            {
              name: "Predicted AQI",

              type: "bar",

              barWidth: "45%",

              data: forecastComparison.map(
                (item) => item.predicted_aqi
              ),

              label: {
                show: true,
                position: "top",
                formatter: "{c}",
              },
            },
          ],
        }
      : {};

  // =========================================================
  // INFRASTRUCTURE CLASS
  // =========================================================

  const infrastructureClass =
    cityData?.infrastructure_level?.toLowerCase() || "";

  // =========================================================
  // RETURN UI
  // =========================================================

  return (
    <div className="dashboard">

      {/* =====================================================
          HEADER
      ===================================================== */}

      <header className="header">

        <div>

          <div className="header-title">
            AQI Research Dashboard
          </div>

          <div className="header-subtitle">
            Multi-City AQI Forecasting & Infrastructure Analysis
          </div>

        </div>

        <div className="header-subtitle">
          Research Prototype
        </div>

      </header>


      <main className="main-container">

        {/* ===================================================
            CITY SELECTOR
        =================================================== */}

        <div className="control-card">

          <label className="control-label">
            SELECT CITY
          </label>

          <select
            className="city-select"
            value={selectedCity}
            onChange={(e) =>
              setSelectedCity(e.target.value)
            }
          >

            {cities.map((city) => (
              <option
                key={city}
                value={city}
              >
                {city}
              </option>
            ))}

          </select>

        </div>


        {/* ===================================================
            LOADING
        =================================================== */}

        {loading && (
          <div className="loading">
            Loading city data...
          </div>
        )}


        {cityData && !loading && (
          <>

            {/* =================================================
                CITY HEADER
            ================================================= */}

            <div className="city-heading">

              <div>

                <div className="city-name">
                  {cityData.city}
                </div>

                <div className="city-description">
                  Historical air quality and monitoring infrastructure
                </div>

              </div>

              <span
                className={`infrastructure-badge infrastructure-${infrastructureClass}`}
              >
                {cityData.infrastructure_level} Infrastructure
              </span>

            </div>


            {/* =================================================
                KPI CARDS
            ================================================= */}

            <div className="kpi-grid">

              <div className="kpi-card">

                <div className="kpi-label">
                  MONITORING STATIONS
                </div>

                <div className="kpi-value">
                  {cityData.monitoring_stations}
                </div>

                <div className="kpi-small">
                  CPCB reference network
                </div>

              </div>


              <div className="kpi-card">

                <div className="kpi-label">
                  LATEST AQI
                </div>

                <div className="kpi-value">
                  {
                    cityData.records[
                      cityData.records.length - 1
                    ].india_aqi
                  }
                </div>

                <div className="kpi-small">
                  Most recent available value
                </div>

              </div>


              <div className="kpi-card">

                <div className="kpi-label">
                  DATA RECORDS
                </div>

                <div className="kpi-value">
                  {cityData.records.length.toLocaleString()}
                </div>

                <div className="kpi-small">
                  Daily observations
                </div>

              </div>

            </div>


            {/* =================================================
                AQI FORECAST
            ================================================= */}

            <div className="prediction-card">

              <div className="prediction-header">

                <div>

                  <div className="prediction-title">
                    AQI Forecast
                  </div>

                  <div className="prediction-subtitle">
                    XGBoost multi-horizon prediction
                  </div>

                </div>

              </div>


              <div className="prediction-controls">

                <div>

                  <label className="control-label">
                    FORECAST HORIZON
                  </label>

                  <select
                    className="city-select"
                    value={horizon}
                    onChange={(e) =>
                      setHorizon(
                        Number(e.target.value)
                      )
                    }
                  >

                    <option value={1}>
                      Next 1 Day
                    </option>

                    <option value={3}>
                      Next 3 Days
                    </option>

                    <option value={7}>
                      Next 7 Days
                    </option>

                  </select>

                </div>


                <button
                  className="predict-button"
                  onClick={handlePrediction}
                  disabled={predictionLoading}
                >

                  {predictionLoading
                    ? "Predicting..."
                    : "Predict AQI"}

                </button>


                <button
                  className="predict-button"
                  onClick={handleForecastComparison}
                  disabled={comparisonLoading}
                >

                  {comparisonLoading
                    ? "Loading..."
                    : "Compare 1D / 3D / 7D"}

                </button>

              </div>


              {/* =================================================
                  SINGLE PREDICTION RESULT
              ================================================= */}

              {prediction && !prediction.error && (

                <div className="prediction-result">

                  <div className="prediction-value-box">

                    <div className="prediction-label">
                      PREDICTED AQI
                    </div>

                    <div className="prediction-value">
                      {prediction.predicted_aqi}
                    </div>

                  </div>


                  <div className="prediction-info">

                    <div className="prediction-info-item">

                      <span>
                        Prediction Date
                      </span>

                      <strong>
                        {prediction.prediction_date}
                      </strong>

                    </div>


                    <div className="prediction-info-item">

                      <span>
                        Horizon
                      </span>

                      <strong>
                        {prediction.horizon_days} Day
                        {prediction.horizon_days > 1
                          ? "s"
                          : ""}
                      </strong>

                    </div>


                    <div className="prediction-info-item">

                      <span>
                        AQI Category
                      </span>

                      <strong>
                        {prediction.category}
                      </strong>

                    </div>


                    <div className="prediction-info-item">

                      <span>
                        Infrastructure
                      </span>

                      <strong>
                        {prediction.infrastructure_level}
                      </strong>

                    </div>

                  </div>

                </div>

              )}


              {prediction?.error && (

                <div className="prediction-error">
                  {prediction.error}
                </div>

              )}

            </div>


            {/* =================================================
                FORECAST COMPARISON
            ================================================= */}

            {forecastComparison.length > 0 && (

              <div className="chart-card">

                <div className="chart-title">
                  Multi-Horizon Forecast Comparison
                </div>

                <div className="chart-subtitle">
                  Predicted AQI across 1-day, 3-day and 7-day horizons
                </div>

                <ReactECharts
                  option={forecastChartOption}
                  style={{
                    height: "350px",
                    width: "100%",
                    marginTop: "15px",
                  }}
                />

              </div>

            )}


            {/* =================================================
                INFRASTRUCTURE ANALYSIS
            ================================================= */}

            <div className="chart-card infrastructure-analysis-card">

              <div className="chart-title">
                Monitoring Infrastructure Analysis
              </div>

              <div className="chart-subtitle">
                Forecasting error across infrastructure levels
              </div>


              <div className="infrastructure-grid">

                {infrastructureAnalysis.map((item) => (

                  <div
                    className="infrastructure-analysis-box"
                    key={item.level}
                  >

                    <div className="infrastructure-analysis-level">
                      {item.level} Infrastructure
                    </div>

                    <div className="infrastructure-analysis-mae">
                      {item.mae}
                    </div>

                    <div className="infrastructure-analysis-label">
                      Mean Absolute Error
                    </div>

                    <div className="infrastructure-analysis-details">
                      {item.cities} cities
                      <br />
                      {item.observations.toLocaleString()} test observations
                    </div>

                  </div>

                ))}

              </div>


              <div className="infrastructure-note">

                <strong>
                  Research interpretation:
                </strong>{" "}

                Infrastructure level provides contextual
                information about forecasting performance,
                but the observed differences should not be
                interpreted as a causal effect of monitoring
                station count.

              </div>

            </div>


            {/* =================================================
                HISTORICAL AQI
            ================================================= */}

            <div className="chart-card">

              <div className="chart-title">
                Historical AQI Trend
              </div>

              <div className="chart-subtitle">
                Daily India AQI values for{" "}
                {cityData.city}
              </div>

              <ReactECharts
                option={chartOption}
                style={{
                  height: "500px",
                  width: "100%",
                  marginTop: "15px",
                }}
              />

            </div>

          </>
        )}

      </main>

    </div>
  );
}

export default App;