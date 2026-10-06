import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import * as echarts from "echarts";
import { useRef } from "react";
import "./App.css";

const API_BASE_URL = "https://aqi-research-dashboard.onrender.com";

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

function HistoricalChart({ data }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartRef.current || !data?.length) return;

    const chart = echarts.init(chartRef.current);

    const dates = data.map((item) => item.date);
    const values = data.map((item) => item.aqi);

    chart.setOption({
      tooltip: {
        trigger: "axis",
      },
      grid: {
        left: 45,
        right: 25,
        top: 30,
        bottom: 45,
      },
      xAxis: {
        type: "category",
        data: dates,
        boundaryGap: false,
        axisLabel: {
          color: "#64748b",
          hideOverlap: true,
        },
      },
      yAxis: {
        type: "value",
        name: "AQI",
        axisLabel: {
          color: "#64748b",
        },
        splitLine: {
          lineStyle: {
            color: "#e2e8f0",
          },
        },
      },
      series: [
        {
          name: "AQI",
          type: "line",
          data: values,
          smooth: true,
          symbol: "none",
          lineStyle: {
            width: 2,
          },
          areaStyle: {
            opacity: 0.08,
          },
        },
      ],
    });

    const handleResize = () => chart.resize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.dispose();
    };
  }, [data]);

  return <div ref={chartRef} className="chart-container" />;
}

function ForecastChart({ predictions }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartRef.current || !predictions?.length) return;

    const chart = echarts.init(chartRef.current);

    chart.setOption({
      tooltip: {
        trigger: "axis",
      },
      grid: {
        left: 45,
        right: 25,
        top: 30,
        bottom: 45,
      },
      xAxis: {
        type: "category",
        data: predictions.map((item) => `${item.horizon_days} Day`),
        axisLabel: {
          color: "#64748b",
        },
      },
      yAxis: {
        type: "value",
        name: "Predicted AQI",
        axisLabel: {
          color: "#64748b",
        },
        splitLine: {
          lineStyle: {
            color: "#e2e8f0",
          },
        },
      },
      series: [
        {
          name: "Predicted AQI",
          type: "bar",
          barWidth: "45%",
          data: predictions.map((item) => item.predicted_aqi),
          label: {
            show: true,
            position: "top",
            formatter: ({ value }) => Number(value).toFixed(1),
          },
        },
      ],
    });

    const handleResize = () => chart.resize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.dispose();
    };
  }, [predictions]);

  return <div ref={chartRef} className="chart-container" />;
}

function App() {
  const [cities, setCities] = useState([]);
  const [selectedCity, setSelectedCity] = useState("Hyderabad");

  const [cityData, setCityData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [forecastComparison, setForecastComparison] = useState([]);

  const [loadingCities, setLoadingCities] = useState(true);
  const [loadingCity, setLoadingCity] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const [horizon, setHorizon] = useState(1);
  const [error, setError] = useState("");

  const selectedInfrastructure = useMemo(() => {
    if (!cityData) return null;

    const stations = cityData.monitoring_stations;

    if (stations <= 1) return "Low";
    if (stations <= 3) return "Medium";
    return "High";
  }, [cityData]);

  useEffect(() => {
    const loadCities = async () => {
      try {
        setLoadingCities(true);

        const response = await axios.get(
          `${API_BASE_URL}/api/cities`
        );

        setCities(response.data.cities || []);
      } catch (err) {
        console.error(err);
        setError("Unable to load cities.");
      } finally {
        setLoadingCities(false);
      }
    };

    loadCities();
  }, []);

  useEffect(() => {
    if (!selectedCity) return;

    const loadCity = async () => {
      try {
        setLoadingCity(true);
        setError("");

        const response = await axios.get(
          `${API_BASE_URL}/api/city/${encodeURIComponent(selectedCity)}`
        );

        setCityData(response.data);
      } catch (err) {
        console.error(err);
        setError("Unable to load city data.");
      } finally {
        setLoadingCity(false);
      }
    };

    loadCity();
  }, [selectedCity]);

  const getPrediction = async (selectedHorizon = horizon) => {
    try {
      setLoadingPrediction(true);
      setError("");

      const response = await axios.get(
        `${API_BASE_URL}/api/predict/${encodeURIComponent(
          selectedCity
        )}?horizon=${selectedHorizon}`
      );

      setPrediction(response.data);
    } catch (err) {
      console.error(err);
      setError("Unable to generate AQI prediction.");
    } finally {
      setLoadingPrediction(false);
    }
  };

  const loadForecastComparison = async () => {
    try {
      setLoadingComparison(true);
      setError("");

      const horizons = [1, 3, 7];

      const responses = await Promise.all(
        horizons.map((h) =>
          axios.get(
            `${API_BASE_URL}/api/predict/${encodeURIComponent(
              selectedCity
            )}?horizon=${h}`
          )
        )
      );

      setForecastComparison(
        responses.map((response) => response.data)
      );
    } catch (err) {
      console.error(err);
      setError("Unable to load forecast comparison.");
    } finally {
      setLoadingComparison(false);
    }
  };

  useEffect(() => {
    if (!selectedCity) return;

    getPrediction(1);
    loadForecastComparison();
  }, [selectedCity]);

  const handleHorizonChange = (value) => {
    setHorizon(value);
    getPrediction(value);
  };

  const historicalData = cityData?.historical_data || [];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">AQ</div>

          <div>
            <h1>AQI Intelligence</h1>
            <p>Infrastructure-Aware Air Quality Forecasting</p>
          </div>
        </div>

        <div className="live-status">
          <span className="status-dot"></span>
          Live API
        </div>
      </header>

      <main className="dashboard">
        <section className="hero-section">
          <div>
            <span className="eyebrow">RESEARCH DASHBOARD</span>

            <h2>
              Air Quality Intelligence
              <br />
              Across Indian Cities
            </h2>

            <p>
              Multi-horizon AQI forecasting with monitoring
              infrastructure analysis.
            </p>
          </div>

          <div className="city-selector-card">
            <label>Select City</label>

            <select
              value={selectedCity}
              onChange={(e) => setSelectedCity(e.target.value)}
              disabled={loadingCities}
            >
              {cities.map((city) => (
                <option key={city} value={city}>
                  {city}
                </option>
              ))}
            </select>
          </div>
        </section>

        {error && (
          <div className="error-box">
            {error}
          </div>
        )}

        {loadingCity ? (
          <div className="loading-card">
            Loading city intelligence...
          </div>
        ) : cityData ? (
          <>
            <section className="city-heading">
              <div>
                <span className="section-label">
                  CITY INTELLIGENCE
                </span>

                <h2>{selectedCity}</h2>

                <p>
                  Latest available air quality and monitoring
                  infrastructure information.
                </p>
              </div>

              <div
                className={`infrastructure-badge ${selectedInfrastructure?.toLowerCase()}`}
              >
                <span>Monitoring Infrastructure</span>
                <strong>{selectedInfrastructure}</strong>
              </div>
            </section>

            <section className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">
                  Latest AQI
                </span>

                <strong className="kpi-value">
                  {cityData.latest_aqi}
                </strong>

                <span className="kpi-sub">
                  {cityData.latest_category}
                </span>
              </div>

              <div className="kpi-card">
                <span className="kpi-label">
                  Monitoring Stations
                </span>

                <strong className="kpi-value">
                  {cityData.monitoring_stations}
                </strong>

                <span className="kpi-sub">
                  CPCB reference network
                </span>
              </div>

              <div className="kpi-card">
                <span className="kpi-label">
                  Latest Data
                </span>

                <strong className="kpi-date">
                  {cityData.latest_available_date}
                </strong>

                <span className="kpi-sub">
                  24-hour observation
                </span>
              </div>
            </section>

            <section className="forecast-section">
              <div className="section-header">
                <div>
                  <span className="section-label">
                    AI FORECAST
                  </span>

                  <h2>Future AQI Prediction</h2>

                  <p>
                    XGBoost multi-horizon forecasting model
                  </p>
                </div>

                <div className="horizon-buttons">
                  {[1, 3, 7].map((value) => (
                    <button
                      key={value}
                      className={
                        horizon === value ? "active" : ""
                      }
                      onClick={() =>
                        handleHorizonChange(value)
                      }
                    >
                      {value} Day
                    </button>
                  ))}
                </div>
              </div>

              <div className="prediction-card">
                {loadingPrediction ? (
                  <div className="prediction-loading">
                    Generating prediction...
                  </div>
                ) : prediction ? (
                  <>
                    <div className="prediction-main">
                      <span>Predicted AQI</span>

                      <strong>
                        {Number(
                          prediction.predicted_aqi
                        ).toFixed(1)}
                      </strong>

                      <div
                        className={`prediction-category ${prediction.category
                          ?.toLowerCase()
                          .replace(" ", "-")}`}
                      >
                        {prediction.category}
                      </div>
                    </div>

                    <div className="prediction-details">
                      <div>
                        <span>Prediction Date</span>
                        <strong>
                          {prediction.prediction_date}
                        </strong>
                      </div>

                      <div>
                        <span>Forecast Horizon</span>
                        <strong>
                          {prediction.horizon_days} day
                        </strong>
                      </div>

                      <div>
                        <span>Infrastructure</span>
                        <strong>
                          {prediction.infrastructure_level}
                        </strong>
                      </div>

                      <div>
                        <span>Stations</span>
                        <strong>
                          {prediction.monitoring_stations}
                        </strong>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="prediction-loading">
                    Select a forecast horizon.
                  </div>
                )}
              </div>
            </section>

            <section className="chart-section">
              <div className="section-header">
                <div>
                  <span className="section-label">
                    FORECAST COMPARISON
                  </span>

                  <h2>1, 3 and 7 Day Forecast</h2>

                  <p>
                    Comparison of predicted AQI across
                    forecast horizons.
                  </p>
                </div>
              </div>

              <div className="chart-card">
                {loadingComparison ? (
                  <div className="chart-loading">
                    Loading forecast comparison...
                  </div>
                ) : (
                  <ForecastChart
                    predictions={forecastComparison}
                  />
                )}
              </div>
            </section>

            <section className="chart-section">
              <div className="section-header">
                <div>
                  <span className="section-label">
                    HISTORICAL ANALYSIS
                  </span>

                  <h2>AQI Trend</h2>

                  <p>
                    Historical daily AQI for {selectedCity}.
                  </p>
                </div>
              </div>

              <div className="chart-card">
                <HistoricalChart data={historicalData} />
              </div>
            </section>
          </>
        ) : null}

        <section className="infrastructure-section">
          <div className="section-header">
            <div>
              <span className="section-label">
                RESEARCH ANALYSIS
              </span>

              <h2>Infrastructure vs Forecasting Error</h2>

              <p>
                Observed 1-day XGBoost MAE grouped by monitoring
                infrastructure level.
              </p>
            </div>
          </div>

          <div className="infrastructure-grid">
            {infrastructureAnalysis.map((item) => (
              <div
                className="infrastructure-card"
                key={item.level}
              >
                <div className="infrastructure-card-top">
                  <span
                    className={`level-dot ${item.level.toLowerCase()}`}
                  ></span>

                  <h3>{item.level}</h3>
                </div>

                <div className="infra-mae">
                  <strong>{item.mae.toFixed(2)}</strong>
                  <span>MAE</span>
                </div>

                <div className="infra-stats">
                  <div>
                    <span>Cities</span>
                    <strong>{item.cities}</strong>
                  </div>

                  <div>
                    <span>Observations</span>
                    <strong>
                      {item.observations.toLocaleString()}
                    </strong>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="research-note">
            <strong>Research interpretation</strong>

            <p>
              Monitoring infrastructure provides contextual
              information about AQI forecasting performance.
              These observed differences should not be
              interpreted as causal evidence that monitoring
              station count directly determines forecasting
              accuracy.
            </p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <div>
          <strong>AQI Intelligence Dashboard</strong>
          <span>
            Multi-Horizon AQI Forecasting Research
          </span>
        </div>

        <span>
          XGBoost • FastAPI • React • CPCB Data
        </span>
      </footer>
    </div>
  );
}

export default App;