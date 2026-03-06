import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import PropTypes from "prop-types";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler,
} from "chart.js";
import { Bar, Line, Doughnut } from "react-chartjs-2";
import "./index.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler,
);

const API = "http://localhost:8000";

const chartDefaults = {
  plugins: {
    legend: { labels: { color: "#94a3b8", font: { family: "Outfit" } } },
  },
  scales: {
    x: {
      ticks: { color: "#64748b" },
      grid: { color: "rgba(255,255,255,0.05)" },
    },
    y: {
      ticks: { color: "#64748b" },
      grid: { color: "rgba(255,255,255,0.05)" },
    },
  },
};

/* ── Small Components ──────────────────────────────────────────────────────── */
const Spinner = () => <span className="spinner" />;

const ConsoleBadge = ({ logs }) => {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs]);
  return (
    <div className="console" ref={ref}>
      {logs || "> System initialised. Awaiting commands...\n"}
    </div>
  );
};
ConsoleBadge.propTypes = { logs: PropTypes.string };

const MetricCard = ({ label, value, cls = "" }) => (
  <div className="metric-card fade-in">
    <div className="metric-label">{label}</div>
    <div className={`metric-value ${cls}`}>{value}</div>
  </div>
);
MetricCard.propTypes = {
  label: PropTypes.string,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  cls: PropTypes.string,
};

/* ── Main App ──────────────────────────────────────────────────────────────── */
export default function App() {
  const [tab, setTab] = useState("train");
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState("");
  const [backendStatus, setBackendStatus] = useState(null);
  const [edaImages, setEdaImages] = useState([]);
  const [evalImages, setEvalImages] = useState([]);
  const [lossHistory, setLossHistory] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [predResult, setPredResult] = useState(null);
  const [predInput, setPredInput] = useState({
    performance_rating: 78,
    goals_assists: 3,
    minutes_played: 200,
    days_injured: 0,
    social_sentiment_score: 0.3,
    contract_duration_months: 24,
    position: "Midfielder",
    model_choice: "xgboost",
  });
  const [pipelineSteps, setPipelineSteps] = useState({
    generated: false,
    sentiment: false,
    preprocessed: false,
    xgboostTrained: false,
    lstmTrained: false,
    evaluated: false,
    eda: false,
  });

  const log = useCallback((msg) => {
    setLogs((prev) => prev + msg + "\n");
  }, []);

  // Status + images check on mount
  useEffect(() => {
    axios
      .get(`${API}/api/status`)
      .then((r) => {
        setBackendStatus(r.data);
        const f = r.data.files;
        setPipelineSteps((prev) => ({
          ...prev,
          generated: f.dataset,
          preprocessed: f.processed,
          xgboostTrained: f.xgboost_model,
          lstmTrained: f.univariate_lstm,
        }));
        if (r.data.metrics) setMetrics(r.data.metrics);
        const msgs = [
          f.dataset
            ? "  [OK] Dataset found."
            : "  [MISSING] Dataset — run Step 1.",
          f.processed
            ? "  [OK] Processed data found."
            : "  [MISSING] Preprocessed data — run Step 3.",
          f.xgboost_model
            ? "  [OK] XGBoost model found."
            : "  [MISSING] XGBoost model.",
          f.univariate_lstm
            ? "  [OK] LSTM model found."
            : "  [MISSING] LSTM model.",
        ].join("\n");
        log("> Backend online. Detected pipeline state:\n" + msgs);
        // Fetch training history if LSTM is trained
        if (f.univariate_lstm || f.multivariate_lstm) {
          axios
            .get(`${API}/api/models/history`)
            .then((histRes) => {
              if (histRes.data.status === "ok") setLossHistory(histRes.data);
            })
            .catch(() => {});
        }
      })
      .catch(() => {
        setBackendStatus(null);
        log(
          "> [WARN] Backend not reachable. Start uvicorn backend.main:app --reload",
        );
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const callApi = async (endpoint, method = "POST", label, stepKey = null) => {
    setLoading(true);
    log(`\n> ${label}...`);
    try {
      const res = await axios({ method, url: `${API}${endpoint}` });
      if (res.data.status === "skipped") {
        log(`  [SKIP] ${res.data.output}`);
      } else {
        log(res.data.output || "  Done.");
        if (stepKey) setPipelineSteps((p) => ({ ...p, [stepKey]: true }));
      }
      // Route images to the correct bucket
      if (res.data.images) {
        const imgs = res.data.images;
        if (imgs.some((i) => i.includes("eda_")))
          setEdaImages(imgs.filter((i) => i.includes("eda_")));
        else setEvalImages(imgs);
      }
      if (res.data.metrics) setMetrics(res.data.metrics);

      return res.data;
    } catch (err) {
      log(`  [ERROR] ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    setPredResult(null);
    log("\n> Running live prediction...");
    try {
      const res = await axios.post(`${API}/api/predict`, predInput);
      setPredResult(res.data);
      log(`  Predicted Transfer Value: €${res.data.predicted_value_millions}M`);
    } catch (err) {
      log(`  [ERROR] ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /* ── Fake chart data for demonstration when models not yet run ── */
  const sentimentChartData = {
    labels: [
      "Very Negative",
      "Negative",
      "Neutral",
      "Positive",
      "Very Positive",
    ],
    datasets: [
      {
        label: "Players",
        data: [8, 22, 35, 28, 7],
        backgroundColor: [
          "#ef4444",
          "#f97316",
          "#eab308",
          "#22c55e",
          "#818cf8",
        ],
        borderRadius: 6,
      },
    ],
  };

  /* ── Dynamic Chart Data ── */
  const lossChartData = useMemo(() => {
    if (
      !lossHistory ||
      (!lossHistory.univariate && !lossHistory.multivariate)
    ) {
      return {
        labels: ["E1", "E2", "E3"],
        datasets: [
          { label: "Train missing", data: [0, 0, 0], borderColor: "#818cf8" },
        ],
      };
    }
    const h = lossHistory.multivariate || lossHistory.univariate;
    const epochs = h.loss?.length || 15;
    return {
      labels: Array.from({ length: epochs }, (_, i) => `E${i + 1}`),
      datasets: [
        {
          label: "Train Loss",
          data: h.loss || [],
          borderColor: "#818cf8",
          backgroundColor: "rgba(129,140,248,0.15)",
          fill: true,
          tension: 0.4,
          pointRadius: 2,
        },
        {
          label: "Val Loss",
          data: h.val_loss || [],
          borderColor: "#c084fc",
          backgroundColor: "rgba(192,132,252,0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 2,
        },
      ],
    };
  }, [lossHistory]);

  const positionChartData = {
    labels: ["Forward", "Midfielder", "Defender", "Goalkeeper"],
    datasets: [
      {
        data: [28, 32, 30, 10],
        backgroundColor: ["#818cf8", "#c084fc", "#34d399", "#fbbf24"],
        borderWidth: 0,
        hoverOffset: 8,
      },
    ],
  };

  /* ── Tabs Content ────────────────────────────────────────────────────────── */
  const tabs = [
    { id: "train", label: "Training" },
    { id: "eda", label: "EDA" },
    { id: "predict", label: "Predict" },
  ];

  return (
    <>
      <div className="bg-mesh">
        <div className="wave wave-1" />
        <div className="wave wave-2" />
      </div>

      <div className="app-wrapper">
        {/* ── Header ── */}
        <header className="site-header">
          <span className="logo-icon" style={{ fontSize: '2rem', color: 'var(--accent)', letterSpacing: '-2px', fontWeight: 800 }}>{ '///' }</span>
          <h1>AI TransferIQ</h1>
          <p>
            Dynamic Football Player Transfer Value Prediction using LSTM,
            XGBoost &amp; VADER Sentiment Analysis
          </p>
          {backendStatus ? (
            <div className="status-pill">
              <span className="status-dot" />
              Backend Online
            </div>
          ) : (
            <div className="status-pill offline">
              <span className="status-dot" />
              Backend Offline
            </div>
          )}
        </header>

        {/* ── Nav Tabs ── */}
        <nav className="nav-tabs">
          {tabs.map((t) => (
            <button
              key={t.id}
              className={`tab-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* ══════════════ TAB: EDA ══════════════ */}
        {tab === "eda" && (
          <div className="fade-in">
            <div className="card" style={{ marginBottom: "1.5rem" }}>
              <div className="card-title">Exploratory Data Analysis</div>
              <div className="card-desc">
                Generate charts exploring player statistics, market value
                distributions, and feature correlations (PDF Milestone 1
                deliverable).
              </div>
              <button
                className="btn btn-primary"
                style={{ maxWidth: 280 }}
                disabled={loading}
                onClick={async () => {
                  // ?force=true causes the backend to actually run the EDA python script in real-time
                  const res = await callApi(
                    "/api/data/eda?force=true",
                    "POST",
                    "Running EDA",
                  );
                  if (res?.images?.length)
                    setEdaImages(res.images.filter((i) => i.includes("eda_")));
                }}
              >
                {loading ? <Spinner /> : "Generate Charts"}
              </button>
            </div>

            {edaImages.length > 0 && (
              <div className="grid-2" style={{ marginBottom: "1.5rem" }}>
                <div className="card">
                  <div className="card-title">
                    VADER Sentiment Distribution
                  </div>
                  <Bar
                    data={sentimentChartData}
                    options={{
                      ...chartDefaults,
                      responsive: true,
                      plugins: {
                        ...chartDefaults.plugins,
                        legend: { display: false },
                      },
                    }}
                  />
                </div>
                <div className="card">
                  <div className="card-title">Player Position Breakdown</div>
                  <Doughnut
                    data={positionChartData}
                    options={{
                      responsive: true,
                      plugins: { legend: { labels: { color: "#94a3b8" } } },
                    }}
                  />
                </div>
              </div>
            )}

            {edaImages.length > 0 && (
              <div className="card fade-in">
                <div className="card-title">Generated EDA Plots</div>
                <div className="img-gallery">
                  {edaImages.map((img, idx) => (
                    <img
                      key={idx}
                      src={`${API}${img}?t=${Date.now()}`}
                      alt="EDA chart"
                      onClick={() => window.open(`${API}${img}`, "_blank")}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══════════════ TAB: TRAINING ══════════════ */}
        {tab === "train" && (
          <div className="fade-in">
            <div className="grid-2" style={{ marginBottom: "1.5rem" }}>
              {/* Model Training */}
              <div className="card">
                <div className="card-title">Model Training Engine</div>
                <div className="card-desc">
                  Train ensemble (XGBoost) and time-series (LSTM) models on the
                  processed dataset.
                </div>

                <div className="steps" style={{ marginBottom: "1.5rem" }}>
                  {[
                    {
                      key: "xgboostTrained",
                      label: "XGBoost ensemble trained",
                    },
                    {
                      key: "lstmTrained",
                      label: "Univariate + Multivariate LSTM trained",
                    },
                  ].map((s, i) => (
                    <div
                      className={`step ${pipelineSteps[s.key] ? "done" : ""}`}
                      key={s.key}
                    >
                      <span className="step-num">
                        {pipelineSteps[s.key] ? "✓" : i + 1}
                      </span>
                      {s.label}
                    </div>
                  ))}
                </div>

                <button
                  className="btn btn-primary"
                  disabled={loading}
                  onClick={() =>
                    callApi(
                      "/api/models/train/xgboost?force=true",
                      "POST",
                      "Training XGBoost Ensemble",
                      "xgboostTrained",
                    )
                  }
                >
                  {loading ? <Spinner /> : "Train XGBoost"}
                  {pipelineSteps.xgboostTrained && (
                    <span
                      className="badge badge-success"
                      style={{ marginLeft: "auto" }}
                    >
                      ✓ Done
                    </span>
                  )}
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={loading}
                  onClick={async () => {
                    await callApi(
                      "/api/models/train/lstm?force=true",
                      "POST",
                      "Training LSTM Models",
                      "lstmTrained",
                    );
                    const histRes = await axios.get(
                      `${API}/api/models/history`,
                    );
                    if (histRes.data.status === "ok")
                      setLossHistory(histRes.data);
                  }}
                >
                  {loading ? <Spinner /> : "Train LSTM Networks"}
                  {pipelineSteps.lstmTrained && (
                    <span
                      className="badge badge-success"
                      style={{ marginLeft: "auto" }}
                    >
                      ✓ Done
                    </span>
                  )}
                </button>
                <button
                  className="btn btn-outline"
                  disabled={loading}
                  onClick={() =>
                    callApi(
                      "/api/models/evaluate",
                      "POST",
                      "Evaluating all models",
                      "evaluated",
                    )
                  }
                >
                  {loading ? <Spinner /> : "Evaluate All Models"}
                </button>
              </div>

              {/* Evaluation Metrics */}
              <div className="card">
                <div className="card-title">
                  Final Model Evaluation Metrics
                </div>
                <div
                  style={{ marginTop: "1rem", marginBottom: "1.5rem" }}
                  className="grid-3"
                >
                  <MetricCard
                    label="XGB R² Target ≥ 0.75"
                    value={metrics ? metrics.r2.toFixed(3) : "--"}
                    cls={metrics && metrics.r2 >= 0.75 ? "green" : "purple"}
                  />
                  <MetricCard
                    label="XGB MAPE (Error)"
                    value={
                      metrics ? `${(metrics.mape * 100).toFixed(1)}%` : "--"
                    }
                    cls={metrics && metrics.mape <= 0.15 ? "green" : "purple"}
                  />
                  <MetricCard
                    label="Naive Forecast MAPE"
                    value={
                      metrics
                        ? `${(metrics.naive_mape * 100).toFixed(1)}%`
                        : "--"
                    }
                  />
                </div>
                {metrics && metrics.lstm_r2 !== undefined && (
                  <div
                    style={{ marginTop: "0", marginBottom: "1.5rem" }}
                    className="grid-3"
                  >
                    <MetricCard
                      label="LSTM R² Score"
                      value={metrics.lstm_r2.toFixed(3)}
                      cls={metrics.lstm_r2 >= 0.75 ? "green" : "purple"}
                    />
                    <MetricCard
                      label="LSTM MAPE (Error)"
                      value={`${(metrics.lstm_mape * 100).toFixed(1)}%`}
                      cls={metrics.lstm_mape <= 0.15 ? "green" : "purple"}
                    />
                    <MetricCard
                      label="Rolling Avg MAPE"
                      value={`${(metrics.rolling_mape * 100).toFixed(1)}%`}
                    />
                  </div>
                )}
                {metrics && metrics.business_explanation && (
                  <div
                    style={{
                      padding: "1rem",
                      background: "rgba(52, 211, 153, 0.1)",
                      borderLeft: "4px solid #34d399",
                      borderRadius: "4px",
                    }}
                  >
                    <strong>Business Assessment:</strong>{" "}
                    {metrics.business_explanation}
                  </div>
                )}
                <div style={{ marginTop: "2rem" }}>
                  <div className="card-title">LSTM Training Loss Curves</div>
                  <Line
                    data={lossChartData}
                    options={{ ...chartDefaults, responsive: true }}
                  />
                </div>
              </div>
            </div>

            {/* Evaluation Plots */}
            {evalImages.length > 0 && (
              <div className="card fade-in">
                <div className="card-title">
                  Model Evaluation Visualizations
                </div>
                <div className="img-gallery">
                  {evalImages.map((img, idx) => (
                    <img
                      key={idx}
                      src={`${API}${img}?t=${Date.now()}`}
                      alt="Eval chart"
                      onClick={() => window.open(`${API}${img}`, "_blank")}
                    />
                  ))}
                </div>
              </div>
            )}

            <div className="card fade-in">
              <div className="card-title">⚙️ System Console</div>
              <ConsoleBadge logs={logs} />
            </div>
          </div>
        )}

        {/* ══════════════ TAB: PREDICT ══════════════ */}
        {tab === "predict" && (
          <div className="fade-in">
            <div className="grid-2">
              {/* Prediction Form */}
              <div className="card">
                <div className="card-title">
                  🎯 Live Transfer Value Predictor
                </div>
                <div className="card-desc">
                  Enter a player&apos;s current stats to estimate their market
                  value using the trained XGBoost model.
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Performance Rating (0–100)</label>
                    <input
                      type="number"
                      className="form-control"
                      min="0"
                      max="100"
                      value={predInput.performance_rating}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          performance_rating: +e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label>Goals + Assists (last month)</label>
                    <input
                      type="number"
                      className="form-control"
                      min="0"
                      value={predInput.goals_assists}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          goals_assists: +e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label>Minutes Played</label>
                    <input
                      type="number"
                      className="form-control"
                      min="0"
                      max="360"
                      value={predInput.minutes_played}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          minutes_played: +e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label>Days Injured</label>
                    <input
                      type="number"
                      className="form-control"
                      min="0"
                      max="30"
                      value={predInput.days_injured}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          days_injured: +e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label>Sentiment Score (−1 to 1)</label>
                    <input
                      type="number"
                      className="form-control"
                      min="-1"
                      max="1"
                      step="0.1"
                      value={predInput.social_sentiment_score}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          social_sentiment_score: +e.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label>Contract Remaining (months)</label>
                    <input
                      type="number"
                      className="form-control"
                      min="0"
                      max="48"
                      value={predInput.contract_duration_months}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          contract_duration_months: +e.target.value,
                        }))
                      }
                    />
                  </div>
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Position</label>
                    <select
                      className="form-control"
                      value={predInput.position}
                      onChange={(e) =>
                        setPredInput((p) => ({
                          ...p,
                          position: e.target.value,
                        }))
                      }
                    >
                      <option>Forward</option>
                      <option>Midfielder</option>
                      <option>Defender</option>
                      <option>Goalkeeper</option>
                    </select>
                  </div>
                </div>

                <button
                  className="btn btn-primary"
                  style={{ marginTop: "0.5rem" }}
                  disabled={loading}
                  onClick={handlePredict}
                >
                  {loading ? <Spinner /> : "⚡"} Predict Transfer Value
                </button>

                {predResult && (
                  <div className="prediction-result" style={{ background: 'transparent', padding: 0, border: 'none', boxShadow: 'none' }}>
                    <div className="grid-2">
                       {/* XGBoost Prediction Card */}
                       <div className="card" style={{ background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(236, 72, 153, 0.04))', borderColor: 'var(--border-accent)', textAlign: 'center' }}>
                         <div className="value-label" style={{ fontWeight: 600, color: 'var(--text-sub)', marginBottom: '0.5rem' }}>XGBoost Ensemble Estimate</div>
                         <div className="value-display gradient-text" style={{ fontSize: '2.5rem', marginBottom: '0.2rem', fontWeight: 800 }}>
                            ₹{predResult.xgboost.inr_crores} Cr
                         </div>
                         <div className="value-label" style={{ fontSize: '0.85rem' }}>
                            €{predResult.xgboost.eur_millions}M &nbsp;|&nbsp; ₹{Number(predResult.xgboost.inr_lakhs).toLocaleString('en-IN')} L
                         </div>
                       </div>
                       
                       {/* LSTM Prediction Card */}
                       <div className="card" style={{ background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.05), rgba(124, 58, 237, 0.08))', borderColor: 'var(--border-accent)', textAlign: 'center' }}>
                         <div className="value-label" style={{ fontWeight: 600, color: 'var(--text-sub)', marginBottom: '0.5rem' }}>Multivariate LSTM Estimate</div>
                         {predResult.lstm.inr_crores > 0 ? (
                           <>
                             <div className="value-display gradient-text" style={{ fontSize: '2.5rem', marginBottom: '0.2rem', fontWeight: 800 }}>
                               ₹{predResult.lstm.inr_crores} Cr
                             </div>
                             <div className="value-label" style={{ fontSize: '0.85rem' }}>
                                €{predResult.lstm.eur_millions}M &nbsp;|&nbsp; ₹{Number(predResult.lstm.inr_lakhs).toLocaleString('en-IN')} L
                             </div>
                           </>
                         ) : (
                            <div style={{ marginTop: '1.5rem', fontSize: '0.9rem', color: 'var(--text-sub)' }}>
                               LSTM Model not yet trained.
                            </div>
                         )}
                       </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Info Panel */}
              <div className="card">
                <div className="card-title">ℹ️ Model Architecture</div>
                <div className="card-desc" style={{ marginBottom: "1.5rem" }}>
                  The prediction uses the trained XGBoost ensemble model with 9
                  engineered features including performance trends, injury
                  proxies, and VADER-derived sentiment scores.
                </div>

                <div className="grid-2" style={{ marginBottom: "1.5rem" }}>
                  <MetricCard label="XGBoost Trees" value="150" />
                  <MetricCard label="Max Depth" value="6" />
                  <MetricCard label="LSTM Layers" value="2" />
                  <MetricCard label="Sequence Length" value="3" />
                </div>

                <div>
                  <div
                    style={{
                      marginBottom: "0.5rem",
                      fontWeight: 600,
                      fontSize: "0.9rem",
                      color: "var(--text-sub)",
                    }}
                  >
                    Tech Stack
                  </div>
                  <div
                    style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}
                  >
                    {[
                      "XGBoost",
                      "LSTM",
                      "VADER NLP",
                      "scikit-learn",
                      "FastAPI",
                      "React",
                    ].map((t) => (
                      <span key={t} className="badge badge-info">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>

                <div style={{ marginTop: "1.5rem" }}>
                  <div
                    style={{
                      marginBottom: "0.5rem",
                      fontWeight: 600,
                      fontSize: "0.9rem",
                      color: "var(--text-sub)",
                    }}
                  >
                    Feature Importance
                  </div>
                  <Bar
                    data={{
                      labels: [
                        "Perf Rating",
                        "Sentiment",
                        "Contract",
                        "Goals",
                        "Minutes",
                        "Injuries",
                      ],
                      datasets: [
                        {
                          label: "Importance",
                          data: [0.42, 0.18, 0.15, 0.12, 0.08, 0.05],
                          backgroundColor: [
                            "#818cf8",
                            "#c084fc",
                            "#34d399",
                            "#fbbf24",
                            "#f97316",
                            "#ef4444",
                          ],
                          borderRadius: 5,
                        },
                      ],
                    }}
                    options={{
                      ...chartDefaults,
                      responsive: true,
                      plugins: {
                        ...chartDefaults.plugins,
                        legend: { display: false },
                      },
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
