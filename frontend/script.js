const API_URL = "http://127.0.0.1:5000/predict";

let forecastChart = null;

// DOM Elements
const form = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const btnText = predictBtn.querySelector('.btn-text');
const loader = predictBtn.querySelector('.loader');
const welcomeState = document.getElementById('welcome-state');
const resultsDashboard = document.getElementById('results-dashboard');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
  form.addEventListener('submit', handlePredict);
});

async function handlePredict(e) {
  e.preventDefault();
  
  // Show loading
  showLoading();
  
  // Get form data with normalization
  const formData = getNormalizedFormData();
  
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: formData.features })
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
      displayResults(data, formData.rawData);
      createForecastChart(formData.rawData);
    } else {
      throw new Error(data.error || 'Prediction failed');
    }
  } catch (error) {
    console.error('Prediction error:', error);
    showError('Failed to connect to ML backend. Ensure Flask server is running on port 5000.');
  } finally {
    hideLoading();
  }
}

function getNormalizedFormData() {
  // Raw form data
  const rawData = {
    player_name: document.getElementById('player_name').value || 'Player',
    position: document.getElementById('position').value,
    performance_rating: parseFloat(document.getElementById('performance_rating').value) || 0,
    goals_assists: parseFloat(document.getElementById('goals_assists').value) || 0,
    prev_market_value: parseFloat(document.getElementById('prev_market_value').value) || 0,
    days_injured: parseFloat(document.getElementById('days_injured').value) || 0,
    social_sentiment_score: parseFloat(document.getElementById('social_sentiment_score').value) || 0,
    contract_duration_months: parseFloat(document.getElementById('contract_duration_months').value) || 36,
    minutes_played: parseFloat(document.getElementById('minutes_played').value) || 2500
  };
  
  // Normalize for backend (scales from new form to match old JS expectations)
  const perf_n = rawData.performance_rating / 10;  // 1-10 → 0-1
  const goals_n = rawData.goals_assists / 50;
  const minutes_n = rawData.minutes_played / 3000;
  const injury_n = rawData.days_injured / 365;
  const contract_n = rawData.contract_duration_months / 60;
  const mv_n = (rawData.prev_market_value / 1000000) / 200;  // € → €M / 200
  const sentiment_n = rawData.social_sentiment_score / 100;  // 1-100 → 0-1
  
  // Position encoding (map new values to old)
  const positionMap = {
    'Forward': 'forward',
    'Midfielder': 'midfielder', 
    'Defender': 'defender',
    'Goalkeeper': 'goalkeeper'
  };
  const pos = positionMap[rawData.position] || 'forward';
  const is_forward = pos === 'forward' ? 1 : 0;
  const is_goalkeeper = pos === 'goalkeeper' ? 1 : 0;
  const is_midfielder = pos === 'midfielder' ? 1 : 0;
  
  // Build exact 27-feature timestep (repeated 3x for LSTM = 81 features)
  function buildTimestep() {
    return [
      0,                            // player_id
      perf_n,                       // performance_rating
      goals_n,                      // goals_assists
      minutes_n,                    // minutes_played
      0,                            // perf_trend_3m
      0,                            // goals_trend_3m
      0,                            // perf_vol_3m
      perf_n,                       // perf_3m_avg
      goals_n,                      // goals_assists_3m_avg
      goals_n / (minutes_n + 0.001), // ga_per_minute
      perf_n,                       // lag_1
      perf_n,                       // lag_2
      injury_n,                     // days_injured
      injury_n,                     // cumulative_days_injured
      injury_n,                     // injury_risk
      injury_n,                     // injury_impact
      contract_n,                   // contract_duration_months
      1 - contract_n,               // contract_urgency
      sentiment_n,                  // social_sentiment_score
      0,                            // sentiment_momentum
      new Date().getMonth() + 1,    // month
      new Date().getFullYear(),     // year
      is_forward,                   // position_Forward
      is_goalkeeper,                // position_Goalkeeper
      is_midfielder,                // position_Midfielder
      mv_n,                         // market_value
      0                             // market_value_trend
    ];
  }
  
  const features = [
    ...buildTimestep(),
    ...buildTimestep(),
    ...buildTimestep()
  ];
  
  return { features, rawData };
}

function displayResults(data, rawData) {
  // Update player info
  document.getElementById('res-player-name').textContent = rawData.player_name;
  document.getElementById('res-position').textContent = rawData.position;
  
  // Format €M values (backend returns normalized, *100 for €M)
  const formatEuro = (val) => `€${(val * 100).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 1})}M`;
  
  document.getElementById('res-ensemble').textContent = formatEuro(data.ensemble_prediction);
  document.getElementById('res-lstm').textContent = formatEuro(data.lstm_prediction);
  document.getElementById('res-sentiment').textContent = `+${formatEuro((data.ensemble_prediction - data.xgboost_prediction))}`; // Dummy sentiment diff
  document.getElementById('res-final-score').textContent = formatEuro(data.ensemble_prediction);
  
  // Show dashboard
  welcomeState.style.display = 'none';
  resultsDashboard.classList.remove('hidden');
}

function createForecastChart(rawData) {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  
  // Dummy multi-step forecast based on ensemble + growth
  const labels = ['Current', 'Window 1', 'Window 2', 'Window 3'];
  const ensembleVal = 50; // Example €50M base
  const dataPoints = [
    ensembleVal,
    ensembleVal * 1.05,
    ensembleVal * 1.12,
    ensembleVal * 1.18
  ];
  
  if (forecastChart) {
    forecastChart.destroy();
  }
  
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Transfer IQ Forecast',
        data: dataPoints,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.2)',
        borderWidth: 4,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#6366f1',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 3,
        pointRadius: 8,
        pointHoverRadius: 10
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(255,255,255,0.1)' },
          ticks: {
            callback: function(value) { return '€' + value.toLocaleString() + 'M'; }
          }
        },
        x: {
          grid: { display: false }
        }
      },
      animation: {
        duration: 1500,
        easing: 'easeOutQuart'
      }
    }
  });
}

function showLoading() {
  welcomeState.style.opacity = '0.5';
  predictBtn.disabled = true;
  btnText.style.opacity = '0';
  loader.classList.remove('hidden');
}

function hideLoading() {
  welcomeState.style.opacity = '1';
  predictBtn.disabled = false;
  loader.classList.add('hidden');
  btnText.style.opacity = '1';
}

function showError(message) {
  hideLoading();
  // Simple alert for now - could enhance with toast
  alert('Error: ' + message);
}
