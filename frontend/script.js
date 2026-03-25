document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultDisplay = document.getElementById('result-display');
    const resultPlaceholder = resultDisplay.querySelector('.result-placeholder');
    const iqResult = document.getElementById('iq-result');
    
    // Sliders
    const perfSlider = document.getElementById('performance-rating');
    const perfVal = document.getElementById('perf-val');
    const sentSlider = document.getElementById('sentiment');
    const sentVal = document.getElementById('sent-val');

    perfSlider.addEventListener('input', (e) => {
        perfVal.textContent = (e.target.value / 10).toFixed(1);
    });

    sentSlider.addEventListener('input', (e) => {
        sentVal.textContent = `${e.target.value}%`;
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI state: Loading
        predictBtn.classList.add('loading');
        predictBtn.querySelector('.btn-text').textContent = 'ANALYZING DATA CLUSTERS...';
        
        // Reset result view
        iqResult.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
        resultPlaceholder.querySelector('p').textContent = 'Running XGBoost Inference...';

        // Simulate "AI" processing time
        await new Promise(resolve => setTimeout(resolve, 2500));

        // Get inputs
        const playerName = document.getElementById('player-name').value;
        const currentVal = parseFloat(document.getElementById('market-value').value);
        const perf = parseInt(perfSlider.value) / 10;
        const goalsAssists = parseInt(document.getElementById('goals-assists').value);
        const minutes = parseInt(document.getElementById('minutes-played').value);
        const sentiment = parseInt(sentSlider.value);
        const contract = parseInt(document.getElementById('contract').value);
        const position = document.getElementById('position').value;

        // REAL API CALL
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    player_name: playerName,
                    market_value: currentVal,
                    performance_rating: perf,
                    goals_assists: goalsAssists,
                    minutes_played: minutes,
                    sentiment: sentiment,
                    contract: contract,
                    position: position
                })
            });

            const result = await response.json();
            
            if (result.error) {
                alert('Engine Error: ' + result.error);
                predictBtn.classList.remove('loading');
                predictBtn.querySelector('.btn-text').textContent = 'INITIALIZE IQ SEQUENCE';
                return;
            }

            const predictedVal = result.predicted_value;
            const changePercent = result.change_percent;

            // UI state: Show Result
            predictBtn.classList.remove('loading');
            predictBtn.querySelector('.btn-text').textContent = 'INITIALIZE IQ SEQUENCE';
            
            resultPlaceholder.classList.add('hidden');
            iqResult.classList.remove('hidden');

            // Update result fields
            document.getElementById('res-player-name').textContent = playerName.toUpperCase();
            
            // Animate numbers
            animateValue('predicted-value', 0, predictedVal, 1500, 1);
            animateValue('trend-val', 0, changePercent, 1500, 1, '%');

            // Trend indicator style
            const trendIndicator = document.getElementById('market-trend-indicator');
            const trendIcon = trendIndicator.querySelector('.trend-icon');
            
            if (changePercent >= 0) {
                trendIndicator.className = 'market-trend trend-up';
                trendIcon.textContent = '↑';
            } else {
                trendIndicator.className = 'market-trend trend-down';
                trendIcon.textContent = '↓';
            }

            // Risk Index
            const riskVal = document.getElementById('risk-val');
            if (Math.abs(changePercent) > 20) {
                riskVal.textContent = 'High Volatility';
                riskVal.style.color = '#ff3e3e';
            } else if (Math.abs(changePercent) > 10) {
                riskVal.textContent = 'Moderate';
                riskVal.style.color = '#ff9f00';
            } else {
                riskVal.textContent = 'Stable';
                riskVal.style.color = '#00ff41';
            }

        } catch (err) {
            console.error(err);
            alert('Valuation Sequence Failed (Server Error)');
            predictBtn.classList.remove('loading');
            predictBtn.querySelector('.btn-text').textContent = 'INITIALIZE IQ SEQUENCE';
        }

        // Scroll to result on mobile
        if (window.innerWidth < 900) {
            resultDisplay.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });

    function animateValue(id, start, end, duration, decimals = 0, suffix = '') {
        const obj = document.getElementById(id);
        const range = end - start;
        let current = start;
        const increment = end > start ? 0.1 : -0.1;
        const stepTime = Math.abs(Math.floor(duration / (range / increment)));
        
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (outQuart)
            const easeProgress = 1 - Math.pow(1 - progress, 4);
            const val = start + range * easeProgress;
            
            obj.textContent = (decimals === 0 ? Math.floor(val) : val.toFixed(decimals)) + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                obj.textContent = (decimals === 0 ? Math.floor(end) : end.toFixed(decimals)) + suffix;
            }
        }
        
        requestAnimationFrame(update);
    }
});
