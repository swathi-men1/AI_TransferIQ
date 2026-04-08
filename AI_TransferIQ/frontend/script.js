document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    const btnText = predictBtn.querySelector(".btn-text");
    const resultEmpty = document.getElementById("result-empty");
    const resultContent = document.getElementById("iq-result");
    const perfSlider = document.getElementById("performance-rating");
    const sentSlider = document.getElementById("sentiment");
    const perfVal = document.getElementById("perf-val");
    const sentVal = document.getElementById("sent-val");

    const riskMap = [
        { max: 6, label: "Stable", note: "Low spread from current valuation", chip: "Conservative move" },
        { max: 15, label: "Balanced", note: "Moderate market sensitivity", chip: "Measured upside" },
        { max: Infinity, label: "Aggressive", note: "High volatility around the estimate", chip: "Momentum scenario" }
    ];

    updateSliderLabels();

    perfSlider.addEventListener("input", updateSliderLabels);
    sentSlider.addEventListener("input", updateSliderLabels);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        setLoadingState(true);
        showPendingState();

        const payload = collectFormData();

        try {
            await wait(850);

            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok || result.error) {
                throw new Error(result.error || "Prediction request failed.");
            }

            renderResult(payload, result);
        } catch (error) {
            console.error(error);
            showErrorState(error.message || "Prediction request failed.");
        } finally {
            setLoadingState(false);
        }
    });

    function updateSliderLabels() {
        perfVal.textContent = `${(Number(perfSlider.value) / 10).toFixed(1)} / 10`;
        sentVal.textContent = `${sentSlider.value}%`;
    }

    function collectFormData() {
        return {
            player_name: document.getElementById("player-name").value.trim(),
            market_value: Number(document.getElementById("market-value").value),
            performance_rating: Number(perfSlider.value) / 10,
            goals_assists: Number(document.getElementById("goals-assists").value),
            minutes_played: Number(document.getElementById("minutes-played").value),
            sentiment: Number(sentSlider.value),
            contract: Number(document.getElementById("contract").value),
            position: document.getElementById("position").value
        };
    }

    function renderResult(payload, result) {
        resultEmpty.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const predictedValue = Number(result.predicted_value || 0);
        const changePercent = Number(result.change_percent || 0);
        const absoluteChange = Math.abs(changePercent);
        const riskProfile = riskMap.find((entry) => absoluteChange <= entry.max) || riskMap[riskMap.length - 1];
        const formPer90 = payload.minutes_played > 0 ? (payload.goals_assists / payload.minutes_played) * 90 : 0;
        const upward = changePercent >= 0;

        document.getElementById("res-player-name").textContent = payload.player_name || "Unnamed player";
        document.getElementById("result-chip").textContent = riskProfile.chip;
        animateNumber("predicted-value", predictedValue, { decimals: 1 });
        animateNumber("trend-val", changePercent, { decimals: 1, suffix: "%" });

        const trendIndicator = document.getElementById("market-trend-indicator");
        document.getElementById("trend-arrow").textContent = upward ? "+" : "-";
        document.getElementById("trend-text").textContent = upward
            ? "Projected increase versus current value"
            : "Projected decline versus current value";

        trendIndicator.classList.toggle("down", !upward);
        document.getElementById("risk-val").textContent = riskProfile.label;
        document.getElementById("confidence-val").textContent = deriveConfidence(payload, absoluteChange);
        document.getElementById("form-output").textContent = formPer90.toFixed(2);
        document.getElementById("scenario-note").textContent = riskProfile.note;

        if (window.innerWidth < 980) {
            document.getElementById("result-display").scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    function deriveConfidence(payload, absoluteChange) {
        const minutesScore = payload.minutes_played >= 1800;
        const outputScore = payload.goals_assists >= 10;
        const sentimentScore = payload.sentiment >= 45 && payload.sentiment <= 80;

        if (minutesScore && outputScore && absoluteChange <= 18 && sentimentScore) {
            return "High";
        }

        if (minutesScore || outputScore) {
            return "Medium";
        }

        return "Exploratory";
    }

    function showPendingState() {
        resultEmpty.classList.remove("hidden");
        resultContent.classList.add("hidden");
        resultEmpty.innerHTML = `
            <p class="result-kicker">Processing</p>
            <h3>Building valuation snapshot</h3>
            <p>We are packaging the current scenario and waiting for the prediction service to return a projection.</p>
        `;
    }

    function showErrorState(message) {
        resultEmpty.classList.remove("hidden");
        resultContent.classList.add("hidden");
        resultEmpty.innerHTML = `
            <p class="result-kicker">Request issue</p>
            <h3>Prediction could not be completed</h3>
            <p>${escapeHtml(message)}</p>
        `;
    }

    function setLoadingState(isLoading) {
        predictBtn.classList.toggle("loading", isLoading);
        btnText.textContent = isLoading ? "Calculating..." : "Calculate valuation";
    }

    function animateNumber(id, targetValue, options = {}) {
        const element = document.getElementById(id);
        const decimals = options.decimals ?? 0;
        const suffix = options.suffix ?? "";
        const duration = options.duration ?? 900;
        const startTime = performance.now();

        function step(currentTime) {
            const progress = Math.min((currentTime - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentValue = targetValue * eased;
            element.textContent = `${currentValue.toFixed(decimals)}${suffix}`;

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                element.textContent = `${targetValue.toFixed(decimals)}${suffix}`;
            }
        }

        requestAnimationFrame(step);
    }

    function wait(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }
});
