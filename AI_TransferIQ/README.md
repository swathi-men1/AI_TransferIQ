# TransferIQ

TransferIQ is a football player valuation project built around a Flask inference API and a lightweight frontend workspace. The app lets a user enter a player scenario, send it to an XGBoost-based prediction endpoint, and review a projected market value along with the estimated percentage movement from the current price.

## What the project does

The goal is to make transfer-value experimentation easier. Instead of scanning raw model outputs, the application provides a small interactive interface where a user can enter:

- current market value
- position
- performance rating
- goals and assists
- minutes played
- contract length
- sentiment score

Those inputs are posted to `/predict`, where the backend prepares the feature vector expected by the trained model and returns:

- `predicted_value`
- `change_percent`
- `log_return`

## Project structure

```text
AI_TransferIQ-AI_TransferIQ_Karthikeya/
|- app.py
|- requirements.txt
|- README.md
|- frontend/
|  |- index.html
|  |- styles.css
|  |- script.js
|- data/
|- models/
|- metrics/
|- notebooks/
```

## Backend summary

The Flask server in [app.py](C:\Users\teluk\OneDrive\Desktop\AI_TransferIQ-AI_TransferIQ_Karthikeya\app.py) does three main jobs:

1. serves the static frontend from the `frontend/` directory
2. accepts JSON requests on `/predict`
3. loads the trained XGBoost model and scaler, prepares features, and returns a prediction response

The current prediction flow includes a few derived values such as `form`, current month, current year, and one-hot position flags.

## Frontend summary

The frontend in [index.html](C:\Users\teluk\OneDrive\Desktop\AI_TransferIQ-AI_TransferIQ_Karthikeya\frontend\index.html), [styles.css](C:\Users\teluk\OneDrive\Desktop\AI_TransferIQ-AI_TransferIQ_Karthikeya\frontend\styles.css), and [script.js](C:\Users\teluk\OneDrive\Desktop\AI_TransferIQ-AI_TransferIQ_Karthikeya\frontend\script.js) is a static interface with three responsibilities:

1. collect player scenario inputs
2. send a request to the Flask API
3. present the returned valuation in a readable dashboard

The current UI is intentionally simple to run locally: no frontend framework, no bundler, and no additional build step.

## Local setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Notes about the current codebase

- The frontend expects the backend to be available on the same origin and to expose `/predict`.
- The backend currently loads model assets from absolute paths inside [app.py](C:\Users\teluk\OneDrive\Desktop\AI_TransferIQ-AI_TransferIQ_Karthikeya\app.py). If you move the project to another machine, those paths will likely need to be updated.
- The repository contains notebooks and model artifacts that appear to support training, preprocessing, and evaluation, but the production app path is the Flask server plus the static frontend.

## Suggested next improvements

- replace absolute model paths with project-relative paths
- add validation and tests for the prediction endpoint
- document the expected schema of the training data
- expose confidence or uncertainty from the model more explicitly

## License

See [LICENSE](C:\Users\teluk\OneDrive\Desktop\AI_TransferIQ-AI_TransferIQ_Karthikeya\LICENSE).
