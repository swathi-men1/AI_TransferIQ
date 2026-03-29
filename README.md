# TransferIQ — AI Player Transfer Value Prediction

## Quick Start

```bash
pip install -r requirements.txt

# Train all models (full pipeline)
python src/main.py

# Predict a single player
python src/best_model.py predict "Bruno Fernandes"
python src/predict.py --player "Casemiro"

# Run web app
cd webapp
python app.py
# Open: http://localhost:5000
```

## Project Structure

```
TransferIQ/
├── data/
│   ├── raw/                        
│   │   ├── player.csv              
│   │   ├── market_value.csv        
│   │   ├── injury.csv              
│   │   └── sentiment.csv          
│   └── processed/                
│       ├── cleaned_dataset_final.csv
│       ├── featured_dataset_final.csv
│       └── lstm_timeseries_dataset.csv
│
├── models/                         
│   ├── best_model_v2.pkl           
│   ├── weighted_ensemble.pkl       
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   ├── ridge_regression.pkl
│   └── lstm_config.pkl
│
├── src/                            
│   ├── main.py                     # Full pipeline runner
│   ├── data_cleaning.py            
│   ├── feature_engineering.py     
│   ├── lstm_model.py               
│   ├── ensemble_model.py           
│   ├── best_model.py               
│   └── predict.py                  
│
├── webapp/                         # Flask web application
│   ├── app.py                     
│   ├── requirements.txt
│   ├── models/
│   │   └── best_model_v2.pkl
│   ├── data/
│   │   └── featured_dataset_final.csv
│   └── templates/
│       └── index.html              # Full frontend
│
├── outputs/                        # Results, charts, reports
│   ├── charts/                     # 7 visualization charts
│   ├── final_predictions.csv
│   ├── feature_importance_final.csv
│   └── full_model_comparison.csv
│
├── TransferIQ_Presentation.pptx   
├── TransferIQ_Report.docx         # Full project report
├── requirements.txt
└── README.md
```





## Tech Stack

Python 3.12 · NumPy · Scikit-learn · Pandas · Flask · Matplotlib
