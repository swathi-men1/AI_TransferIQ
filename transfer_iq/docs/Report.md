<div align="center">

# CT University, Ludhiana

## School of Engineering and Technology

## Department of Computer Science and Engineering

# Mid Term Report

## On

# "TransferIQ: AI-Driven Football Transfer Valuation System for Football Player Transfer Fee Prediction"

Submitted in partial fulfilment of the requirements

for the award of the degree of

**Bachelor of Technology**

in

**Computer Science and Engineering**

to

**School of Engineering and Technology**  
**CT University, Ludhiana**

</div>

**Submitted By:**

- **Name:** ______________________
- **Roll No.:** ______________________
- **Semester:** ______________________
- **Batch:** ______________________

**Under the Guidance of**

**______________________**

Department of Computer Science and Engineering  
School of Engineering and Technology  
CT University, Ludhiana, Punjab, India - 142024

\newpage

---

# Formatting Note

This markdown file is the master content copy for the mid-term report. When preparing the final print-ready Word or Google Docs version, apply the following formatting captured from the university template and current report notes:

- **Font:** Times New Roman
- **Font size:** 12 pt body text
- **Line spacing:** 1.5
- **Left margin:** 3.5 cm
- **Top margin:** 2.5 cm
- **Right margin:** 1.25 cm
- **Bottom margin:** 1.25 cm
- **Preliminary pages:** Roman numbering
- **Main chapters:** Arabic numbering

\newpage

---

# Certificate / Declaration

This is to certify that the project report entitled **"TransferIQ: AI-Driven Football Transfer Valuation System for Football Player Transfer Fee Prediction"** is a bonafide record of work carried out by:

- **Name:** ______________________
- **Roll No.:** ______________________
- **Semester:** ______________________
- **Batch:** ______________________

under my guidance during the academic session _____________ in partial fulfilment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering** from **CT University, Ludhiana**.

The work presented in this report is original to the best of our knowledge and has not been submitted elsewhere for any other degree, diploma, or academic award.

**Guide / Supervisor Signature:** ______________________  
**Name of Guide:** ______________________  
**Designation:** ______________________  
Department of Computer Science and Engineering  
CT University, Ludhiana

**Date:** ______________________  
**Place:** Ludhiana

\newpage

---

# Acknowledgement

The successful progress of this project has been possible because of the support, direction, and encouragement received from faculty members, mentors, peers, and the institution. I would like to express sincere gratitude to my project guide for continuous guidance, constructive feedback, and academic motivation throughout the development of this work.

I also extend my thanks to the Department of Computer Science and Engineering, CT University, Ludhiana, for providing the academic environment and resources necessary to undertake this project. The department's emphasis on practical implementation, experimentation, and project-based learning has played an important role in shaping this work.

Special appreciation is also due to classmates, friends, and all those who contributed directly or indirectly through discussion, review, and encouragement during dataset preparation, model experimentation, interface design, and documentation.

Finally, I acknowledge the open-source software ecosystem and public technical resources that made it possible to study, build, and evaluate an AI-assisted football transfer valuation pipeline in a reproducible way.

\newpage

---

# Abstract

TransferIQ is an AI-driven football transfer valuation system developed to estimate the transfer fee or transfer value of a football player by integrating structured player statistics, market indicators, injury burden, contract context, club movement signals, and sentiment-oriented features. Football transfer valuation is inherently multi-factorial. Real-world decisions are influenced by sporting performance, age profile, injury record, popularity, contract duration, club prestige, transfer timing, and recent public perception. As a result, static heuristics or manual judgment alone often fail to capture the dynamic and data-rich character of modern football valuation.

The purpose of TransferIQ is to provide a data-driven computational framework that can transform heterogeneous football-related data into predictive insights. The repository contains an end-to-end implementation that begins from curated raw data and proceeds through preprocessing, feature engineering, model training, evaluation, persisted artefacts, and user-facing inference interfaces. The system combines conventional machine learning with sequence-oriented deep learning to compare multiple predictive strategies under a common workflow.

The project currently includes a structured-feature **XGBoost regressor**, a multivariate **LSTM**, a **univariate LSTM**, and an **encoder-decoder LSTM** for short-horizon forecasting. It also includes an **ensemble mechanism** that blends tree-based and sequence-based outputs. According to the saved training metadata in the repository, the present modelling workflow uses **961 modelling rows** derived from **1,989 engineered rows**, with **104 engineered columns** and **109 transformed model features**. The current training configuration uses a **sequence length of 8** and a **forecast horizon of 3**.

The saved evaluation snapshot indicates that the **XGBoost model is currently the strongest standalone model** in practical terms. Its saved performance is approximately **RMSE = 2,601,882.75**, **MAE = 860,455.62**, and **R2 = 0.3530**. The saved ensemble configuration is heavily **XGBoost-dominant** with a weight of **0.9** for XGBoost and **0.1** for the LSTM branch, reflecting that the structured-feature model presently captures the available dataset more effectively than the sequence-oriented models. The LSTM, univariate LSTM, and encoder-decoder LSTM are fully implemented and evaluated, but they underperform on the current split.

An important component of TransferIQ is its sentiment subsystem. The project includes a dedicated sentiment analysis pipeline that prefers **VADER**, uses **TextBlob** when needed, and falls back to a deterministic lexicon-based scorer if optional libraries are unavailable. This keeps the project executable across varying academic evaluation environments while still demonstrating the intended natural language processing component.

The repository further includes two inference surfaces: a command-line backend predictor and a Streamlit-based interactive dashboard. The dashboard supports player search, custom valuation scenarios, confidence-aware prediction ranges, bulk scanning, and model intelligence views. Taken together, these features make TransferIQ a complete academic prototype demonstrating applied AI, feature engineering, predictive modelling, inference engineering, and interactive analytics in the context of football transfer markets.

\newpage

---

# Table of Contents

**To be generated automatically in the final Word or Google Docs version.**

Suggested major entries:

1. Certificate / Declaration
2. Acknowledgement
3. Abstract
4. Introduction
5. Problem Statement and Objectives
6. Existing Work Completed
7. System Architecture and Project Structure
8. Dataset Description and Data Preparation
9. Feature Engineering and Sentiment Pipeline
10. Model Development
11. Implementation Details
12. Results and Evaluation
13. Dashboard and User Workflow
14. Current Limitations
15. Planned Work / Timeline
16. Conclusion
17. References
18. Appendices

\newpage

---

# List of Figures

**To be generated in the final document.**

Recommended figures for the final formatted version:

- Figure 1. Overall TransferIQ workflow
- Figure 2. High-level system architecture
- Figure 3. Data preparation and feature engineering pipeline
- Figure 4. Model training and evaluation flow
- Figure 5. Prediction and inference workflow
- Figure 6. Streamlit dashboard module layout
- Figure 7. Top XGBoost feature importance chart
- Figure 8. Model comparison visual summary

\newpage

---

# List of Tables

**To be generated in the final document.**

Recommended tables for the final formatted version:

- Table 1. Repository structure summary
- Table 2. Source dataset inventory
- Table 3. Processed dataset inventory
- Table 4. Feature category summary
- Table 5. Model configuration summary
- Table 6. Training metadata snapshot
- Table 7. Evaluation metrics comparison
- Table 8. Timeline for remaining work

\newpage

---

# 1. Introduction

## 1.1 Background

Football player transfers represent one of the most financially visible decision-making processes in modern sport. A transfer fee is not determined only by recent goals or general popularity. It is shaped by several interacting variables including long-term performance, age and career stage, injury risk, current market value, club reputation, contract duration, playing history, competition strength, and media sentiment. In practice, clubs, analysts, and media platforms often rely on a combination of scouting reports, experience, estimated market heuristics, and public discussion when assessing likely player values.

With the growth of data science and sports analytics, there is increasing scope for computational systems that estimate player transfer values using transparent, reproducible, and multi-source signals. Such a system is particularly relevant in football because the transfer market is dynamic, high-value, and affected by short-term events such as exceptional form, injury setbacks, contract expiry pressure, and sudden surges in public attention.

TransferIQ has been developed in this context as a practical AI project that combines data engineering, predictive modelling, sentiment analysis, and interactive analytics. The project aims not merely to create a model, but to provide an end-to-end workflow that starts with curated football data and ends with usable transfer-value predictions and comparative model insights.

## 1.2 Need for the Project

The need for TransferIQ arises from the limitations of static or purely manual valuation practices. A player's transfer value can fluctuate because of:

- changes in goal and assist output
- recurring injuries and missed matches
- contract risk and time remaining on the deal
- movement between clubs and leagues
- differences in competition quality
- changing public and media perception
- market pressure and timing within transfer windows

Traditional analysis may observe some of these signals qualitatively, but it is difficult to combine them consistently at scale. A machine learning based system can provide a more systematic framework for aggregating and learning from these indicators.

## 1.3 Project Scope

TransferIQ is designed as a prototype system focused on football player transfer valuation. It is not intended to replace expert scouting, legal negotiation, or full club strategy. Instead, it provides a decision-support style estimate using the data available in the repository. The current scope includes:

- building a curated football transfer modelling dataset
- engineering structured, temporal, and sentiment-aware features
- training multiple predictive model families
- comparing model performance on saved evaluation metrics
- exposing prediction workflows through a backend script and a dashboard
- supporting optional external data collection workflows for future expansion

## 1.4 Academic and Technical Relevance

From an academic point of view, the project is relevant because it brings together several important themes of modern computer science:

- machine learning for regression and forecasting
- feature engineering for tabular and temporal data
- natural language sentiment analysis
- software engineering for reusable pipelines
- dashboard-based presentation of model outputs
- reproducibility through saved artefacts and scripts

This combination makes TransferIQ suitable for industrial training, project-based learning, and academic reporting in artificial intelligence, data science, and applied software engineering.

## 1.5 Report Roadmap

This report is organized to first describe the problem and project objectives, then document the work already completed in the repository, explain the system architecture and datasets, discuss feature engineering and model development, present implementation details and saved evaluation results, and finally summarize current limitations and planned future work.

\newpage

---

# 2. Problem Statement and Objectives

## 2.1 Problem Statement

The core problem addressed by this project is the difficulty of estimating football player transfer value in a consistent, data-driven, and explainable manner. A player's true transfer value depends on more than a single numerical statistic. It is influenced by structured on-field performance, market context, injury burden, contract status, transfer timing, and public sentiment. Manual analysis can become inconsistent, while single-source data models may ignore important variables.

Therefore, the problem can be stated as follows:

**To design and implement an AI-based system that predicts football player transfer value using a combination of structured performance data, contextual football information, injury-related indicators, contract and market factors, and sentiment-aware features, while also providing reproducible workflows and user-facing prediction interfaces.**

## 2.2 Objectives of the Project

The objectives of TransferIQ are:

1. To create a repository-based end-to-end system for football transfer value prediction.
2. To integrate multiple football-related data sources into a unified modelling dataset.
3. To design a robust preprocessing and feature engineering pipeline that transforms raw transfer records into meaningful model-ready features.
4. To implement multiple predictive approaches, including XGBoost and LSTM-based models.
5. To compare model performance using regression metrics such as RMSE, MAE, R2, sMAPE, and non-zero RMSE.
6. To include a sentiment analysis component so that perception-oriented signals can contribute to valuation.
7. To build practical inference interfaces for both terminal and dashboard usage.
8. To preserve artefacts, metadata, and processed datasets for reproducibility and academic documentation.

## 2.3 Success Criteria

The repository can be considered successful at the mid-term stage if it demonstrates the following:

- a working raw-to-featured data flow
- trained model artefacts saved to disk
- reproducible training scripts
- prediction capability on user-provided or sample player profiles
- comparative evaluation across model families
- presentable output through an interactive dashboard
- enough structure and evidence to support academic reporting

## 2.4 In-Scope and Out-of-Scope Considerations

### In Scope

- football player value estimation based on available project data
- structured and sequence model comparison
- sentiment-aware feature integration
- backend and dashboard prediction workflows
- external data collection scripts at the implementation level

### Out of Scope for the Current Mid-Term Stage

- real-time production deployment for live club use
- guaranteed access to all live APIs or proprietary transfer platforms
- official club-grade valuation methodology
- fully automated ETL scheduling in production
- advanced causal inference about transfer economics

\newpage

---

# 3. Existing Work Completed

## 3.1 Repository Status at Mid-Term Stage

The TransferIQ repository has already progressed well beyond the ideation stage. It contains coordinated modules for data preparation, feature engineering, sentiment analysis, model training, model persistence, prediction, and dashboard presentation. This indicates that the project is a functioning academic prototype rather than a proposal or a set of disconnected experimental notebooks.

The major areas already completed are:

- dataset preparation and curated modelling assets
- preprocessing and engineered feature generation
- comparative model development
- saved training metadata and feature importances
- command-line prediction backend
- Streamlit dashboard application
- external data bootstrap and collection utilities
- project documentation and configuration files

## 3.2 Completed Implementation Areas

### 3.2.1 Data Assets

The project already includes both original football-related CSV resources and a curated raw modelling dataset. The `Dataset/` folder contains broad source files such as player profiles, injuries, market values, club details, competition-season statistics, national performances, and sentiment records. In addition, the main modelling dataset used by the training workflow is stored in `transfer_iq/data/raw/transfer_prediction_with_sentiment_cleaned.csv`.

### 3.2.2 Processed Data Outputs

The repository already contains processed outputs that show the feature engineering pipeline has been executed:

- `enhanced_transfer_dataset.csv` with 1,989 rows and 83 columns
- `advanced_transfer_features.csv` with 1,989 rows and 104 columns
- `optimized_transfer_dataset.csv` with 1,989 rows and 27 columns
- `optimized_players_final.csv` with 83 rows and 35 columns
- `player_prediction_library.csv` with 961 rows and 25 columns
- `test_predictions.csv` with 145 rows and 112 columns

These artefacts are strong evidence that data transformation, feature creation, modelling preparation, and evaluation export have already been completed within the repository.

### 3.2.3 Core Modelling Logic

The central implementation file `src/transfer_value_system.py` includes utility functions, neural network builders, a feature engineering class, a training class, and a prediction class. This means the project does not rely on isolated scripts alone. Instead, reusable program logic is encapsulated into defined classes and workflows.

### 3.2.4 Sentiment Module

The file `src/sentiment_pipeline.py` provides a dedicated sentiment analysis utility. This is already integrated into the project rather than being left as future work. The module supports three layers of sentiment handling:

- VADER when available
- TextBlob when VADER is unavailable
- a fallback lexicon-based scorer when optional dependencies are not present

### 3.2.5 Training and Evaluation Workflow

The file `scripts/train_transfer_models.py` already trains the system through the `TransferValueTrainer` class with a sequence length of 8 and prints a training summary. The repository also stores:

- trained XGBoost model
- trained multivariate LSTM model
- trained univariate LSTM model
- trained encoder-decoder LSTM model
- ensemble weight artefact
- preprocessing bundle
- feature column metadata
- training summary JSON
- XGBoost feature importance CSV

### 3.2.6 Inference and Presentation Layer

The command-line predictor in `app/backend_predict.py` supports:

- CSV-driven prediction
- sample-library prediction
- manual player entry through command-line arguments
- readable printing of current value, predictions, deltas, and confidence
- dual-currency display in EUR and INR

The Streamlit dashboard in `app/app.py` provides three user-facing sections:

- AI Player Lab
- Bulk Scan
- Model Intelligence

This indicates the repository is ready not only for experimentation but also for interactive project demonstration.

## 3.3 Why the Completed Work Matters

The already completed work is significant because it demonstrates a full project lifecycle:

- data collection and curation
- preprocessing and feature engineering
- model training and comparison
- inference engineering
- analytics presentation
- academic documentation support

As a result, the project already has enough depth to justify detailed mid-term reporting and a meaningful final improvement phase.

\newpage

---

# 4. System Architecture and Project Structure

## 4.1 High-Level Architecture

TransferIQ follows a layered pipeline architecture. At a high level, the workflow can be described in five connected stages:

1. **Data Layer**: source football datasets and curated raw modelling data
2. **Processing Layer**: cleaning, parsing, transformation, and feature engineering
3. **Model Layer**: XGBoost, LSTM, univariate LSTM, encoder-decoder LSTM, and ensemble logic
4. **Inference Layer**: predictor loading, transformation of user input, prediction generation, and confidence scoring
5. **Presentation Layer**: terminal outputs and Streamlit dashboard interface

> **Figure Callout:** Figure 1 in the final document should show the above pipeline from raw datasets to dashboard inference.

## 4.2 Architectural Components

### 4.2.1 Data Layer

The project stores source datasets in the top-level `Dataset/` folder and the main raw modelling table inside `transfer_iq/data/raw/`. This separation is useful because it distinguishes broad source resources from the curated dataset actually used by the training and prediction workflows.

### 4.2.2 Processing Layer

The processing layer is primarily implemented in `src/transfer_value_system.py`. It includes:

- reproducibility helpers
- date and list parsing
- safe log transformations
- metric computation
- feature construction
- preprocessing for categorical and numerical data
- sequence preparation for LSTM models

### 4.2.3 Model Layer

The model layer contains:

- an XGBoost regression workflow for structured features
- multivariate LSTM for sequence windows across transformed features
- univariate LSTM for target-history modelling
- encoder-decoder LSTM for short-horizon forecasting
- ensemble weighting for combined predictions

### 4.2.4 Inference Layer

The inference layer is handled by the `TransferValuePredictor` class. It loads persisted preprocessing objects and trained models, transforms new player data, applies prediction logic, generates ensemble values, adds forecast columns, and estimates confidence.

### 4.2.5 Presentation Layer

The terminal interface is provided by `app/backend_predict.py`, while the dashboard interface is delivered through `app/app.py` using Streamlit and Plotly. This layer is important because it demonstrates how the trained system can be consumed by users rather than remaining an offline modelling exercise.

## 4.3 Repository Structure

The current repository is structured as follows:

```text
AI-TransferIQ/
|-- Dataset/
|-- transfer_iq/
|   |-- app/
|   |-- config/
|   |-- data/
|   |-- docs/
|   |-- models/
|   |-- scripts/
|   |-- src/
|   |-- README.md
|   |-- setup.py
|-- transfer_iq_env/
|-- .gitattributes
|-- .gitignore
```

Inside the main application package:

```text
transfer_iq/
|-- app/
|   |-- app.py
|   |-- backend_predict.py
|-- config/
|   |-- config.yaml
|   |-- requirements.txt
|-- data/
|   |-- raw/
|   |-- processed/
|-- docs/
|   |-- Report.md
|-- models/
|   |-- metadata/
|   |-- preprocessing/
|   |-- trained/
|-- scripts/
|   |-- bootstrap_external_services.py
|   |-- collect_external_data.py
|   |-- train_transfer_models.py
|-- src/
|   |-- sentiment_pipeline.py
|   |-- transfer_value_system.py
|-- README.md
|-- setup.py
```

> **Table Callout:** Table 1 in the final document should summarize each major folder and its purpose.

## 4.4 Role of Important Files

- `README.md` explains the project goals, repository structure, setup steps, and current evaluation snapshot.
- `src/transfer_value_system.py` is the core implementation for feature engineering, training, and prediction.
- `src/sentiment_pipeline.py` provides reusable sentiment scoring logic.
- `scripts/train_transfer_models.py` starts the training process.
- `app/backend_predict.py` offers command-line inference.
- `app/app.py` delivers the Streamlit dashboard.
- `scripts/collect_external_data.py` handles optional external data workflows.
- `scripts/bootstrap_external_services.py` prepares local external collection folders and environment scaffolding.
- `models/metadata/training_summary.json` stores metrics, feature counts, weights, and training summary details.

\newpage

---

# 5. Dataset Description and Data Preparation

## 5.1 Source Dataset Inventory

The project includes a broad collection of football-related source files in the `Dataset/` folder. Verified dataset shapes from the repository are shown below.

| File Name | Rows | Columns | Description |
|---|---:|---:|---|
| `team_details.csv` | 2,175 | 12 | Club metadata and competition details |
| `team_competitions_seasons.csv` | 58,247 | 29 | Club-season competition statistics |
| `player_sentiment_data.csv` | 23,868 | 11 | Sentiment and engagement records |
| `player_profiles.csv` | 92,671 | 34 | Player identity and profile data |
| `player_national_performances.csv` | 92,701 | 9 | National team performance records |
| `player_market_value.csv` | 901,429 | 3 | Historical market value observations |
| `player_injuries.csv` | 143,195 | 7 | Injury history and missed-time data |

These files collectively provide background context about clubs, competitions, player profiles, injuries, and market evolution. They are useful for curation, enrichment, and academic explanation of the data landscape behind the project.

## 5.2 Main Raw Modelling Dataset

The principal dataset used by the current training and inference workflow is:

- `transfer_iq/data/raw/transfer_prediction_with_sentiment_cleaned.csv`

This file contains **1,989 rows** and **29 columns**. It is the curated raw modelling dataset that integrates player identity, club information, transfer details, market value, performance fields, injury measures, and sentiment-related variables. A compressed copy of the same dataset is also stored as `transfer_prediction_with_sentiment_cleaned.csv.gz`.

Typical columns represented in the raw modelling data include:

- `player_name`
- `current_club_name`
- `contract_expires`
- `seasons`
- `competitions`
- `clubs`
- `total_goals`
- `total_assists`
- `current_market_value`
- `total_injuries`
- `total_days_missed`
- `transfer_date`
- `from_team_name`
- `to_team_name`
- multiple sentiment-oriented features such as mention volume and sentiment ratios

## 5.3 Processed and Derived Data Assets

The project contains several processed data files generated by the implemented pipelines:

| File Name | Rows | Columns | Purpose |
|---|---:|---:|---|
| `enhanced_transfer_dataset.csv` | 1,989 | 83 | intermediate enhanced feature table |
| `advanced_transfer_features.csv` | 1,989 | 104 | main engineered dataset |
| `optimized_transfer_dataset.csv` | 1,989 | 27 | reduced transfer-focused dataset |
| `optimized_players_final.csv` | 83 | 35 | compact player-oriented table |
| `player_prediction_library.csv` | 961 | 25 | player library used in inference and dashboard workflows |
| `test_predictions.csv` | 145 | 112 | saved holdout predictions and evaluation output |

These processed files show that the system already has a reproducible data pipeline capable of converting raw modelling data into richer feature sets and prediction-ready libraries.

## 5.4 Data Preparation Logic

The repository prepares data through a combination of cleaning, parsing, enrichment, and transformation. The general steps include:

1. reading the curated raw transfer dataset
2. standardizing columns and filling missing values
3. parsing date-like fields such as transfer date and contract expiry
4. parsing list-style text fields such as seasons, competitions, and clubs
5. deriving counts, ratios, logs, and composite football indicators
6. integrating sentiment-aware features from structured and optional text-based sources
7. encoding categorical variables and scaling numerical variables for modelling

## 5.5 Data Preparation Challenges

Working with football transfer data creates several practical challenges:

- list-like fields may be stored as strings and require safe parsing
- date fields may appear in mixed formats
- sentiment information may be sparse or inconsistent
- not all players have equally rich histories
- injury counts and missed days need meaningful summarization
- different feature types must coexist in one modelling workflow

The project addresses these issues through safe utility functions and robust fallback behavior implemented in the core system.

## 5.6 Relevance of the Dataset Design

The dataset design is important because it moves beyond a single-source statistical table. It brings together performance, contextual, temporal, injury, market, and sentiment information, which aligns with the real-world complexity of football transfer valuation.

> **Figure Callout:** Figure 2 in the final document should illustrate the flow from source files to the curated raw modelling dataset and then to engineered outputs.

\newpage

---

# 6. Feature Engineering and Sentiment Pipeline

## 6.1 Importance of Feature Engineering

Machine learning performance depends not only on model selection but also on how raw domain information is represented. In football valuation, raw columns such as goals, assists, injuries, and contract dates are useful, but they become more informative when transformed into ratios, temporal indicators, burden indices, stage categories, and interaction-driven signals. TransferIQ addresses this through a dedicated feature engineering workflow in the `TransferFeatureBuilder` class.

## 6.2 Core Utility Support

The core system includes utility functions for:

- global seed initialization for reproducibility
- safe parsing of list-like fields
- safe conversion of date-like values
- stable logarithmic transformation
- consistent metric calculation for regression and forecasting

These utilities are important because they reduce fragility in preprocessing and make the training and prediction flows more stable.

## 6.3 Major Feature Categories

The implemented feature engineering logic covers several major categories.

### 6.3.1 Performance Features

The system derives football performance features from goals, assists, and historical participation style. Examples include:

- goal contributions
- goals per season
- assists per season
- contribution-based ratios
- performance index style transformations
- log-scaled performance features

### 6.3.2 Injury Features

Injury information is converted into more meaningful measures rather than being left as raw counts alone. The system includes concepts such as:

- total injuries
- total days missed
- injury burden index
- injury-based risk interpretation
- recent event support

These features are important because injury history can affect transfer negotiations, perceived reliability, and future availability.

### 6.3.3 Contract and Transfer Context Features

The project constructs contract-oriented signals such as:

- contract days remaining
- contract status categories
- expiry-related pressure
- transfer timing indicators
- transfer year, month, and quarter
- transfer window classification

These variables are especially important in football transfer economics because contract risk can significantly influence negotiation power and final transfer fees.

### 6.3.4 Career and Club Context Features

TransferIQ also models player movement and contextual football environment through features such as:

- number of seasons
- number of clubs
- number of competitions
- club prestige score
- competition score
- team transition category
- career stage

These help the system distinguish between developing prospects, established players, and high-mobility profiles.

### 6.3.5 Market Features

Market signals are central to the project. Examples include:

- current market value
- market value log
- market value per season
- market value per contribution
- market pressure indicators

The saved feature importance output shows that market-related features are among the most influential variables in the XGBoost model.

### 6.3.6 Sentiment Features

The project includes structured sentiment variables and text-derived sentiment support. Derived sentiment-oriented features include:

- sentiment composite
- sentiment momentum
- sentiment stability
- buzz and mention patterns
- engagement metrics
- positive and negative sentiment ratios
- event-related flags

## 6.4 Sentiment Pipeline Design

The file `src/sentiment_pipeline.py` contains a `TransferSentimentAnalyzer` class and a `SentimentResult` data structure. The sentiment workflow is intentionally designed to remain functional across environments with different dependency availability.

The sentiment backends are prioritized as follows:

1. **VADER** for social-media-style polarity scoring when installed
2. **TextBlob** when VADER is unavailable
3. **Fallback lexicon-based scorer** when neither optional package is available

This layered design is academically valuable because it shows both ambition and practical robustness. The project does not fail entirely when optional NLP libraries are absent.

## 6.5 Output of the Sentiment Module

For a given text input, the sentiment module returns:

- compound score
- scaled compound score
- positive ratio
- negative ratio
- token count
- magnitude
- source backend label

The helper function `sentiment_features_from_text()` converts these outputs into a dictionary suitable for integration into broader feature engineering.

## 6.6 Why Feature Engineering is a Major Strength of the Project

One of the strongest aspects of TransferIQ is that it does not treat football transfer valuation as a simple prediction from raw totals. Instead, it constructs a richer representation of player value drivers. This design better reflects the fact that transfer decisions are shaped by interacting indicators rather than isolated statistics.

> **Table Callout:** Table 4 in the final document should summarize feature categories with examples and justification.  
> **Figure Callout:** Figure 3 should show the preprocessing and feature engineering pipeline.

\newpage

---

# 7. Model Development

## 7.1 Model Development Strategy

The modelling strategy in TransferIQ is intentionally comparative. Rather than assuming that one algorithm family will dominate, the project implements multiple approaches and evaluates them on the saved dataset. This is good academic practice because it allows the report to discuss what performs best and why.

## 7.2 XGBoost Regression Model

The structured-feature baseline and current strongest model is XGBoost. It is suitable for tabular data with mixed engineered signals and non-linear feature relationships. In this project, XGBoost benefits from the rich engineered representation produced by the preprocessing pipeline.

Reasons for using XGBoost include:

- strong performance on structured tabular data
- ability to model non-linear interactions
- practical feature importance extraction
- robustness for moderate-size datasets

The current saved results show that XGBoost is the best standalone model in the repository.

## 7.3 Multivariate LSTM Model

The multivariate LSTM is designed to learn from chronological windows of transformed feature sequences. It is included because football valuation may contain temporal structure, especially when player form, sentiment, and event history are viewed over time.

In the present saved snapshot, the multivariate LSTM is implemented and functional but does not outperform the structured XGBoost model. This is an important result, not a weakness of documentation. It shows honest evaluation and comparative analysis.

## 7.4 Univariate LSTM Model

The univariate LSTM is a simplified sequence model focused more narrowly on target-history style prediction. Its role is to test whether a simpler sequential approach can capture value dynamics without relying on the full multivariate feature space.

At present, it underperforms relative to XGBoost and also trails the multivariate LSTM in the saved metrics.

## 7.5 Encoder-Decoder LSTM

The encoder-decoder LSTM is included for short-horizon forecasting. It is intended to support multi-step future valuation prediction rather than only a single target estimate. In the current system, it uses a forecast horizon of 3.

Although it does not currently outperform the tree-based baseline, its inclusion demonstrates more advanced experimentation with sequence-to-sequence design and future value forecasting.

## 7.6 Ensemble Logic

The project also includes an ensemble mechanism. The purpose of the ensemble is to combine the strengths of the structured model and the sequence model. However, the saved training summary shows:

- `ensemble_weight_xgb = 0.9`
- `ensemble_weight_lstm = 0.1`

This means the ensemble is strongly dominated by XGBoost in the current saved run. Therefore, the ensemble should be interpreted as a blended extension of the best structured model rather than the primary headline model.

## 7.7 Training Metadata Snapshot

The current saved metadata reports:

| Parameter | Value |
|---|---:|
| Target column | `value_at_transfer` |
| Modelling rows | 961 |
| Full engineered rows | 1,989 |
| Engineered columns | 104 |
| Transformed feature count | 109 |
| Sequence length | 8 |
| Forecast horizon | 3 |

## 7.8 Literature and Conceptual Context

The modelling choices are consistent with broader machine learning and sports analytics literature:

- **sports analytics** supports the use of statistical and contextual data for player and team evaluation
- **gradient boosting** methods such as XGBoost are widely used for tabular prediction tasks
- **LSTM networks** are commonly used for sequential and temporal modelling
- **ensemble methods** aim to combine model strengths
- **sentiment analysis** can enrich structured prediction by incorporating perception and discourse signals

This project does not claim to solve football economics completely, but it applies these concepts in a coherent practical system.

> **Figure Callout:** Figure 4 should show the relationship between structured modelling, sequential modelling, and ensemble prediction.

\newpage

---

# 8. Implementation Details

## 8.1 Core Source File: `transfer_value_system.py`

The file `src/transfer_value_system.py` acts as the backbone of the project. Its role is broader than a standard model script because it contains:

- helper functions for reproducibility and parsing
- model-building functions for neural architectures
- the `TransferFeatureBuilder` class
- the `TransferValueTrainer` class
- the `TransferValuePredictor` class

This file effectively centralizes the shared logic needed across training and inference stages.

## 8.2 Feature Builder Implementation

The `TransferFeatureBuilder` class converts raw player-transfer data into engineered features suitable for training and prediction. This includes:

- handling missing columns gracefully
- generating sentiment source text
- parsing season, competition, and club lists
- deriving counts and ratios
- constructing temporal features
- applying logarithmic transforms
- deriving prestige, burden, and market indicators

Because the same logic can be used during both training and inference, the system reduces train-predict mismatch.

## 8.3 Training Class Implementation

The `TransferValueTrainer` class manages:

- loading the raw modelling dataset
- feature engineering
- preprocessing transformations
- train-test workflow
- model fitting across model families
- metric calculation
- model and metadata persistence

The training runner script initializes this trainer with `sequence_length=8`, which matches the saved metadata used in this report.

## 8.4 Predictor Implementation

The `TransferValuePredictor` class is responsible for deployed-style prediction behavior. Its functions include:

- loading saved preprocessing bundles
- loading trained model artefacts
- transforming new rows into the required model space
- producing XGBoost predictions
- producing LSTM-based predictions where sequence context is sufficient
- generating ensemble predictions
- adding encoder-decoder forecast values
- estimating confidence scores
- applying business-rule style protection against unrealistic inflation in injury-heavy cases

The confidence handling is important from a usability perspective because it communicates uncertainty rather than presenting predictions as absolute truth.

## 8.5 Command-Line Inference Flow

The backend predictor in `app/backend_predict.py` supports three input modes:

1. prediction from a supplied CSV file
2. prediction from the saved sample library
3. prediction from manual arguments representing a custom player profile

It then calls the shared `TransferValuePredictor`, formats the outputs, and displays:

- current market value
- XGBoost prediction
- LSTM prediction
- ensemble prediction
- delta against current value
- confidence

Another practical detail is that the backend formats outputs in both **EUR** and **INR**, which improves readability for local academic demonstrations.

## 8.6 Dashboard Implementation

The dashboard in `app/app.py` is more than a simple wrapper around model outputs. It includes:

- a custom visual layout and landing experience
- player-level exploration
- editable valuation scenario controls
- bulk upload and batch prediction mode
- model intelligence charts and metrics
- confidence-aware display logic
- Plotly-based visualization

The three major workspaces are:

- **AI Player Lab** for individual exploration
- **Bulk Scan** for batch predictions
- **Model Intelligence** for inspecting metrics and behaviour

## 8.7 External Data Utility Layer

The file `scripts/collect_external_data.py` supports optional data acquisition and normalization for:

- StatsBomb open-data downloads
- Transfermarkt-like table scraping
- Twitter/X recent-search collection
- local injury CSV normalization

The script also supports:

- config loading from YAML
- environment variable loading from `.env`
- sync-all workflow
- manifest writing
- fail-fast option

The companion `scripts/bootstrap_external_services.py` creates the external-source directory and copies `.env.example` to `.env` when appropriate. This shows that the project has been designed with future expansion in mind.

## 8.8 Configuration Layer

The `config/config.yaml` file captures project-level settings for:

- data paths
- model hyperparameters
- sequence settings
- feature categories
- preprocessing rules
- evaluation choices
- app display settings
- external source settings
- sentiment pipeline preferences

The presence of centralized configuration improves maintainability and demonstrates sound project organization.

## 8.9 Notes on `setup.py`

The repository also contains a `setup.py` file that appears to reflect an older setup workflow. It creates directories and checks for several files that do not exactly match the present repository structure. For academic reporting, this file can still be described as part of the broader setup intent, but the actual source of truth for current operation is the combination of:

- `README.md`
- `config/requirements.txt`
- `scripts/train_transfer_models.py`
- `app/backend_predict.py`
- `app/app.py`

This is worth noting in the report because it demonstrates careful inspection of the current codebase rather than repeating assumptions.

\newpage

---

# 9. Results and Evaluation

## 9.1 Evaluation Metrics Used

The project evaluates predictive performance using:

- **RMSE**
- **MAE**
- **R2**
- **sMAPE**
- **non-zero RMSE**

These metrics provide a balanced view of error magnitude, average deviation, explanatory power, percentage-style error interpretation, and robustness for meaningful-value cases.

## 9.2 Saved Evaluation Snapshot

The saved training summary reports the following values:

| **Model** | **RMSE** | **MAE** | **R2** | **sMAPE** | **non-zero RMSE** |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | 2,601,882.75 | 860,455.62 | 0.3530 | 83.3208 | 2,610,776.88 |
| **Multivariate LSTM** | 3,435,310.64 | 1,173,395.91 | -0.1278 | 109.8673 | 3,447,209.18 |
| **Univariate LSTM** | 3,466,873.60 | 1,247,201.74 | -0.1487 | 199.9835 | 3,478,890.53 |
| **Encoder-Decoder LSTM** | 3,490,902.78 | 1,261,500.84 | -0.1502 | 199.9910 | 3,503,173.13 |
| **Ensemble** | 2,620,356.67 | 820,423.98 | 0.3438 | 75.9458 | 2,629,332.22 |

## 9.3 Interpretation of Results

The key conclusions from the saved evaluation are:

1. **XGBoost is the strongest current standalone model.**  
   It achieves the best RMSE and best R2 among the saved model families.

2. **The ensemble is useful but XGBoost-dominant.**  
   Although the ensemble slightly improves MAE and sMAPE, it is heavily dependent on XGBoost because the saved blending weight is 0.9 for XGBoost.

3. **The LSTM variants underperform on the current split.**  
   This suggests that the available dataset or sequence construction may not yet provide sufficient temporal richness for sequence models to outperform the tabular baseline.

4. **The project still benefits from comparative modelling.**  
   Even when XGBoost wins, the existence of trained sequence models strengthens the report and reveals where future work should focus.

## 9.4 Practical Reading of the Metrics

An RMSE in the low millions indicates that transfer values are being predicted in a high-variance financial space. Football transfer markets can contain large differences between players, so absolute error values must be understood in context. The saved R2 of approximately 0.353 for XGBoost shows moderate explanatory power at this stage. While this leaves clear room for improvement, it is a valid and reportable result for a mid-term project using a relatively compact modelling set.

## 9.5 Top Feature Importance Signals

The saved XGBoost feature importance output highlights several influential features:

| **Rank** | **Feature** | **Importance** |
|---|---|---:|
| **1** | `cat__career_stage_Prospect` | 0.1291 |
| **2** | `num__market_value_log` | 0.0941 |
| **3** | `cat__contract_status_Long` | 0.0536 |
| **4** | `num__num_clubs` | 0.0401 |
| **5** | `cat__contract_status_Expired` | 0.0362 |
| **6** | `num__current_market_value` | 0.0358 |
| **7** | `num__assists_log` | 0.0296 |
| **8** | `num__market_value_per_season` | 0.0292 |
| **9** | `num__num_seasons` | 0.0256 |
| **10** | `num__injury_burden_index` | 0.0251 |

These results are insightful because they show that transfer valuation in the current system is strongly influenced by:

- career stage
- market value related measures
- contract status
- movement history
- performance contribution
- injury burden

This aligns well with domain expectations and supports the overall design of the feature engineering pipeline.

## 9.6 Evaluation Strengths

The evaluation design is strong at the repository level because it:

- compares multiple model families under one workflow
- stores machine-readable summary metadata
- saves prediction outputs
- supports interpretability through feature importance
- exposes performance information in the dashboard

## 9.7 Evaluation Gaps

At the same time, the project has understandable mid-term limitations:

- results are based on the current data split and present engineered dataset
- deeper cross-validation analysis is not yet fully surfaced in the saved outputs
- richer ablation studies and player case studies remain future opportunities

> **Figure Callout:** Figure 5 should compare model performance visually.  
> **Figure Callout:** Figure 6 should present the top feature importance chart.

\newpage

---

# 10. Dashboard and User Workflow

## 10.1 Purpose of the Dashboard

The Streamlit dashboard makes TransferIQ easier to evaluate, explain, and demonstrate. Many academic projects stop at model training, but this project extends into user interaction and visualization. This is important because it shows how predictive systems can be consumed by non-developer users.

## 10.2 Major Dashboard Sections

The application exposes three tabs:

### 10.2.1 AI Player Lab

This section allows user-level exploration of an individual player or selected library profile. It supports editable controls so that a valuation scenario can be altered and re-run. This makes the dashboard useful for explaining how features such as injuries, sentiment, or market value can influence outputs.

### 10.2.2 Bulk Scan

This section supports multi-row prediction workflows. Batch usage is important because practical analytics systems often need to evaluate several players together rather than one player at a time. The dashboard also includes prediction confidence output in this mode.

### 10.2.3 Model Intelligence

This section is designed for interpretability and performance review. It provides saved metrics and visual intelligence that help explain model behaviour rather than only showing a final number.

## 10.3 User Workflow

The dashboard user flow can be described as:

1. launch the application
2. choose an existing player or upload/provide player data
3. adjust inputs if needed
4. run prediction through the loaded predictor
5. review current value, predicted value, delta, and confidence
6. inspect bulk outputs or model intelligence if desired

## 10.4 Communication of Uncertainty

A useful feature of the dashboard is that it presents a confidence-aware valuation range rather than only a single number. This is good practice in decision-support applications because it acknowledges uncertainty and helps prevent over-interpretation of a single estimate.

## 10.5 Demonstration Value

The dashboard significantly improves the project's value during:

- mentor review
- viva presentation
- classroom demonstration
- report screenshot preparation
- comparative model explanation

It turns the repository into a presentation-ready prototype rather than a purely backend project.

> **Figure Callout:** Figure 7 should show the tab-level dashboard layout.  
> **Figure Callout:** Figure 8 should show a sample player valuation screen.

\newpage

---

# 11. Current Limitations

## 11.1 Model Limitations

The current saved results show that the LSTM-based models underperform relative to XGBoost. This suggests that either:

- the available dataset is not yet rich enough for sequential models to shine
- sequence construction needs refinement
- player histories are not yet aligned in the most informative way

## 11.2 Data Limitations

Although the project contains meaningful source datasets, the curated modelling set used in the present training workflow has 1,989 rows, with 961 rows used in the modelling snapshot. This is enough for a valuable academic prototype but still limited compared with full-scale production sports analytics datasets.

## 11.3 Sentiment Limitations

Sentiment is included thoughtfully, but the predictive value of sentiment depends on the quality and recency of text data. The fallback scorer maintains portability, but richer live text streams would likely improve realism and analytical strength.

## 11.4 External Data Limitations

The repository includes external collection logic, but live execution depends on:

- network access
- valid URLs
- API credentials
- data availability from third-party sources

Therefore, the code structure is present, but continuous live ingestion is not guaranteed in all evaluation settings.

## 11.5 Deployment and Maintenance Limitations

The project is organized and runnable locally, but it is still at an academic prototype stage. It does not yet include a full deployment pipeline, automated scheduled refresh workflow, or broad automated testing coverage.

## 11.6 Documentation and Legacy File Considerations

The current `setup.py` reflects an older repository expectation and does not fully match the latest file layout. This is not a failure of the project, but it is a useful observation for future cleanup and final-stage polishing.

\newpage

---

# 12. Planned Work / Timeline

## 12.1 Planned Technical Improvements

The next stage of work should focus on improving predictive quality, robustness, and presentation depth. Planned activities include:

- refining sequence construction for LSTM models
- tuning model hyperparameters further
- enriching sentiment sources
- improving external data validation
- adding targeted automated tests
- strengthening reproducibility and setup clarity
- expanding analysis with selected player case studies

## 12.2 Proposed Timeline

| Week / Phase | Planned Activity | Expected Output |
|---|---|---|
| Week 1 | Review current datasets and verify raw-to-processed flow | updated data notes and mapping |
| Week 2 | Refine feature engineering and inspect weak features | improved feature set observations |
| Week 3 | Tune XGBoost and sequential model parameters | comparative retraining results |
| Week 4 | Improve sequence modelling strategy | revised LSTM evaluation |
| Week 5 | Strengthen external-data validation and error handling | more robust ingestion workflow |
| Week 6 | Expand sentiment experiments | richer sentiment analysis section |
| Week 7 | Add tests for inference and feature processing | baseline reliability checks |
| Week 8 | Improve dashboard figures and documentation assets | report-ready screenshots and visuals |
| Week 9 | Prepare result interpretation and case studies | stronger evaluation chapter |
| Week 10 | Finalize report and presentation material | submission-ready documentation |

## 12.3 Expected Outcome of the Planned Work

If these steps are completed successfully, the final version of TransferIQ should:

- improve predictive reliability
- produce stronger comparative analysis
- better justify the role of sequence models
- provide richer academic discussion of results
- become easier to reproduce and evaluate

\newpage

---

# 13. Conclusion

TransferIQ is a substantial AI and machine learning project focused on football player transfer valuation. The repository already demonstrates an end-to-end workflow that includes curated data, preprocessing, feature engineering, sentiment integration, comparative model development, prediction interfaces, and an interactive dashboard.

At the current mid-term stage, the project's strongest practical outcome is the structured-feature XGBoost model, which outperforms the implemented LSTM variants on the saved evaluation snapshot. This is an important and honest finding. It shows that the project is being evaluated on evidence rather than assumptions. At the same time, the existence of multivariate LSTM, univariate LSTM, and encoder-decoder LSTM implementations expands the academic depth of the work and creates a clear path for future improvement.

The project is also strong from a software engineering perspective. It stores processed datasets, trained artefacts, metadata summaries, reusable core classes, configuration files, and presentation layers. The inclusion of both backend and dashboard prediction pathways makes the system suitable for academic demonstration, project review, and further extension.

Overall, TransferIQ can be confidently described as a meaningful applied AI project that combines sports analytics, machine learning, sentiment analysis, and interactive software design. The work completed so far provides a solid foundation for final-stage refinement, deeper evaluation, and submission-ready academic presentation.

\newpage

---

# 14. References

1. Python Software Foundation. *Python Language Reference*. Available at: https://www.python.org/
2. McKinney, W. *pandas Documentation*. Available at: https://pandas.pydata.org/
3. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362.
4. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
6. Abadi, M., Barham, P., Chen, J., et al. (2016). TensorFlow: A system for large-scale machine learning. In *12th USENIX Symposium on Operating Systems Design and Implementation*, 265-283.
7. Chollet, F. and others. *Keras Documentation*. Available at: https://keras.io/
8. Streamlit Inc. *Streamlit Documentation*. Available at: https://docs.streamlit.io/
9. Plotly Technologies Inc. *Plotly Python Open Source Graphing Library*. Available at: https://plotly.com/python/
10. Hutto, C. J. and Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the International AAAI Conference on Web and Social Media*, 216-225.
11. Loria, S. *TextBlob Documentation*. Available at: https://textblob.readthedocs.io/
12. Richardson, L. *Beautiful Soup Documentation*. Available at: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
13. Reitz, K. and others. *Requests Documentation*. Available at: https://requests.readthedocs.io/
14. YAML Language Development Team. *YAML Ain't Markup Language Version 1.2*. Available at: https://yaml.org/
15. StatsBomb Open Data. Available at: https://github.com/statsbomb/open-data
16. Transfermarkt. *Football transfer market platform*. Available at: https://www.transfermarkt.com/
17. General literature on sports analytics, player valuation, sequential forecasting, ensemble modelling, and sentiment-aware predictive systems may be cited additionally in the final report under supervisor guidance.

\newpage

---

# 15. Appendices

## Appendix A. Important Repository Files and Roles

| **File** | **Role in the Project** |
|---|---|
| `README.md` | repository overview, setup notes, and saved performance summary |
| `src/transfer_value_system.py` | main training, feature engineering, and prediction logic |
| `src/sentiment_pipeline.py` | sentiment analysis subsystem |
| `scripts/train_transfer_models.py` | model training entry point |
| `app/backend_predict.py` | command-line inference interface |
| `app/app.py` | Streamlit dashboard |
| `scripts/collect_external_data.py` | optional external collection workflows |
| `scripts/bootstrap_external_services.py` | local bootstrap for external services |
| `models/metadata/training_summary.json` | saved metrics and model metadata |
| `models/metadata/xgboost_feature_importance.csv` | saved feature importance output |

## Appendix B. Dataset Summary

| **Dataset** | **Rows** | **Columns** | **Notes** |
|---|---:|---:|---|
| `team_details.csv` | 2,175 | 12 | club metadata |
| `team_competitions_seasons.csv` | 58,247 | 29 | competition-season club statistics |
| `player_sentiment_data.csv` | 23,868 | 11 | sentiment and engagement information |
| `player_profiles.csv` | 92,671 | 34 | player profile information |
| `player_national_performances.csv` | 92,701 | 9 | national performance history |
| `player_market_value.csv` | 901,429 | 3 | historical market values |
| `player_injuries.csv` | 143,195 | 7 | injury history |
| `transfer_prediction_with_sentiment_cleaned.csv` | 1,989 | 29 | main raw modelling dataset |
| `enhanced_transfer_dataset.csv` | 1,989 | 83 | enhanced processed dataset |
| `advanced_transfer_features.csv` | 1,989 | 104 | main engineered features |
| `optimized_transfer_dataset.csv` | 1,989 | 27 | compact transfer-focused table |
| `optimized_players_final.csv` | 83 | 35 | compact player subset |
| `player_prediction_library.csv` | 961 | 25 | library used in app workflows |
| `test_predictions.csv` | 145 | 112 | saved prediction outputs |
| `sample_injury.csv` | 1 | 4 | example injury input |
| `injury_records_normalized.csv` | 1 | 5 | normalized injury output example |

## Appendix C. Training Metadata Snapshot

| **Field** | **Value** |
|---|---|
| Target column | `value_at_transfer` |
| Modelling rows | 961 |
| Full engineered rows | 1,989 |
| Engineered columns | 104 |
| Feature count after transformation | 109 |
| Sequence length | 8 |
| Forecast horizon | 3 |
| Ensemble XGBoost weight | 0.9 |
| Ensemble LSTM weight | 0.1 |

## Appendix D. Current Saved Metrics

| **Model** | **RMSE** | **MAE** | **R2** |
|---|---:|---:|---:|
| **XGBoost** | 2,601,882.75 | 860,455.62 | 0.3530 |
| **Multivariate LSTM** | 3,435,310.64 | 1,173,395.91 | -0.1278 |
| **Univariate LSTM** | 3,466,873.60 | 1,247,201.74 | -0.1487 |
| **Encoder-Decoder LSTM** | 3,490,902.78 | 1,261,500.84 | -0.1502 |
| **Ensemble** | 2,620,356.67 | 820,423.98 | 0.3438 |

## Appendix E. Suggested Screenshots for Final Submission

The final formatted Word or Google Docs document should ideally include screenshots of:

- the dashboard landing page
- AI Player Lab prediction screen
- Bulk Scan results table
- Model Intelligence visualizations
- feature importance chart
- sample command-line prediction output

## Appendix F. Commands for Demonstration

### Train the models

```powershell
.\venv\Scripts\python.exe scripts\train_transfer_models.py
```

### Run backend prediction with sample library

```powershell
.\venv\Scripts\python.exe app\backend_predict.py --sample-library
```

### Launch the Streamlit dashboard

```powershell
.\venv\Scripts\streamlit.exe run app\app.py
```

### Bootstrap external data services

```powershell
.\venv\Scripts\python.exe scripts\bootstrap_external_services.py
```

### Run external data collection

```powershell
.\venv\Scripts\python.exe scripts\collect_external_data.py --sync-all --write-manifest
```

## Appendix G. Notes for Final Formatting Transfer

When moving this markdown into Word or Google Docs:

- use page breaks between major front-matter sections
- generate TOC, List of Figures, and List of Tables automatically
- keep chapter numbering consistent
- convert figure callouts into actual captioned figures
- preserve placeholders for student and guide details until final submission

\newpage

---

# 16. Publication (If Any)

At the present mid-term stage, no formal publication, journal paper, conference paper, patent, or book chapter associated specifically with TransferIQ is recorded in the repository materials.

However, the project has a sufficiently strong technical base to support future academic extension in the form of:

- a comparative sports analytics study
- a report on sentiment-aware transfer value prediction
- a paper on structured versus sequential modelling for football transfer estimation

Any such publication would require broader experimentation, stronger comparative baselines, and more formal evaluation in the final phase.
