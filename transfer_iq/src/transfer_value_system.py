"""Shared training and inference system for TransferIQ."""

from __future__ import annotations

import ast
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

from src.sentiment_pipeline import TransferSentimentAnalyzer, sentiment_features_from_text


SEED = 42
TARGET_COLUMN = "value_at_transfer"
FORECAST_HORIZON = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transfer_prediction_with_sentiment_cleaned.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "advanced_transfer_features.csv"
PLAYER_LIBRARY_PATH = PROJECT_ROOT / "data" / "processed" / "player_prediction_library.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PREPROCESS_DIR = MODEL_DIR / "preprocessing"
TRAINED_DIR = MODEL_DIR / "trained"
METADATA_DIR = MODEL_DIR / "metadata"


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def safe_list(value: Any) -> list[str]:
    """Parse stringified or comma-separated list-like values."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text in {"Unknown", "Not Available", "nan"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    cleaned = text.strip('"').strip("'")
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def safe_date(value: Any) -> pd.Timestamp:
    """Parse dates while tolerating missing sentinels."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if text in {"", "Unknown", "Not Available", "01-01-1900", "1900-01-01"}:
        return pd.NaT
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def stable_log1p(series: pd.Series) -> pd.Series:
    """Log-transform non-negative values safely."""
    return np.log1p(np.clip(series.astype(float), a_min=0.0, a_max=None))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return core regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    safe_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / safe_denom) * 100.0)
    non_zero_mask = y_true > 0
    non_zero_rmse = float(np.sqrt(mean_squared_error(y_true[non_zero_mask], y_pred[non_zero_mask]))) if non_zero_mask.any() else rmse
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
        "non_zero_rmse": non_zero_rmse,
    }


def multi_step_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Aggregate metrics for multi-step forecasts."""
    flattened_true = np.asarray(y_true, dtype=float).reshape(-1)
    flattened_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    metrics = metric_dict(flattened_true, flattened_pred)
    first_step_metrics = metric_dict(np.asarray(y_true, dtype=float)[:, 0], np.asarray(y_pred, dtype=float)[:, 0])
    metrics["first_step_rmse"] = first_step_metrics["rmse"]
    metrics["first_step_mae"] = first_step_metrics["mae"]
    return metrics


def make_lstm_model(sequence_length: int, feature_count: int) -> keras.Model:
    """Create a compact LSTM for chronological transfer sequences."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, feature_count)),
            keras.layers.Masking(mask_value=0.0),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def make_univariate_lstm_model(sequence_length: int) -> keras.Model:
    """Create a univariate LSTM using only past target history."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, 1)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.Dropout(0.15),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def make_encoder_decoder_lstm(sequence_length: int, feature_count: int, forecast_horizon: int) -> keras.Model:
    """Create an encoder-decoder LSTM for multi-window forecasting."""
    inputs = keras.layers.Input(shape=(sequence_length, feature_count))
    encoded = keras.layers.LSTM(64, return_sequences=False)(inputs)
    encoded = keras.layers.Dropout(0.2)(encoded)
    repeated = keras.layers.RepeatVector(forecast_horizon)(encoded)
    decoded = keras.layers.LSTM(48, return_sequences=True)(repeated)
    decoded = keras.layers.TimeDistributed(keras.layers.Dense(24, activation="relu"))(decoded)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(1))(decoded)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


@dataclass
class SequencePayload:
    """Container for chronological sequence datasets."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_target_indices: np.ndarray
    val_target_indices: np.ndarray
    test_target_indices: np.ndarray


@dataclass
class ForecastPayload:
    """Container for multi-step forecasting datasets."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_source_indices: np.ndarray
    val_source_indices: np.ndarray
    test_source_indices: np.ndarray


class TransferFeatureBuilder:
    """Feature engineering built around the provided raw dataset."""
    SENTIMENT_TEXT_COLUMNS = [
        "sentiment_text",
        "sentiment_source_text",
        "social_posts",
        "social_media_posts",
        "media_excerpt",
        "news_excerpt",
    ]

    def __init__(self, target_reference_date: str = "2026-01-01") -> None:
        self.target_reference_date = pd.Timestamp(target_reference_date)
        self.sentiment_analyzer = TransferSentimentAnalyzer()

    def load_raw_data(self, path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
        return pd.read_csv(path)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        analyzer = getattr(self, "sentiment_analyzer", None)
        if analyzer is None:
            analyzer = TransferSentimentAnalyzer()
            self.sentiment_analyzer = analyzer

        base_numeric_defaults = {
            "player_id": 0.0,
            "total_goals": 0.0,
            "total_assists": 0.0,
            "current_market_value": 0.0,
            "total_injuries": 0.0,
            "total_days_missed": 0.0,
            "avg_sentiment_3m": 0.5,
            "sentiment_trend": 0.0,
            "sentiment_volatility": 0.0,
            "avg_monthly_mentions": 0.0,
            "mention_trend": 0.0,
            "engagement_rate": 0.0,
            "positive_sentiment_ratio": 0.5,
            "negative_sentiment_ratio": 0.5,
            "event_count": 0.0,
            "peak_sentiment": 0.5,
            "lowest_sentiment": 0.5,
            TARGET_COLUMN: 0.0,
            "transfer_fee": 0.0,
        }
        for column, default in base_numeric_defaults.items():
            if column not in frame.columns:
                frame[column] = default
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

        for text_column in [
            "player_name",
            "current_club_name",
            "from_team_name",
            "to_team_name",
            "contract_expires",
            "transfer_date",
            "seasons",
            "competitions",
            "clubs",
        ]:
            if text_column not in frame.columns:
                frame[text_column] = "Unknown"
            frame[text_column] = frame[text_column].fillna("Unknown")

        sentiment_text_columns = [column for column in self.SENTIMENT_TEXT_COLUMNS if column in frame.columns]
        if sentiment_text_columns:
            frame["sentiment_source_text"] = (
                frame[sentiment_text_columns]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.strip()
            )
        else:
            frame["sentiment_source_text"] = ""
        sentiment_text_features = frame["sentiment_source_text"].apply(
            lambda value: sentiment_features_from_text(value, analyzer=analyzer)
        )
        sentiment_text_df = pd.DataFrame(sentiment_text_features.tolist(), index=frame.index)
        replacement_columns = [column for column in sentiment_text_df.columns if column in frame.columns]
        if replacement_columns:
            frame = frame.drop(columns=replacement_columns)
        frame = pd.concat([frame, sentiment_text_df], axis=1)

        frame["transfer_date_dt"] = frame["transfer_date"].apply(safe_date)
        frame["contract_expires_dt"] = frame["contract_expires"].apply(safe_date)
        transfer_median = frame["transfer_date_dt"].dropna().median()
        if pd.isna(transfer_median):
            transfer_median = pd.Timestamp("2024-07-01")
        frame["transfer_date_dt"] = frame["transfer_date_dt"].fillna(transfer_median)

        frame["seasons_list"] = frame["seasons"].apply(safe_list)
        frame["competitions_list"] = frame["competitions"].apply(safe_list)
        frame["clubs_list"] = frame["clubs"].apply(safe_list)

        frame["num_seasons"] = frame["seasons_list"].apply(len).clip(lower=1)
        frame["num_competitions"] = frame["competitions_list"].apply(len)
        frame["num_clubs"] = frame["clubs_list"].apply(len).clip(lower=1)
        frame["goal_contributions"] = frame["total_goals"] + frame["total_assists"]
        frame["goals_per_season"] = frame["total_goals"] / frame["num_seasons"]
        frame["assists_per_season"] = frame["total_assists"] / frame["num_seasons"]
        frame["contributions_per_season"] = frame["goal_contributions"] / frame["num_seasons"]
        frame["goal_share"] = frame["total_goals"] / (frame["goal_contributions"] + 1.0)
        frame["assist_share"] = frame["total_assists"] / (frame["goal_contributions"] + 1.0)
        frame["goal_assist_balance"] = np.abs(frame["goal_share"] - frame["assist_share"])
        frame["clubs_per_season"] = frame["num_clubs"] / frame["num_seasons"]
        frame["competitions_per_season"] = frame["num_competitions"] / frame["num_seasons"]

        frame["injury_days_per_injury"] = frame["total_days_missed"] / (frame["total_injuries"] + 1.0)
        frame["injuries_per_season"] = frame["total_injuries"] / frame["num_seasons"]
        frame["days_missed_per_season"] = frame["total_days_missed"] / frame["num_seasons"]
        frame["injury_burden_index"] = frame["injuries_per_season"] * 0.45 + frame["days_missed_per_season"] * 0.55

        frame["sentiment_composite"] = (
            frame["avg_sentiment_3m"] * 0.35
            + frame["positive_sentiment_ratio"] * 0.25
            + (1.0 - frame["negative_sentiment_ratio"]) * 0.20
            + np.clip(frame["peak_sentiment"] - frame["lowest_sentiment"], 0.0, None) * 0.20
        )
        frame["sentiment_composite"] = (
            frame["sentiment_composite"] * 0.8
            + frame["text_sentiment_compound_scaled"] * 0.2
        )
        frame["sentiment_polarity_gap"] = frame["positive_sentiment_ratio"] - frame["negative_sentiment_ratio"]
        frame["sentiment_shock"] = frame["peak_sentiment"] - frame["lowest_sentiment"]
        frame["buzz_score"] = np.log1p(np.clip(frame["avg_monthly_mentions"], 0.0, None)) * (frame["engagement_rate"] + 1.0)
        frame["engagement_per_mention"] = frame["engagement_rate"] / (frame["avg_monthly_mentions"] + 1.0)
        frame["sentiment_momentum"] = frame["sentiment_trend"] * (1.0 - np.clip(frame["sentiment_volatility"], 0.0, 1.0))
        frame["sentiment_stability"] = 1.0 - np.clip(frame["sentiment_volatility"], 0.0, 1.0)
        frame["text_sentiment_alignment"] = frame["text_sentiment_compound_scaled"] - frame["avg_sentiment_3m"]
        frame["text_popularity_proxy"] = frame["text_sentiment_magnitude"] * (frame["text_sentiment_token_count"] + 1.0)

        frame["transfer_year"] = frame["transfer_date_dt"].dt.year.astype(int)
        frame["transfer_month"] = frame["transfer_date_dt"].dt.month.astype(int)
        frame["transfer_quarter"] = frame["transfer_date_dt"].dt.quarter.astype(int)
        frame["days_since_transfer"] = (self.target_reference_date - frame["transfer_date_dt"]).dt.days.astype(float)
        frame["contract_days_remaining"] = (frame["contract_expires_dt"] - frame["transfer_date_dt"]).dt.days
        frame["contract_days_remaining"] = frame["contract_days_remaining"].fillna(-1.0)
        frame["contract_known_flag"] = (frame["contract_days_remaining"] >= 0).astype(int)
        frame["contract_years_remaining"] = np.clip(frame["contract_days_remaining"], 0, None) / 365.0

        frame["age_proxy"] = np.clip(17 + frame["num_seasons"] + np.where(frame["num_clubs"] > 5, 2, 0), 17, 38)
        frame["prime_years_flag"] = frame["age_proxy"].between(23, 29).astype(int)
        frame["career_stage"] = pd.cut(
            frame["age_proxy"],
            bins=[0, 21, 24, 29, 33, 99],
            labels=["Prospect", "Emerging", "Prime", "Experienced", "Veteran"],
            include_lowest=True,
        ).astype(str)

        frame["market_value_log"] = stable_log1p(frame["current_market_value"])
        frame["goals_log"] = stable_log1p(frame["total_goals"])
        frame["assists_log"] = stable_log1p(frame["total_assists"])
        frame["injury_days_log"] = stable_log1p(frame["total_days_missed"])
        frame["mentions_log"] = stable_log1p(frame["avg_monthly_mentions"])
        frame["event_count_log"] = stable_log1p(frame["event_count"])
        frame["market_value_per_contribution"] = frame["current_market_value"] / (frame["goal_contributions"] + 1.0)
        frame["market_value_per_season"] = frame["current_market_value"] / frame["num_seasons"]
        frame["performance_index"] = (
            frame["goals_per_season"] * 0.45
            + frame["assists_per_season"] * 0.20
            + frame["sentiment_composite"] * 3.5
            - frame["injury_burden_index"] * 0.015
            + frame["engagement_per_mention"] * 0.35
        )
        frame["market_pressure_index"] = frame["buzz_score"] * np.maximum(frame["sentiment_composite"], 0.05) / (frame["injury_burden_index"] + 1.0)
        frame["competition_score"] = frame["competitions_list"].apply(self._competition_score)
        frame["club_prestige_score"] = frame["clubs_list"].apply(self._club_prestige_score)
        frame["league_exposure_index"] = frame["competition_score"] / frame["num_seasons"]
        frame["mobility_index"] = frame["clubs_per_season"] * (1.0 + frame["num_competitions"] / 10.0)

        frame["sentiment_band"] = pd.cut(
            frame["avg_sentiment_3m"],
            bins=[-0.01, 0.2, 0.45, 0.65, 0.8, 1.1],
            labels=["Cold", "Mixed", "Positive", "Strong", "Elite"],
            include_lowest=True,
        ).astype(str)
        frame["injury_risk_category"] = pd.cut(
            frame["injury_burden_index"],
            bins=[-1, 0.1, 10, 45, 2000],
            labels=["Low", "Moderate", "Elevated", "High"],
            include_lowest=True,
        ).astype(str)
        frame["contract_status"] = pd.cut(
            frame["contract_days_remaining"],
            bins=[-9999, 0, 180, 365, 730, 10000],
            labels=["Expired", "Expiring", "Short", "Medium", "Long"],
            include_lowest=True,
        ).astype(str)
        frame["transfer_window"] = np.select(
            [
                frame["transfer_month"].isin([1, 2]),
                frame["transfer_month"].isin([6, 7, 8, 9]),
            ],
            ["Winter", "Summer"],
            default="Other",
        )
        frame["team_transition_type"] = np.select(
            [
                frame["to_team_name"].eq("Without Club") | frame["current_club_name"].eq("Without Club"),
                frame["from_team_name"].eq(frame["to_team_name"]),
                frame["to_team_name"].eq("Unknown"),
            ],
            ["Exit", "Internal", "Unknown"],
            default="External",
        )

        frame["from_team_frequency"] = frame["from_team_name"].map(frame["from_team_name"].value_counts())
        frame["to_team_frequency"] = frame["to_team_name"].map(frame["to_team_name"].value_counts())
        frame["current_club_frequency"] = frame["current_club_name"].map(frame["current_club_name"].value_counts())
        frame["is_free_transfer_like"] = (frame["transfer_fee"] <= 0).astype(int)
        frame["has_market_value"] = (frame["current_market_value"] > 0).astype(int)
        frame["has_contract_info"] = frame["contract_known_flag"]
        recent_event_series = frame["recent_event"] if "recent_event" in frame.columns else pd.Series(False, index=frame.index)
        frame["recent_event_flag"] = recent_event_series.astype(str).str.lower().isin(["true", "1"]).astype(int)

        frame["target_value"] = frame[TARGET_COLUMN].astype(float)
        frame = frame.sort_values(["transfer_date_dt", "player_id"], kind="mergesort").reset_index(drop=True)
        return frame

    @staticmethod
    def _competition_score(competitions: list[str]) -> float:
        weights = {
            "champions league": 6,
            "premier league": 5,
            "laliga": 5,
            "serie a": 5,
            "bundesliga": 5,
            "ligue 1": 5,
            "europa": 4,
            "championship": 3,
            "serie b": 2,
            "j league": 2,
            "ncaa": 1,
        }
        score = 0.0
        for competition in competitions:
            label = str(competition).lower()
            matched = False
            for key, value in weights.items():
                if key in label:
                    score += value
                    matched = True
                    break
            if not matched:
                score += 1.0
        return score

    @staticmethod
    def _club_prestige_score(clubs: list[str]) -> float:
        elite_terms = ("fc", "united", "city", "real", "athletic", "inter", "milan", "juventus", "arsenal", "chelsea", "barcelona")
        score = 0.0
        for club in clubs:
            label = str(club).lower()
            score += 1.0
            if any(term in label for term in elite_terms):
                score += 1.5
        return score

    def feature_columns(self) -> tuple[list[str], list[str], list[str]]:
        """Return columns used by the models."""
        numeric_columns = [
            "total_goals",
            "total_assists",
            "current_market_value",
            "total_injuries",
            "total_days_missed",
            "avg_sentiment_3m",
            "sentiment_trend",
            "sentiment_volatility",
            "avg_monthly_mentions",
            "mention_trend",
            "engagement_rate",
            "positive_sentiment_ratio",
            "negative_sentiment_ratio",
            "event_count",
            "peak_sentiment",
            "lowest_sentiment",
            "num_seasons",
            "num_competitions",
            "num_clubs",
            "goal_contributions",
            "goals_per_season",
            "assists_per_season",
            "contributions_per_season",
            "goal_share",
            "assist_share",
            "goal_assist_balance",
            "clubs_per_season",
            "competitions_per_season",
            "injury_days_per_injury",
            "injuries_per_season",
            "days_missed_per_season",
            "injury_burden_index",
            "sentiment_composite",
            "sentiment_polarity_gap",
            "sentiment_shock",
            "buzz_score",
            "engagement_per_mention",
            "sentiment_momentum",
            "sentiment_stability",
            "text_sentiment_compound",
            "text_sentiment_compound_scaled",
            "text_sentiment_positive_ratio",
            "text_sentiment_negative_ratio",
            "text_sentiment_token_count",
            "text_sentiment_magnitude",
            "text_sentiment_alignment",
            "text_popularity_proxy",
            "transfer_year",
            "transfer_month",
            "transfer_quarter",
            "days_since_transfer",
            "contract_days_remaining",
            "contract_years_remaining",
            "age_proxy",
            "market_value_log",
            "goals_log",
            "assists_log",
            "injury_days_log",
            "mentions_log",
            "event_count_log",
            "market_value_per_contribution",
            "market_value_per_season",
            "performance_index",
            "market_pressure_index",
            "competition_score",
            "club_prestige_score",
            "league_exposure_index",
            "mobility_index",
            "from_team_frequency",
            "to_team_frequency",
            "current_club_frequency",
            "is_free_transfer_like",
            "has_market_value",
            "has_contract_info",
            "contract_known_flag",
            "prime_years_flag",
            "recent_event_flag",
        ]
        categorical_columns = [
            "current_club_name",
            "from_team_name",
            "to_team_name",
            "career_stage",
            "sentiment_band",
            "injury_risk_category",
            "contract_status",
            "transfer_window",
            "team_transition_type",
        ]
        return numeric_columns + categorical_columns, numeric_columns, categorical_columns


class TransferValueTrainer:
    """Train XGBoost, LSTM, and weighted ensemble models."""

    def __init__(self, sequence_length: int = 8, forecast_horizon: int = FORECAST_HORIZON, random_seed: int = SEED) -> None:
        set_global_seed(random_seed)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.random_seed = random_seed
        self.builder = TransferFeatureBuilder()
        self.preprocessor: ColumnTransformer | None = None
        self.xgb_model: XGBRegressor | None = None
        self.lstm_model: keras.Model | None = None
        self.univariate_lstm_model: keras.Model | None = None
        self.encoder_decoder_model: keras.Model | None = None
        self.bundle: dict[str, Any] = {}

    @staticmethod
    def _build_monotone_constraints(transformed_feature_names: list[str]) -> str:
        """Bias the tree model toward football-plausible monotonic relationships."""
        positive_features = {
            "total_goals",
            "total_assists",
            "current_market_value",
            "avg_sentiment_3m",
            "avg_monthly_mentions",
            "engagement_rate",
            "positive_sentiment_ratio",
            "peak_sentiment",
            "num_competitions",
            "goal_contributions",
            "goals_per_season",
            "assists_per_season",
            "contributions_per_season",
            "competitions_per_season",
            "sentiment_composite",
            "sentiment_polarity_gap",
            "buzz_score",
            "engagement_per_mention",
            "sentiment_momentum",
            "sentiment_stability",
            "text_sentiment_compound",
            "text_sentiment_compound_scaled",
            "text_sentiment_positive_ratio",
            "text_sentiment_magnitude",
            "text_popularity_proxy",
            "contract_days_remaining",
            "contract_years_remaining",
            "market_value_log",
            "goals_log",
            "assists_log",
            "mentions_log",
            "market_value_per_season",
            "performance_index",
            "market_pressure_index",
            "competition_score",
            "club_prestige_score",
            "league_exposure_index",
            "has_market_value",
            "has_contract_info",
            "contract_known_flag",
            "prime_years_flag",
        }
        negative_features = {
            "total_injuries",
            "total_days_missed",
            "negative_sentiment_ratio",
            "sentiment_volatility",
            "text_sentiment_negative_ratio",
            "injury_days_per_injury",
            "injuries_per_season",
            "days_missed_per_season",
            "injury_burden_index",
            "injury_days_log",
        }

        constraints: list[int] = []
        for name in transformed_feature_names:
            feature_name = name.split("__", 1)[1] if "__" in name else name
            if feature_name in positive_features:
                constraints.append(1)
            elif feature_name in negative_features:
                constraints.append(-1)
            else:
                constraints.append(0)
        return "(" + ",".join(str(value) for value in constraints) + ")"

    def train(self, raw_path: Path | str = RAW_DATA_PATH) -> dict[str, Any]:
        full_frame = self.builder.engineer_features(self.builder.load_raw_data(raw_path))
        frame = full_frame[(full_frame["target_value"] > 0) | (full_frame["current_market_value"] > 0)].copy().reset_index(drop=True)
        feature_columns, numeric_columns, categorical_columns = self.builder.feature_columns()
        y = frame["target_value"].astype(float)
        X = frame[feature_columns]

        train_end = int(len(frame) * 0.70)
        val_end = int(len(frame) * 0.85)

        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
            ]
        )
        self.preprocessor = ColumnTransformer(
            [
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

        X_train_t = self.preprocessor.fit_transform(X_train).astype(np.float32)
        X_val_t = self.preprocessor.transform(X_val).astype(np.float32)
        X_test_t = self.preprocessor.transform(X_test).astype(np.float32)
        X_all_t = self.preprocessor.transform(X).astype(np.float32)
        transformed_feature_names = self.preprocessor.get_feature_names_out().tolist()
        monotone_constraints = self._build_monotone_constraints(transformed_feature_names)

        y_train_log = np.log1p(np.clip(y_train.to_numpy(), 0.0, None))
        y_val_log = np.log1p(np.clip(y_val.to_numpy(), 0.0, None))

        self.xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.2,
            monotone_constraints=monotone_constraints,
            random_state=self.random_seed,
            n_jobs=4,
        )
        self.xgb_model.fit(X_train_t, y_train_log, eval_set=[(X_val_t, y_val_log)], verbose=False)

        xgb_val_pred = np.expm1(self.xgb_model.predict(X_val_t))
        xgb_test_pred = np.expm1(self.xgb_model.predict(X_test_t))

        sequences = self._make_sequences(X_all_t, y.to_numpy(), train_end, val_end)
        self.lstm_model = make_lstm_model(self.sequence_length, X_all_t.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ]
        self.lstm_model.fit(
            sequences.X_train,
            np.log1p(np.clip(sequences.y_train, 0.0, None)),
            validation_data=(sequences.X_val, np.log1p(np.clip(sequences.y_val, 0.0, None))),
            epochs=40,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
        )

        lstm_val_pred = np.expm1(self.lstm_model.predict(sequences.X_val, verbose=0).reshape(-1))
        lstm_test_pred = np.expm1(self.lstm_model.predict(sequences.X_test, verbose=0).reshape(-1))
        xgb_val_seq_pred = xgb_val_pred[sequences.val_target_indices - train_end]
        xgb_test_seq_pred = xgb_test_pred[sequences.test_target_indices - val_end]

        target_logs = np.log1p(np.clip(y.to_numpy(dtype=float), 0.0, None))
        uni_sequences = self._make_univariate_sequences(target_logs, train_end, val_end)
        self.univariate_lstm_model = make_univariate_lstm_model(self.sequence_length)
        self.univariate_lstm_model.fit(
            uni_sequences.X_train,
            uni_sequences.y_train,
            validation_data=(uni_sequences.X_val, uni_sequences.y_val),
            epochs=40,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
        )
        univariate_test_pred = np.expm1(self.univariate_lstm_model.predict(uni_sequences.X_test, verbose=0).reshape(-1))

        forecast_sequences = self._make_forecast_sequences(X_all_t, target_logs, train_end, val_end)
        self.encoder_decoder_model = make_encoder_decoder_lstm(self.sequence_length, X_all_t.shape[1], self.forecast_horizon)
        self.encoder_decoder_model.fit(
            forecast_sequences.X_train,
            forecast_sequences.y_train[..., np.newaxis],
            validation_data=(forecast_sequences.X_val, forecast_sequences.y_val[..., np.newaxis]),
            epochs=40,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
        )
        encoder_decoder_test_pred = np.expm1(
            self.encoder_decoder_model.predict(forecast_sequences.X_test, verbose=0).reshape(-1, self.forecast_horizon)
        )

        best_weight = 0.5
        best_rmse = float("inf")
        for weight in np.arange(0.0, 1.01, 0.05):
            ensemble_val = weight * xgb_val_seq_pred + (1.0 - weight) * lstm_val_pred
            rmse = np.sqrt(mean_squared_error(sequences.y_val, ensemble_val))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weight = float(weight)

        ensemble_test_pred = best_weight * xgb_test_seq_pred + (1.0 - best_weight) * lstm_test_pred

        xgb_metrics = metric_dict(y_test.to_numpy(), xgb_test_pred)
        lstm_metrics = metric_dict(sequences.y_test, lstm_test_pred)
        univariate_lstm_metrics = metric_dict(np.expm1(uni_sequences.y_test), univariate_test_pred)
        encoder_decoder_metrics = multi_step_metric_dict(np.expm1(forecast_sequences.y_test), encoder_decoder_test_pred)
        ensemble_metrics = metric_dict(sequences.y_test, ensemble_test_pred)

        predictions_df = frame.iloc[val_end:].copy()
        predictions_df["xgb_prediction"] = xgb_test_pred
        predictions_df["lstm_prediction"] = np.nan
        predictions_df["univariate_lstm_prediction"] = np.nan
        predictions_df["ensemble_prediction"] = np.nan
        predictions_df.loc[sequences.test_target_indices, "lstm_prediction"] = lstm_test_pred
        predictions_df.loc[sequences.test_target_indices, "ensemble_prediction"] = ensemble_test_pred
        predictions_df.loc[uni_sequences.test_target_indices, "univariate_lstm_prediction"] = univariate_test_pred
        for horizon_idx in range(self.forecast_horizon):
            predictions_df[f"encoder_decoder_forecast_t+{horizon_idx + 1}"] = np.nan
            target_rows = forecast_sequences.test_source_indices + horizon_idx + 1
            valid_mask = target_rows < len(frame)
            predictions_df.loc[target_rows[valid_mask], f"encoder_decoder_forecast_t+{horizon_idx + 1}"] = encoder_decoder_test_pred[valid_mask, horizon_idx]
        predictions_df["prediction_residual"] = predictions_df["target_value"] - predictions_df["ensemble_prediction"]

        feature_importance = pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": self.xgb_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.bundle = {
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "seed": self.random_seed,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "transformed_feature_names": transformed_feature_names,
            "monotone_constraints": monotone_constraints,
            "ensemble_weight_xgb": best_weight,
            "ensemble_weight_lstm": 1.0 - best_weight,
            "reference_date": str(self.builder.target_reference_date.date()),
            "median_target_value": float(frame["target_value"].median()),
        }

        self._persist_artifacts(
            full_frame,
            frame,
            predictions_df,
            feature_importance,
            xgb_metrics,
            lstm_metrics,
            univariate_lstm_metrics,
            encoder_decoder_metrics,
            ensemble_metrics,
        )

        return {
            "xgb_metrics": xgb_metrics,
            "lstm_metrics": lstm_metrics,
            "univariate_lstm_metrics": univariate_lstm_metrics,
            "encoder_decoder_metrics": encoder_decoder_metrics,
            "ensemble_metrics": ensemble_metrics,
            "ensemble_weight_xgb": best_weight,
            "train_rows": int(train_end),
            "validation_rows": int(val_end - train_end),
            "test_rows": int(len(frame) - val_end),
            "active_training_rows": int(len(frame)),
            "full_engineered_rows": int(len(full_frame)),
            "feature_count": int(len(transformed_feature_names)),
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
        }

    def _make_sequences(self, X_all: np.ndarray, y_all: np.ndarray, train_end: int, val_end: int) -> SequencePayload:
        sequence_rows: list[np.ndarray] = []
        targets: list[float] = []
        target_indices: list[int] = []
        for target_idx in range(self.sequence_length - 1, len(X_all)):
            sequence_rows.append(X_all[target_idx - self.sequence_length + 1 : target_idx + 1])
            targets.append(float(y_all[target_idx]))
            target_indices.append(target_idx)
        sequences = np.asarray(sequence_rows, dtype=np.float32)
        targets_np = np.asarray(targets, dtype=np.float32)
        indices_np = np.asarray(target_indices, dtype=int)

        train_mask = indices_np < train_end
        val_mask = (indices_np >= train_end) & (indices_np < val_end)
        test_mask = indices_np >= val_end
        return SequencePayload(
            X_train=sequences[train_mask],
            y_train=targets_np[train_mask],
            X_val=sequences[val_mask],
            y_val=targets_np[val_mask],
            X_test=sequences[test_mask],
            y_test=targets_np[test_mask],
            train_target_indices=indices_np[train_mask],
            val_target_indices=indices_np[val_mask],
            test_target_indices=indices_np[test_mask],
        )

    def _make_univariate_sequences(self, y_all_log: np.ndarray, train_end: int, val_end: int) -> SequencePayload:
        sequence_rows: list[np.ndarray] = []
        targets: list[float] = []
        target_indices: list[int] = []
        for target_idx in range(self.sequence_length, len(y_all_log)):
            sequence_rows.append(y_all_log[target_idx - self.sequence_length : target_idx].reshape(self.sequence_length, 1))
            targets.append(float(y_all_log[target_idx]))
            target_indices.append(target_idx)
        sequences = np.asarray(sequence_rows, dtype=np.float32)
        targets_np = np.asarray(targets, dtype=np.float32)
        indices_np = np.asarray(target_indices, dtype=int)

        train_mask = indices_np < train_end
        val_mask = (indices_np >= train_end) & (indices_np < val_end)
        test_mask = indices_np >= val_end
        return SequencePayload(
            X_train=sequences[train_mask],
            y_train=targets_np[train_mask],
            X_val=sequences[val_mask],
            y_val=targets_np[val_mask],
            X_test=sequences[test_mask],
            y_test=targets_np[test_mask],
            train_target_indices=indices_np[train_mask],
            val_target_indices=indices_np[val_mask],
            test_target_indices=indices_np[test_mask],
        )

    def _make_forecast_sequences(self, X_all: np.ndarray, y_all_log: np.ndarray, train_end: int, val_end: int) -> ForecastPayload:
        sequence_rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        source_indices: list[int] = []
        max_start = len(X_all) - self.forecast_horizon
        for source_idx in range(self.sequence_length - 1, max_start):
            sequence_rows.append(X_all[source_idx - self.sequence_length + 1 : source_idx + 1])
            targets.append(y_all_log[source_idx + 1 : source_idx + 1 + self.forecast_horizon])
            source_indices.append(source_idx)
        sequences = np.asarray(sequence_rows, dtype=np.float32)
        targets_np = np.asarray(targets, dtype=np.float32)
        indices_np = np.asarray(source_indices, dtype=int)

        train_mask = (indices_np + self.forecast_horizon) < train_end
        val_mask = ((indices_np + 1) >= train_end) & ((indices_np + self.forecast_horizon) < val_end)
        test_mask = (indices_np + 1) >= val_end
        return ForecastPayload(
            X_train=sequences[train_mask],
            y_train=targets_np[train_mask],
            X_val=sequences[val_mask],
            y_val=targets_np[val_mask],
            X_test=sequences[test_mask],
            y_test=targets_np[test_mask],
            train_source_indices=indices_np[train_mask],
            val_source_indices=indices_np[val_mask],
            test_source_indices=indices_np[test_mask],
        )

    def _persist_artifacts(
        self,
        full_engineered_frame: pd.DataFrame,
        modeling_frame: pd.DataFrame,
        predictions_df: pd.DataFrame,
        feature_importance: pd.DataFrame,
        xgb_metrics: dict[str, float],
        lstm_metrics: dict[str, float],
        univariate_lstm_metrics: dict[str, float],
        encoder_decoder_metrics: dict[str, float],
        ensemble_metrics: dict[str, float],
    ) -> None:
        for path in [PROCESSED_DATA_PATH.parent, PLAYER_LIBRARY_PATH.parent, PREPROCESS_DIR, TRAINED_DIR, METADATA_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        full_engineered_frame.to_csv(PROCESSED_DATA_PATH, index=False)
        predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)
        feature_importance.to_csv(METADATA_DIR / "xgboost_feature_importance.csv", index=False)

        player_library = modeling_frame[
            [
                "player_id",
                "player_name",
                "current_club_name",
                "from_team_name",
                "to_team_name",
                "target_value",
                "current_market_value",
                "total_goals",
                "total_assists",
                "total_injuries",
                "avg_sentiment_3m",
                "contract_days_remaining",
                "goal_contributions",
                "goals_per_season",
                "assists_per_season",
                "performance_index",
                "market_pressure_index",
                "career_stage",
                "sentiment_band",
                "injury_risk_category",
                "transfer_window",
                "competition_score",
                "age_proxy",
                "transfer_year",
            ]
        ].copy()
        transformed = self.preprocessor.transform(modeling_frame[self.bundle["feature_columns"]]).astype(np.float32)
        player_library["model_estimated_value"] = np.expm1(self.xgb_model.predict(transformed))
        player_library = player_library.sort_values(["current_market_value", "model_estimated_value"], ascending=False).reset_index(drop=True)
        player_library.to_csv(PLAYER_LIBRARY_PATH, index=False)

        with open(PREPROCESS_DIR / "pipeline_bundle.pkl", "wb") as handle:
            pickle.dump(
                {
                    "builder": self.builder,
                    "preprocessor": self.preprocessor,
                    "bundle": self.bundle,
                },
                handle,
            )
        with open(PREPROCESS_DIR / "feature_columns.json", "w", encoding="utf-8") as handle:
            json.dump(self.bundle, handle, indent=2)
        with open(TRAINED_DIR / "xgb_model.pkl", "wb") as handle:
            pickle.dump(self.xgb_model, handle)
        self.lstm_model.save(TRAINED_DIR / "lstm_model.keras", overwrite=True)
        self.univariate_lstm_model.save(TRAINED_DIR / "univariate_lstm_model.keras", overwrite=True)
        self.encoder_decoder_model.save(TRAINED_DIR / "encoder_decoder_lstm.keras", overwrite=True)
        with open(TRAINED_DIR / "ensemble_model.pkl", "wb") as handle:
            pickle.dump(
                {
                    "ensemble_weight_xgb": self.bundle["ensemble_weight_xgb"],
                    "ensemble_weight_lstm": self.bundle["ensemble_weight_lstm"],
                },
                handle,
            )

        summary = {
            "target_column": TARGET_COLUMN,
            "dataset_rows": int(len(modeling_frame)),
            "dataset_columns": int(full_engineered_frame.shape[1]),
            "full_engineered_rows": int(len(full_engineered_frame)),
            "xgb_metrics": xgb_metrics,
            "lstm_metrics": lstm_metrics,
            "univariate_lstm_metrics": univariate_lstm_metrics,
            "encoder_decoder_metrics": encoder_decoder_metrics,
            "ensemble_metrics": ensemble_metrics,
            "ensemble_weight_xgb": self.bundle["ensemble_weight_xgb"],
            "ensemble_weight_lstm": self.bundle["ensemble_weight_lstm"],
            "sequence_length": self.bundle["sequence_length"],
            "forecast_horizon": self.bundle["forecast_horizon"],
            "feature_count": len(self.bundle["transformed_feature_names"]),
            "top_features": feature_importance.head(15).to_dict(orient="records"),
        }
        with open(METADATA_DIR / "training_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


class TransferValuePredictor:
    """Inference service used by both terminal backend and Streamlit."""

    def __init__(self) -> None:
        with open(PREPROCESS_DIR / "pipeline_bundle.pkl", "rb") as handle:
            loaded = pickle.load(handle)
        self.builder: TransferFeatureBuilder = loaded["builder"]
        if not hasattr(self.builder, "sentiment_analyzer"):
            self.builder.sentiment_analyzer = TransferSentimentAnalyzer()
        self.preprocessor: ColumnTransformer = loaded["preprocessor"]
        self.bundle: dict[str, Any] = loaded["bundle"]
        with open(TRAINED_DIR / "xgb_model.pkl", "rb") as handle:
            self.xgb_model: XGBRegressor = pickle.load(handle)
        self.lstm_model = keras.models.load_model(TRAINED_DIR / "lstm_model.keras")
        encoder_decoder_path = TRAINED_DIR / "encoder_decoder_lstm.keras"
        self.encoder_decoder_model = keras.models.load_model(encoder_decoder_path) if encoder_decoder_path.exists() else None

    @staticmethod
    def _injury_discount_factor(frame: pd.DataFrame) -> np.ndarray:
        injury_count = np.clip(frame["total_injuries"].to_numpy(dtype=float), 0.0, None)
        days_missed = np.clip(frame["total_days_missed"].to_numpy(dtype=float), 0.0, None)
        burden = np.clip(frame["injury_burden_index"].to_numpy(dtype=float), 0.0, None)

        discount = (
            injury_count * 0.05
            + days_missed * 0.00022
            + np.minimum(burden, 120.0) * 0.0018
        )
        return 1.0 - np.clip(discount, 0.0, 0.72)

    @classmethod
    def _apply_injury_business_rule(
        cls,
        values: np.ndarray,
        frame: pd.DataFrame,
        healthy_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cap injured profiles beneath an equivalent healthy-profile ceiling."""
        discount_factor = cls._injury_discount_factor(frame)
        if healthy_reference is not None:
            adjusted = healthy_reference * discount_factor
        else:
            adjusted = values * discount_factor
        return np.clip(adjusted, 0.0, None)

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        engineered = self.builder.engineer_features(raw_df)
        features = engineered[self.bundle["feature_columns"]]
        transformed = self.preprocessor.transform(features).astype(np.float32)
        xgb_pred = np.clip(np.expm1(self.xgb_model.predict(transformed)), 0.0, None)

        lstm_pred = np.full(len(engineered), np.nan, dtype=float)
        if len(engineered) >= self.bundle["sequence_length"]:
            sequences = []
            for idx in range(self.bundle["sequence_length"] - 1, len(engineered)):
                sequences.append(transformed[idx - self.bundle["sequence_length"] + 1 : idx + 1])
            sequence_array = np.asarray(sequences, dtype=np.float32)
            seq_pred = np.clip(np.expm1(self.lstm_model.predict(sequence_array, verbose=0).reshape(-1)), 0.0, None)
            lstm_pred[self.bundle["sequence_length"] - 1 :] = seq_pred
            lstm_pred[: self.bundle["sequence_length"] - 1] = xgb_pred[: self.bundle["sequence_length"] - 1]
        else:
            lstm_pred = xgb_pred.copy()

        ensemble_pred = (
            self.bundle["ensemble_weight_xgb"] * xgb_pred
            + self.bundle["ensemble_weight_lstm"] * lstm_pred
        )
        healthy_reference = None
        if {"total_injuries", "total_days_missed"}.issubset(raw_df.columns):
            healthy_raw = raw_df.copy()
            healthy_raw["total_injuries"] = 0.0
            healthy_raw["total_days_missed"] = 0.0
            if "recent_event" in healthy_raw.columns:
                healthy_raw["recent_event"] = False
            healthy_engineered = self.builder.engineer_features(healthy_raw)
            healthy_features = healthy_engineered[self.bundle["feature_columns"]]
            healthy_transformed = self.preprocessor.transform(healthy_features).astype(np.float32)
            healthy_reference = np.clip(np.expm1(self.xgb_model.predict(healthy_transformed)), 0.0, None)
        ensemble_pred = self._apply_injury_business_rule(ensemble_pred, engineered, healthy_reference=healthy_reference)

        output = engineered.copy()
        output["xgb_prediction"] = xgb_pred
        output["lstm_prediction"] = lstm_pred
        output["ensemble_prediction"] = ensemble_pred
        output["prediction_delta_vs_current_value"] = output["ensemble_prediction"] - output["current_market_value"]
        for horizon_idx in range(int(self.bundle.get("forecast_horizon", 0))):
            output[f"encoder_decoder_forecast_t+{horizon_idx + 1}"] = np.nan
        if self.encoder_decoder_model is not None and len(engineered) >= self.bundle["sequence_length"]:
            forecast_sequences = []
            forecast_indices = []
            for idx in range(self.bundle["sequence_length"] - 1, len(engineered)):
                forecast_sequences.append(transformed[idx - self.bundle["sequence_length"] + 1 : idx + 1])
                forecast_indices.append(idx)
            forecast_array = np.asarray(forecast_sequences, dtype=np.float32)
            forecast_pred = np.expm1(
                self.encoder_decoder_model.predict(forecast_array, verbose=0).reshape(-1, int(self.bundle["forecast_horizon"]))
            )
            for horizon_idx in range(int(self.bundle["forecast_horizon"])):
                target_rows = np.asarray(forecast_indices, dtype=int) + horizon_idx + 1
                valid_mask = target_rows < len(output)
                if valid_mask.any():
                    output.loc[target_rows[valid_mask], f"encoder_decoder_forecast_t+{horizon_idx + 1}"] = forecast_pred[valid_mask, horizon_idx]
        output["prediction_confidence"] = np.clip(
            68.0
            + output["sentiment_composite"] * 18.0
            + output["has_contract_info"] * 4.0
            - output["injury_burden_index"] * 0.08,
            35.0,
            96.0,
        )
        return output

    def load_player_library(self) -> pd.DataFrame:
        return pd.read_csv(PLAYER_LIBRARY_PATH)
