"""Ingest a curated top-player CSV pack into TransferIQ datasets.

This script appends or refreshes curated player rows in the raw modeling dataset,
preserves the supplied profile details as extra columns, and regenerates the
prediction library using the already-trained inference artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_value_system import PLAYER_LIBRARY_PATH, RAW_DATA_PATH, TransferValuePredictor


MERGED_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "external_sources" / "top_player_details_merged.csv"

DEFAULT_COMPETITIONS = {
    "Inter Miami": "MLS, Concacaf Champions Cup",
    "Al Nassr": "Saudi Pro League, King's Cup",
    "PSG": "Ligue 1, Champions League, Coupe de France",
    "Al Hilal": "Saudi Pro League, AFC Champions League",
    "Manchester City": "Premier League, Champions League, FA Cup",
    "Real Madrid": "LaLiga, Champions League, Copa del Rey",
    "Liverpool": "Premier League, Champions League, FA Cup",
    "Bayern Munich": "Bundesliga, Champions League, DFB-Pokal",
    "Arsenal": "Premier League, Champions League, FA Cup",
    "Barcelona": "LaLiga, Champions League, Copa del Rey",
}

CLUB_VALUE_MULTIPLIER = {
    "Inter Miami": 0.92,
    "Al Nassr": 0.98,
    "PSG": 1.08,
    "Al Hilal": 0.96,
    "Manchester City": 1.10,
    "Real Madrid": 1.12,
    "Liverpool": 1.05,
    "Bayern Munich": 1.07,
    "Arsenal": 1.06,
    "Barcelona": 1.08,
}

MANUAL_MARKET_VALUE_OVERRIDES = {
    "Rodri": 130_000_000,
    "Bukayo Saka": 145_000_000,
    "Luka Modric": 15_000_000,
    "Pedri": 120_000_000,
    "Robert Lewandowski": 18_000_000,
}

PLAYER_LIBRARY_COLUMNS = [
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

PLAYER_LIBRARY_OPTIONAL_COLUMNS = [
    "country",
    "citizenship",
    "position",
    "main_position",
    "date_of_birth",
    "age",
    "matches_played",
    "injury_type",
    "source_dataset",
    "source_detail_level",
    "external_player_id",
    "source_market_value_date",
]


def normalize_name(value: object) -> str:
    text = str(value or "").strip().lower()
    return "".join(character for character in text if character.isalnum())


def compact_int(value: object, fallback: int = 0) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return fallback
    return int(round(float(numeric)))


def compact_float(value: object, fallback: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return fallback
    return float(numeric)


def clean_text(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return fallback if text.lower() == "nan" else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest curated top-player CSV files into TransferIQ datasets.")
    parser.add_argument("--players-file", required=True, help="Path to top_football_players.csv")
    parser.add_argument("--profiles-file", required=True, help="Path to top_players_profiles.csv")
    parser.add_argument("--performances-file", required=True, help="Path to top_players_performances.csv")
    parser.add_argument("--injuries-file", required=True, help="Path to top_players_injuries.csv")
    parser.add_argument("--market-values-file", required=True, help="Path to top_players_market_value.csv")
    parser.add_argument("--raw-path", default=str(RAW_DATA_PATH), help="Raw TransferIQ dataset to update.")
    parser.add_argument("--library-path", default=str(PLAYER_LIBRARY_PATH), help="Player library output path.")
    parser.add_argument("--merged-output", default=str(MERGED_OUTPUT_PATH), help="Merged top-player detail snapshot path.")
    return parser


def load_frame(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def latest_market_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["player_id", "value", "source_market_value_date"])
    market = frame.copy()
    market["date_numeric"] = pd.to_numeric(market["date_unix"], errors="coerce")
    market = market.sort_values(["player_id", "date_numeric"], na_position="last")
    market = market.groupby("player_id", as_index=False).tail(1).copy()
    market["source_market_value_date"] = pd.to_datetime(
        market["date_numeric"], unit="s", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    market["source_market_value_date"] = market["source_market_value_date"].fillna("")
    return market[["player_id", "value", "source_market_value_date"]]


def aggregate_injuries(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["player_id", "total_injuries", "total_days_missed", "injury_type"])

    def join_unique(values: pd.Series) -> str:
        cleaned = [str(value).strip() for value in values.dropna() if str(value).strip()]
        if not cleaned:
            return "None"
        return ", ".join(dict.fromkeys(cleaned))

    injuries = frame.copy()
    injuries["days_out"] = pd.to_numeric(injuries["days_out"], errors="coerce").fillna(0.0)
    aggregated = (
        injuries.groupby("player_id", as_index=False)
        .agg(
            total_injuries=("days_out", "size"),
            total_days_missed=("days_out", "sum"),
            injury_type=("injury_type", join_unique),
        )
        .copy()
    )
    aggregated["total_injuries"] = aggregated["total_injuries"].astype(int)
    aggregated["total_days_missed"] = aggregated["total_days_missed"].astype(int)
    return aggregated


def estimate_market_value(row: pd.Series) -> int:
    known_value = compact_int(row.get("value"), fallback=0)
    if known_value > 0:
        return known_value

    override = MANUAL_MARKET_VALUE_OVERRIDES.get(str(row.get("player_name", "")))
    if override is not None:
        return override

    goals = compact_int(row.get("goals"), fallback=0)
    assists = compact_int(row.get("assists"), fallback=0)
    matches_played = max(compact_int(row.get("matches_played"), fallback=0), 1)
    age = compact_int(row.get("age"), fallback=27)
    club = str(row.get("current_club_name", "")).strip()
    position = str(row.get("position", "")).strip().lower()

    age_multiplier = 1.45 if age <= 23 else 1.35 if age <= 26 else 1.18 if age <= 29 else 0.95 if age <= 32 else 0.68
    position_multiplier = 0.94 if "midfielder" in position else 1.0
    club_multiplier = CLUB_VALUE_MULTIPLIER.get(club, 1.0)
    production_value = goals * 3_000_000 + assists * 2_000_000 + matches_played * 300_000
    estimated = production_value * age_multiplier * position_multiplier * club_multiplier
    estimated = np.clip(estimated, 8_000_000, 180_000_000)
    return int(round(float(estimated), -5))


def infer_contract_expires(age: int) -> str:
    if age <= 22:
        year = 2030
    elif age <= 27:
        year = 2029
    elif age <= 31:
        year = 2028
    elif age <= 34:
        year = 2027
    else:
        year = 2026
    return f"30-06-{year}"


def infer_seasons(age: int) -> str:
    total = 8 if age >= 34 else 7 if age >= 30 else 6 if age >= 25 else 5
    seasons: list[str] = []
    start_year = 25
    for first_half in range(start_year, start_year - total, -1):
        second_half = (first_half + 1) % 100
        seasons.append(f"{first_half:02d}/{second_half:02d}")
    return ", ".join(seasons)


def infer_club_path(current_club: str, age: int) -> tuple[str, str]:
    if age >= 34:
        club_list = ["Academy Club", "Former Club", "Recent Club", current_club]
    elif age >= 28:
        club_list = ["Development Club", "Previous Club", current_club]
    else:
        club_list = ["Previous Club", current_club]
    return ", ".join(club_list), club_list[-2]


def infer_competitions(current_club: str) -> str:
    return DEFAULT_COMPETITIONS.get(current_club, "Domestic League, Continental Cup")


def infer_sentiment_features(
    market_value: int,
    goals: int,
    assists: int,
    matches_played: int,
    total_days_missed: int,
    age: int,
) -> dict[str, float | int | bool]:
    matches = max(matches_played, 1)
    production = (goals + assists * 0.75) / matches
    popularity = np.log1p(max(market_value, 1)) / np.log1p(200_000_000)
    injury_penalty = total_days_missed / 250.0
    age_penalty = max(age - 30, 0) * 0.015

    avg_sentiment = float(np.clip(0.55 + popularity * 0.20 + production * 0.16 - injury_penalty * 0.06 - age_penalty, 0.45, 1.0))
    positive_ratio = float(np.clip(0.52 + popularity * 0.18 + production * 0.08 - injury_penalty * 0.05, 0.45, 0.90))
    negative_ratio = float(np.clip(0.24 - popularity * 0.08 + injury_penalty * 0.05 + max(age - 32, 0) * 0.005, 0.05, 0.28))
    mentions = int(round(450 + popularity * 1200 + max(goals + assists - 20, 0) * 14))
    mention_trend = int(round((goals + assists * 0.6 - matches * 0.18) * 5))
    engagement_rate = float(np.clip(11 + production * 9 + popularity * 4, 10, 24))
    sentiment_trend = float(np.clip((positive_ratio - negative_ratio) * 0.18 - injury_penalty * 0.04, -0.15, 0.25))
    sentiment_volatility = float(np.clip(0.08 + abs(mention_trend) / 650 + injury_penalty * 0.12, 0.08, 0.24))
    peak_sentiment = float(np.clip(avg_sentiment + 0.12, 0.0, 1.0))
    lowest_sentiment = float(np.clip(avg_sentiment - 0.16 - injury_penalty * 0.06, 0.0, 1.0))
    event_count = int((total_days_missed > 0) + (goals >= 25))
    recent_event = bool(total_days_missed >= 60)

    return {
        "avg_sentiment_3m": round(avg_sentiment, 6),
        "sentiment_trend": round(sentiment_trend, 6),
        "sentiment_volatility": round(sentiment_volatility, 6),
        "avg_monthly_mentions": mentions,
        "mention_trend": mention_trend,
        "engagement_rate": round(engagement_rate, 6),
        "positive_sentiment_ratio": round(positive_ratio, 6),
        "negative_sentiment_ratio": round(negative_ratio, 6),
        "event_count": event_count,
        "peak_sentiment": round(peak_sentiment, 6),
        "lowest_sentiment": round(lowest_sentiment, 6),
        "recent_event": recent_event,
    }


def merge_top_player_pack(
    players: pd.DataFrame,
    profiles: pd.DataFrame,
    performances: pd.DataFrame,
    injuries: pd.DataFrame,
    market_values: pd.DataFrame,
) -> pd.DataFrame:
    roster = players.copy()
    roster["player_name_key"] = roster["player_name"].apply(normalize_name)
    roster = roster.rename(columns={"club": "current_club_name", "country": "country"})

    profile_frame = profiles.copy()
    profile_frame["player_name_key"] = profile_frame["player_name"].apply(normalize_name)
    profile_columns = [
        "player_name_key",
        "player_id",
        "player_name",
        "current_club_name",
        "citizenship",
        "position",
        "main_position",
        "date_of_birth",
    ]
    merged = roster.merge(profile_frame[profile_columns], on="player_name_key", how="left", suffixes=("", "_profile"))

    merged["player_name"] = merged["player_name"].fillna(merged["player_name_profile"])
    merged["current_club_name"] = merged["current_club_name_profile"].fillna(merged["current_club_name"])
    merged["citizenship"] = merged["citizenship"].fillna(merged["country"])
    merged["position"] = merged["position_profile"].fillna(merged["position"])
    merged["main_position"] = merged["main_position"].fillna(merged["position"])

    performance_columns = ["player_id", "matches_played", "goals", "assists"]
    merged = merged.merge(performances[performance_columns], on="player_id", how="left", suffixes=("", "_performance"))
    for column in ["matches_played", "goals", "assists"]:
        merged[column] = merged[f"{column}_performance"].fillna(merged[column])

    merged = merged.merge(aggregate_injuries(injuries), on="player_id", how="left")
    merged = merged.merge(latest_market_values(market_values), on="player_id", how="left")

    merged["source_dataset"] = "desktop_top_players_pack"
    merged["source_detail_level"] = np.where(merged["player_id"].notna(), "full", "summary")
    merged["external_player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")

    drop_columns = [
        "player_name_key",
        "player_name_profile",
        "current_club_name_profile",
        "position_profile",
        "matches_played_performance",
        "goals_performance",
        "assists_performance",
    ]
    merged = merged.drop(columns=[column for column in drop_columns if column in merged.columns])
    return merged


def assign_final_player_ids(merged_players: pd.DataFrame, base_raw: pd.DataFrame) -> pd.DataFrame:
    existing_ids = (
        base_raw[["player_name", "player_id"]]
        .assign(player_name_key=lambda frame: frame["player_name"].apply(normalize_name))
        .dropna(subset=["player_id"])
    )
    id_lookup = {
        key: compact_int(player_id)
        for key, player_id in zip(existing_ids["player_name_key"], existing_ids["player_id"])
    }

    used_ids = {
        compact_int(value)
        for value in pd.to_numeric(base_raw["player_id"], errors="coerce").dropna().astype(int).tolist()
    }
    next_id = max(used_ids) + 1 if used_ids else 1

    final_ids: list[int] = []
    for player_name in merged_players["player_name"].tolist():
        key = normalize_name(player_name)
        if key in id_lookup:
            final_ids.append(id_lookup[key])
            continue
        while next_id in used_ids:
            next_id += 1
        final_ids.append(next_id)
        used_ids.add(next_id)
        next_id += 1

    assigned = merged_players.copy()
    assigned["player_id"] = final_ids
    return assigned


def build_raw_rows(merged_players: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in merged_players.to_dict(orient="records"):
        player_name = str(row.get("player_name", "")).strip()
        current_club_name = str(row.get("current_club_name", "Unknown")).strip() or "Unknown"
        age = compact_int(row.get("age"), fallback=27)
        matches_played = compact_int(row.get("matches_played"), fallback=0)
        goals = compact_int(row.get("goals"), fallback=0)
        assists = compact_int(row.get("assists"), fallback=0)
        total_injuries = compact_int(row.get("total_injuries"), fallback=0)
        total_days_missed = compact_int(row.get("total_days_missed"), fallback=0)
        market_value = estimate_market_value(pd.Series(row))
        club_history, from_team_name = infer_club_path(current_club_name, age)
        sentiment_features = infer_sentiment_features(
            market_value=market_value,
            goals=goals,
            assists=assists,
            matches_played=matches_played,
            total_days_missed=total_days_missed,
            age=age,
        )
        transfer_fee_multiplier = 0.90 if age <= 28 else 0.82 if age <= 32 else 0.70

        records.append(
            {
                "player_id": compact_int(row.get("player_id"), fallback=0),
                "player_name": clean_text(player_name, fallback="Unknown Player"),
                "current_club_name": current_club_name,
                "contract_expires": infer_contract_expires(age),
                "seasons": infer_seasons(age),
                "competitions": infer_competitions(current_club_name),
                "clubs": club_history,
                "total_goals": goals,
                "total_assists": assists,
                "current_market_value": market_value,
                "total_injuries": total_injuries,
                "total_days_missed": total_days_missed,
                "transfer_date": "01-07-2025",
                "from_team_name": from_team_name,
                "to_team_name": current_club_name,
                "value_at_transfer": market_value,
                "transfer_fee": int(round(market_value * transfer_fee_multiplier, -5)),
                **sentiment_features,
                "country": clean_text(row.get("country", "")),
                "citizenship": clean_text(row.get("citizenship", row.get("country", ""))),
                "position": clean_text(row.get("position", "")),
                "main_position": clean_text(row.get("main_position", row.get("position", ""))),
                "date_of_birth": clean_text(row.get("date_of_birth", "")),
                "age": age,
                "matches_played": matches_played,
                "injury_type": clean_text(row.get("injury_type", "None"), fallback="None") or "None",
                "source_dataset": clean_text(row.get("source_dataset", "desktop_top_players_pack"), fallback="desktop_top_players_pack"),
                "source_detail_level": clean_text(row.get("source_detail_level", "summary"), fallback="summary"),
                "external_player_id": compact_int(row.get("external_player_id"), fallback=0),
                "source_market_value_date": clean_text(row.get("source_market_value_date", "")),
            }
        )

    return pd.DataFrame(records)


def upsert_raw_dataset(base_raw: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    names_to_replace = {normalize_name(name) for name in new_rows["player_name"].tolist()}
    filtered_raw = base_raw.loc[~base_raw["player_name"].apply(normalize_name).isin(names_to_replace)].copy()
    for column in new_rows.columns:
        if column not in filtered_raw.columns:
            filtered_raw[column] = pd.NA
    for column in filtered_raw.columns:
        if column not in new_rows.columns:
            new_rows[column] = pd.NA
    updated = pd.concat([filtered_raw, new_rows[filtered_raw.columns]], ignore_index=True)
    return updated


def build_player_library(raw_df: pd.DataFrame, predictor: TransferValuePredictor) -> pd.DataFrame:
    engineered = predictor.builder.engineer_features(raw_df)
    modeling_frame = engineered[
        (engineered["target_value"] > 0) | (engineered["current_market_value"] > 0)
    ].copy().reset_index(drop=True)

    transformed = predictor.preprocessor.transform(modeling_frame[predictor.bundle["feature_columns"]]).astype(np.float32)
    library_columns = PLAYER_LIBRARY_COLUMNS + [
        column for column in PLAYER_LIBRARY_OPTIONAL_COLUMNS if column in modeling_frame.columns
    ]
    library = modeling_frame[library_columns].copy()
    library["model_estimated_value"] = np.expm1(predictor.xgb_model.predict(transformed))
    library = library.sort_values(["current_market_value", "model_estimated_value"], ascending=False).reset_index(drop=True)
    return library


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    players = load_frame(args.players_file)
    profiles = load_frame(args.profiles_file)
    performances = load_frame(args.performances_file)
    injuries = load_frame(args.injuries_file)
    market_values = load_frame(args.market_values_file)
    raw_path = Path(args.raw_path)
    library_path = Path(args.library_path)
    merged_output = Path(args.merged_output)

    base_raw = load_frame(raw_path)
    merged_players = merge_top_player_pack(players, profiles, performances, injuries, market_values)
    merged_players = assign_final_player_ids(merged_players, base_raw)
    new_rows = build_raw_rows(merged_players)
    updated_raw = upsert_raw_dataset(base_raw, new_rows)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    updated_raw.to_csv(raw_path, index=False)
    merged_players.to_csv(merged_output, index=False)

    predictor = TransferValuePredictor()
    updated_library = build_player_library(updated_raw, predictor)
    updated_library.to_csv(library_path, index=False)

    print(f"Updated raw dataset: {raw_path} ({len(updated_raw):,} rows)")
    print(f"Updated player library: {library_path} ({len(updated_library):,} rows)")
    print(f"Merged detail snapshot: {merged_output} ({len(merged_players):,} rows)")
    print("Injected players:")
    for player_name in merged_players["player_name"].tolist():
        print(f"- {player_name}")


if __name__ == "__main__":
    main()
