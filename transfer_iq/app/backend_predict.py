"""Terminal backend for TransferIQ predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_value_system import RAW_DATA_PATH, TransferValuePredictor

EUR_TO_INR_RATE = 106.6


def format_indian_number(value: float) -> str:
    rounded = int(round(float(value)))
    sign = "-" if rounded < 0 else ""
    digits = str(abs(rounded))
    if len(digits) <= 3:
        return sign + digits
    last_three = digits[-3:]
    remaining = digits[:-3]
    groups: list[str] = []
    while len(remaining) > 2:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.append(remaining)
    return sign + ",".join(reversed(groups)) + "," + last_three


def format_dual_currency(value: float) -> str:
    return f"EUR {value:,.2f} / INR {format_indian_number(value * EUR_TO_INR_RATE)}"


def manual_player_row(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 9999999,
                "player_name": args.player_name,
                "current_club_name": args.current_club,
                "contract_expires": args.contract_expires,
                "seasons": args.seasons,
                "competitions": args.competitions,
                "clubs": args.clubs,
                "total_goals": args.total_goals,
                "total_assists": args.total_assists,
                "current_market_value": args.current_market_value,
                "total_injuries": args.total_injuries,
                "total_days_missed": args.total_days_missed,
                "transfer_date": args.transfer_date,
                "from_team_name": args.from_team_name,
                "to_team_name": args.to_team_name,
                "avg_sentiment_3m": args.avg_sentiment_3m,
                "sentiment_trend": args.sentiment_trend,
                "sentiment_volatility": args.sentiment_volatility,
                "avg_monthly_mentions": args.avg_monthly_mentions,
                "mention_trend": args.mention_trend,
                "engagement_rate": args.engagement_rate,
                "positive_sentiment_ratio": args.positive_sentiment_ratio,
                "negative_sentiment_ratio": args.negative_sentiment_ratio,
                "event_count": args.event_count,
                "peak_sentiment": args.peak_sentiment,
                "lowest_sentiment": args.lowest_sentiment,
                "recent_event": args.recent_event,
            }
        ]
    )


def print_results(results: pd.DataFrame) -> None:
    show_cols = [
        "player_name",
        "current_club_name",
        "current_market_value",
        "xgb_prediction",
        "lstm_prediction",
        "ensemble_prediction",
        "prediction_delta_vs_current_value",
        "prediction_confidence",
    ]
    printable = results[[col for col in show_cols if col in results.columns]].copy()
    for col in printable.columns:
        if pd.api.types.is_numeric_dtype(printable[col]):
            if "value" in col or "prediction" in col or "delta" in col:
                printable[col] = printable[col].map(format_dual_currency)
            else:
                printable[col] = printable[col].map(lambda value: f"{value:,.2f}")
    print(printable.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict transfer value with TransferIQ backend.")
    parser.add_argument("--csv", help="Path to a CSV file containing raw-player columns.")
    parser.add_argument("--sample-library", action="store_true", help="Run predictions on the saved player library.")
    parser.add_argument("--player-name", default="Sample Prospect")
    parser.add_argument("--current-club", default="AI FC")
    parser.add_argument("--contract-expires", default="30-06-2028")
    parser.add_argument("--seasons", default="25/26, 24/25, 23/24")
    parser.add_argument("--competitions", default="Premier League, FA Cup")
    parser.add_argument("--clubs", default="AI FC, Data United")
    parser.add_argument("--total-goals", type=float, default=18.0)
    parser.add_argument("--total-assists", type=float, default=9.0)
    parser.add_argument("--current-market-value", type=float, default=32000000.0)
    parser.add_argument("--total-injuries", type=float, default=1.0)
    parser.add_argument("--total-days-missed", type=float, default=18.0)
    parser.add_argument("--transfer-date", default="01-07-2026")
    parser.add_argument("--from-team-name", default="Data United")
    parser.add_argument("--to-team-name", default="AI FC")
    parser.add_argument("--avg-sentiment-3m", type=float, default=0.74)
    parser.add_argument("--sentiment-trend", type=float, default=0.06)
    parser.add_argument("--sentiment-volatility", type=float, default=0.12)
    parser.add_argument("--avg-monthly-mentions", type=float, default=1800.0)
    parser.add_argument("--mention-trend", type=float, default=120.0)
    parser.add_argument("--engagement-rate", type=float, default=15.5)
    parser.add_argument("--positive-sentiment-ratio", type=float, default=0.67)
    parser.add_argument("--negative-sentiment-ratio", type=float, default=0.16)
    parser.add_argument("--event-count", type=float, default=4.0)
    parser.add_argument("--peak-sentiment", type=float, default=0.91)
    parser.add_argument("--lowest-sentiment", type=float, default=0.43)
    parser.add_argument("--recent-event", default="True")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    predictor = TransferValuePredictor()

    if args.csv:
        input_df = pd.read_csv(args.csv)
    elif args.sample_library:
        library = pd.read_csv(RAW_DATA_PATH)
        input_df = (
            library.sort_values(["current_market_value", "value_at_transfer"], ascending=False)
            .head(12)
            .copy()
        )
    else:
        input_df = manual_player_row(args)

    results = predictor.predict(input_df)
    print_results(results)


if __name__ == "__main__":
    main()
