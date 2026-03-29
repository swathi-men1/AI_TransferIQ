"""Collect external source data for TransferIQ.

Supported collection targets:
- StatsBomb open-data match/player event files
- Transfermarkt HTML table scraping from a provided URL
- Twitter/X API v2 recent-search sentiment mentions
- Local injury-record CSV normalization

The script is intentionally opt-in and credential-driven so the project folder
contains the required data-collection stage without forcing network access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment_pipeline import TransferSentimentAnalyzer


RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = RAW_DIR / "external_sources"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / "config" / ".env"


def ensure_output_dir() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        if key:
            values[key] = value
    return values


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def collect_statsbomb_open_data(competition_id: int, season_id: int, base_url: str | None = None) -> Path:
    """Download open StatsBomb match metadata for a competition and season."""
    root = (base_url or "https://raw.githubusercontent.com/statsbomb/open-data/master/data/").rstrip("/")
    url = f"{root}/matches/{competition_id}/{season_id}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    output_path = EXTERNAL_DIR / f"statsbomb_matches_{competition_id}_{season_id}.csv"
    pd.DataFrame(payload).to_csv(output_path, index=False)
    return output_path


def collect_transfermarkt_table(source_url: str, user_agent: str = "TransferIQ/1.0") -> Path:
    """Scrape a simple HTML table from a Transfermarkt-like page."""
    response = requests.get(source_url, timeout=30, headers={"User-Agent": user_agent})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No table found on the provided Transfermarkt URL.")

    headers = [cell.get_text(strip=True) for cell in table.find_all("th")]
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all("td")]
        if cells:
            rows.append(cells)
    output_path = EXTERNAL_DIR / "transfermarkt_scrape.csv"
    pd.DataFrame(rows, columns=headers[: len(rows[0])] if rows and headers else None).to_csv(output_path, index=False)
    return output_path


def collect_twitter_mentions(query: str, max_results: int = 50) -> Path:
    """Fetch recent mentions from the Twitter/X API v2 recent search endpoint."""
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise EnvironmentError("TWITTER_BEARER_TOKEN is required to collect Twitter mentions.")

    response = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {bearer_token}"},
        params={
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,lang",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    analyzer = TransferSentimentAnalyzer()
    records: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        sentiment = analyzer.analyze(item.get("text", ""))
        records.append(
            {
                "tweet_id": item.get("id"),
                "text": item.get("text", ""),
                "created_at": item.get("created_at"),
                "lang": item.get("lang"),
                "retweet_count": item.get("public_metrics", {}).get("retweet_count", 0),
                "reply_count": item.get("public_metrics", {}).get("reply_count", 0),
                "like_count": item.get("public_metrics", {}).get("like_count", 0),
                "quote_count": item.get("public_metrics", {}).get("quote_count", 0),
                "sentiment_compound": sentiment.compound,
                "sentiment_source": sentiment.source,
            }
        )
    output_path = EXTERNAL_DIR / "twitter_mentions.csv"
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def normalize_injury_csv(path: Path | str) -> Path:
    """Normalize local injury-history records into a TransferIQ-friendly CSV."""
    input_path = Path(path)
    frame = pd.read_csv(input_path)
    normalized = frame.rename(
        columns={
            "player": "player_name",
            "injuries": "total_injuries",
            "days_missed": "total_days_missed",
            "latest_injury_date": "recent_injury_date",
        }
    )
    if "recent_event" not in normalized.columns:
        normalized["recent_event"] = normalized.get("recent_injury_date", "").astype(str).ne("")
    output_path = EXTERNAL_DIR / "injury_records_normalized.csv"
    normalized.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect external data for TransferIQ.")
    parser.add_argument("--statsbomb-competition-id", type=int)
    parser.add_argument("--statsbomb-season-id", type=int)
    parser.add_argument("--transfermarkt-url")
    parser.add_argument("--twitter-query")
    parser.add_argument("--twitter-max-results", type=int, default=50)
    parser.add_argument("--injury-csv")
    parser.add_argument("--sync-all", action="store_true", help="Run all configured external collectors.")
    parser.add_argument("--write-manifest", action="store_true", help="Write a JSON manifest of collected files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one provider fails.")
    return parser


def collect_all_from_config(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    sources = config.get("external_sources", {})

    statsbomb_cfg = sources.get("statsbomb", {})
    competition_id = args.statsbomb_competition_id or int(os.getenv("STATSBOMB_COMPETITION_ID", "0") or 0)
    season_id = args.statsbomb_season_id or int(os.getenv("STATSBOMB_SEASON_ID", "0") or 0)
    if statsbomb_cfg.get("enabled") and competition_id and season_id:
        try:
            outputs["statsbomb"] = str(
                collect_statsbomb_open_data(competition_id, season_id, base_url=statsbomb_cfg.get("base_url"))
            )
        except Exception as exc:
            errors["statsbomb"] = str(exc)
            if args.fail_fast:
                raise

    transfermarkt_cfg = sources.get("transfermarkt", {})
    transfermarkt_url = args.transfermarkt_url or os.getenv("TRANSFERMARKT_URL", "")
    if transfermarkt_cfg.get("enabled") and transfermarkt_url:
        try:
            outputs["transfermarkt"] = str(
                collect_transfermarkt_table(transfermarkt_url, user_agent=transfermarkt_cfg.get("user_agent", "TransferIQ/1.0"))
            )
        except Exception as exc:
            errors["transfermarkt"] = str(exc)
            if args.fail_fast:
                raise

    twitter_cfg = sources.get("twitter", {})
    twitter_query = args.twitter_query or os.getenv("TWITTER_QUERY", "")
    twitter_max_results = args.twitter_max_results or int(os.getenv("TWITTER_MAX_RESULTS", "50"))
    if twitter_cfg.get("enabled") and twitter_query:
        try:
            outputs["twitter"] = str(collect_twitter_mentions(twitter_query, twitter_max_results))
        except Exception as exc:
            errors["twitter"] = str(exc)
            if args.fail_fast:
                raise

    injury_cfg = sources.get("injury_data", {})
    injury_csv = args.injury_csv or injury_cfg.get("local_csv")
    if injury_cfg.get("enabled") and injury_csv:
        try:
            outputs["injury"] = str(normalize_injury_csv(PROJECT_ROOT / injury_csv if not Path(injury_csv).is_absolute() else injury_csv))
        except Exception as exc:
            errors["injury"] = str(exc)
            if args.fail_fast:
                raise

    return outputs, errors


def main() -> None:
    ensure_output_dir()
    load_env_file()
    config = load_config()
    parser = build_parser()
    args = parser.parse_args()

    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    if args.sync_all:
        outputs, errors = collect_all_from_config(config, args)
    else:
        if args.statsbomb_competition_id is not None and args.statsbomb_season_id is not None:
            statsbomb_cfg = config.get("external_sources", {}).get("statsbomb", {})
            outputs["statsbomb"] = str(
                collect_statsbomb_open_data(args.statsbomb_competition_id, args.statsbomb_season_id, base_url=statsbomb_cfg.get("base_url"))
            )
        if args.transfermarkt_url:
            transfermarkt_cfg = config.get("external_sources", {}).get("transfermarkt", {})
            outputs["transfermarkt"] = str(
                collect_transfermarkt_table(args.transfermarkt_url, user_agent=transfermarkt_cfg.get("user_agent", "TransferIQ/1.0"))
            )
        if args.twitter_query:
            outputs["twitter"] = str(collect_twitter_mentions(args.twitter_query, args.twitter_max_results))
        if args.injury_csv:
            outputs["injury"] = str(normalize_injury_csv(args.injury_csv))

    payload = {"outputs": outputs, "errors": errors}
    if args.write_manifest or args.sync_all:
        manifest_path = EXTERNAL_DIR / "collection_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["manifest"] = str(manifest_path)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
