"""Sentiment analysis utilities for TransferIQ.

This module prefers VADER or TextBlob when available and falls back to a
deterministic lexicon-based scorer so the training pipeline remains usable
without extra installs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover - optional dependency
    TextBlob = None


@dataclass
class SentimentResult:
    compound: float
    compound_scaled: float
    positive_ratio: float
    negative_ratio: float
    token_count: float
    magnitude: float
    source: str


class TransferSentimentAnalyzer:
    """Analyze free-text sentiment for player mentions and media excerpts."""

    POSITIVE_TERMS = {
        "elite", "excellent", "great", "strong", "positive", "breakthrough", "star", "top",
        "winner", "dominant", "impressive", "creative", "clinical", "valuable", "popular",
        "sharp", "brilliant", "secure", "quality", "confident", "rapid",
    }
    NEGATIVE_TERMS = {
        "injury", "poor", "negative", "weak", "decline", "struggle", "risk", "concern",
        "expensive", "overpriced", "absence", "setback", "uncertain", "volatile", "bad",
        "slow", "panic", "collapse", "doubt", "miss", "problem",
    }

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

    def analyze(self, value: Any) -> SentimentResult:
        text = str(value).strip()
        if not text or text.lower() == "unknown":
            return SentimentResult(0.0, 0.5, 0.5, 0.5, 0.0, 0.0, "empty")

        if self._vader is not None:
            scores = self._vader.polarity_scores(text)
            return SentimentResult(
                compound=float(scores["compound"]),
                compound_scaled=float(np.clip((scores["compound"] + 1.0) / 2.0, 0.0, 1.0)),
                positive_ratio=float(scores["pos"] if scores["pos"] > 0 else 0.5),
                negative_ratio=float(scores["neg"] if scores["neg"] > 0 else 0.5),
                token_count=float(len(re.findall(r"[a-zA-Z']+", text))),
                magnitude=float(abs(scores["compound"])),
                source="vader",
            )

        if TextBlob is not None:
            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            return SentimentResult(
                compound=polarity,
                compound_scaled=float(np.clip((polarity + 1.0) / 2.0, 0.0, 1.0)),
                positive_ratio=float(np.clip((polarity + 1.0) / 2.0, 0.0, 1.0)),
                negative_ratio=float(np.clip((1.0 - polarity) / 2.0, 0.0, 1.0)),
                token_count=float(len(blob.words)),
                magnitude=float(abs(polarity)),
                source="textblob",
            )

        return self._fallback(text)

    def _fallback(self, text: str) -> SentimentResult:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        if not tokens:
            return SentimentResult(0.0, 0.5, 0.5, 0.5, 0.0, 0.0, "fallback")

        positive_hits = sum(token in self.POSITIVE_TERMS for token in tokens)
        negative_hits = sum(token in self.NEGATIVE_TERMS for token in tokens)
        scored_hits = positive_hits + negative_hits
        compound = (positive_hits - negative_hits) / max(scored_hits, 1)
        positive_ratio = positive_hits / max(scored_hits, 1) if scored_hits else 0.5
        negative_ratio = negative_hits / max(scored_hits, 1) if scored_hits else 0.5
        return SentimentResult(
            compound=float(compound),
            compound_scaled=float(np.clip((compound + 1.0) / 2.0, 0.0, 1.0)),
            positive_ratio=float(positive_ratio),
            negative_ratio=float(negative_ratio),
            token_count=float(len(tokens)),
            magnitude=float(abs(compound)),
            source="fallback",
        )


def sentiment_features_from_text(value: Any, analyzer: TransferSentimentAnalyzer | None = None) -> dict[str, float | str]:
    active_analyzer = analyzer or TransferSentimentAnalyzer()
    result = active_analyzer.analyze(value)
    return {
        "text_sentiment_compound": result.compound,
        "text_sentiment_compound_scaled": result.compound_scaled,
        "text_sentiment_positive_ratio": result.positive_ratio,
        "text_sentiment_negative_ratio": result.negative_ratio,
        "text_sentiment_token_count": result.token_count,
        "text_sentiment_magnitude": result.magnitude,
        "text_sentiment_source": result.source,
    }
