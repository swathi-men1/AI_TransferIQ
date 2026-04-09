"""
Sentiment Analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner)
as required by the project PDF. Analyzes simulated social media comments about
players and produces a sentiment_score for the dataset.
"""
import pandas as pd
import numpy as np
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def generate_player_comments(performance_rating, days_injured):
    """
    Simulates social media comments based on player performance/injury state.
    In a production system, these would come from the Twitter API.
    """
    positive_phrases = [
        "What a fantastic performance!", "Absolute legend.", "Best player in the league!",
        "Incredible skill and vision.", "Superb form lately.", "Transfer value skyrocketing!",
        "Can't stop watching him play.", "Club will never sell him.", "Masterclass performance.",
    ]
    negative_phrases = [
        "Terrible performance today.", "Not worth the hype.", "Always getting injured.",
        "Transfer value dropping fast.", "Badly out of form.", "Huge disappointment.",
        "Should be benched.", "Not the same player anymore.", "Waste of money.",
    ]
    neutral_phrases = [
        "Decent game.", "Average performance.", "Could do better.", "Solid if not spectacular.",
        "Workmanlike display.", "Jury is still out on him.",
    ]

    # Weight comments by performance
    if performance_rating > 78:
        pool = positive_phrases * 3 + neutral_phrases
    elif performance_rating < 65 or days_injured > 20:
        pool = negative_phrases * 3 + neutral_phrases
    else:
        pool = positive_phrases + neutral_phrases + negative_phrases

    return random.choice(pool)


def run_vader_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs VADER sentiment analysis on simulated social media posts for each
    player-month record and returns a vader_compound_score column.
    """
    print("Running VADER Sentiment Analysis on player social media data...")
    analyzer = SentimentIntensityAnalyzer()

    df = df.copy()
    df['simulated_comment'] = df.apply(
        lambda r: generate_player_comments(r['performance_rating'], r['days_injured']),
        axis=1
    )

    df['vader_compound'] = df['simulated_comment'].apply(
        lambda comment: analyzer.polarity_scores(comment)['compound']
    )

    # Update the social_sentiment_score with the NLP-derived score for richer analysis
    df['social_sentiment_score'] = (
        df['social_sentiment_score'] * 0.4 + df['vader_compound'] * 0.6
    ).clip(-1, 1)

    # Print summary
    avg_sentiment = df['vader_compound'].mean()
    pos = (df['vader_compound'] > 0.05).sum()
    neg = (df['vader_compound'] < -0.05).sum()
    neu = ((df['vader_compound'] >= -0.05) & (df['vader_compound'] <= 0.05)).sum()
    print(f"  Average Compound Sentiment: {avg_sentiment:.4f}")
    print(f"  Positive: {pos} | Neutral: {neu} | Negative: {neg}")

    return df


if __name__ == "__main__":
    df = pd.read_csv("transferiq_dataset.csv")
    df_with_sentiment = run_vader_analysis(df)
    df_with_sentiment.to_csv("transferiq_with_sentiment.csv", index=False)
    print("Sentiment analysis complete. Saved to transferiq_with_sentiment.csv")
