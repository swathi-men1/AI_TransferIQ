from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

cricket_positive_words = {
    "dangerous": 2.5,
    "killer": 2.5,
    "deadly": 2.5,
    "brilliant": 3.0,
    "matchwinner": 3.0,
    "explosive": 2.8,
    "powerful": 2.5,
    "fantastic": 3.0,
    "great": 2.7,
    "excellent": 3.0,
    "dominant": 2.5,
    "aggressive": 2.0,
    "masterclass": 3.0,
    "unplayable": 3.0,
    "worldclass": 3.0
}

analyzer.lexicon.update(cricket_positive_words)

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]
