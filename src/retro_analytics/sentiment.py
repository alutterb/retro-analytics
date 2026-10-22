"""Score Reddit posts and comments with NLTK's VADER and average them per game (optional)."""

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from retro_analytics import paths

SENTIMENT_COLUMNS = ['console', 'game', 'sentiment', 'n_texts']


def make_analyzer():
    """A VADER analyzer, downloading its lexicon the first time it is needed."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()


def score_texts(texts, analyzer):
    """VADER compound score (-1 to 1) per text; None for missing or blank text.

    Raw text goes straight to VADER: it handles negation, punctuation and
    capitalisation itself, so stripping stopwords would only lose signal.
    """
    return [
        analyzer.polarity_scores(text)['compound'] if isinstance(text, str) and text.strip() else None
        for text in texts
    ]


def game_sentiment(posts, comments, analyzer):
    """Mean compound score over every post title, post body and comment of each game."""
    texts = pd.concat([
        posts[['console', 'game', 'title']].rename(columns={'title': 'text'}),
        posts[['console', 'game', 'body']].rename(columns={'body': 'text'}),
        comments[['console', 'game', 'comment_text']].rename(columns={'comment_text': 'text'}),
    ], ignore_index=True)
    texts['compound'] = score_texts(texts['text'], analyzer)
    texts = texts.dropna(subset=['compound'])
    if texts.empty:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)
    grouped = texts.groupby(['console', 'game'])['compound']
    return pd.DataFrame({'sentiment': grouped.mean(), 'n_texts': grouped.size()}).reset_index()[SENTIMENT_COLUMNS]


def main(analyzer=None):
    posts = pd.read_csv(paths.require(paths.reddit_posts(), "sentiment"))
    comments = pd.read_csv(paths.require(paths.reddit_comments(), "sentiment"))
    scores = game_sentiment(posts, comments, analyzer or make_analyzer())
    scores.to_csv(paths.sentiment(), index=False)
    print(f"Scored sentiment for {len(scores)} games into {paths.sentiment()}")


if __name__ == "__main__":
    main()
