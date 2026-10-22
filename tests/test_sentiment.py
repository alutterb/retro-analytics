import json
from types import SimpleNamespace

import pandas as pd
import pytest

from retro_analytics import PipelineError, paths, reddit, sentiment


class FakeAnalyzer:
    """Scores 'great' as positive and 'awful' as negative, like VADER would."""

    def polarity_scores(self, text):
        words = text.lower().split()
        return {"compound": 0.5 * words.count("great") - 0.5 * words.count("awful")}


def posts_frame():
    return pd.DataFrame({
        "console": ["nes", "nes", "super-nintendo"],
        "game": ["zelda", "zelda", "chrono"],
        "post_id": ["a", "b", "c"],
        "title": ["great game", "awful price", "great"],
        "body": [None, "  ", "great great"],
        "score": [10, 2, 5],
    })


def comments_frame():
    return pd.DataFrame({
        "console": ["nes"],
        "game": ["zelda"],
        "post_id": ["a"],
        "comment_id": ["x"],
        "score": [1],
        "comment_text": ["great"],
        "timestamp": [0],
    })


def test_score_texts_skips_missing_and_blank():
    assert sentiment.score_texts(["great", None, "", "awful awful"], FakeAnalyzer()) == [0.5, None, None, -1.0]


def test_game_sentiment_is_mean_over_titles_bodies_and_comments():
    scores = sentiment.game_sentiment(posts_frame(), comments_frame(), FakeAnalyzer())
    assert list(scores.columns) == sentiment.SENTIMENT_COLUMNS
    scores = scores.set_index("game")
    # Zelda: titles 0.5 and -0.5 plus comment 0.5 (blank body skipped).
    assert scores.loc["zelda", "sentiment"] == pytest.approx(0.5 / 3)
    assert scores.loc["zelda", "n_texts"] == 3
    assert scores.loc["chrono", "sentiment"] == pytest.approx(0.75)
    assert scores.loc["chrono", "console"] == "super-nintendo"


def test_sentiment_main_writes_scores_file():
    posts_frame().to_csv(paths.reddit_posts(), index=False)
    comments_frame().to_csv(paths.reddit_comments(), index=False)
    sentiment.main(analyzer=FakeAnalyzer())
    assert sorted(pd.read_csv(paths.sentiment())["game"]) == ["chrono", "zelda"]


def test_make_analyzer_downloads_lexicon_once_when_missing(monkeypatch):
    attempts, downloads = [], []

    def analyzer():
        attempts.append(1)
        if not downloads:
            raise LookupError("vader_lexicon not found")
        return FakeAnalyzer()

    monkeypatch.setattr(sentiment, "SentimentIntensityAnalyzer", analyzer)
    monkeypatch.setattr(sentiment.nltk, "download", lambda name, quiet: downloads.append(name))
    assert isinstance(sentiment.make_analyzer(), FakeAnalyzer)
    assert downloads == ["vader_lexicon"]
    assert len(attempts) == 2


def test_search_query_uses_title_and_console_name():
    assert reddit.search_query("super-nintendo", "Chrono Trigger") == '"Chrono Trigger" SNES'
    assert reddit.search_query("atari-2600", "Pitfall") == '"Pitfall" atari 2600'


class FakeComments:
    def __init__(self, comments):
        self._comments = comments
        self.replaced = []

    def replace_more(self, limit):
        self.replaced.append(limit)

    def list(self):
        return self._comments


def fake_post(post_id, title, is_self=True, selftext="body", comments=()):
    return SimpleNamespace(id=post_id, title=title, is_self=is_self, selftext=selftext, score=3,
                           comments=FakeComments(list(comments)))


class FakeReddit:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def subreddit(self, name):
        def search(query, limit):
            self.queries.append((name, query, limit))
            return self.results.get(name, [])
        return SimpleNamespace(search=search)


def test_collect_keeps_comments_from_every_subreddit_and_tags_console():
    comment = SimpleNamespace(id="c1", score=2, body="great", created_utc=1.0)
    client = FakeReddit({
        "gamecollecting": [fake_post("p1", "Found Chrono Trigger", comments=[comment])],
        "retrogaming": [],  # the old code threw away earlier comments when a subreddit had no hits
    })
    games = pd.DataFrame({"console": ["super-nintendo"], "game": ["chrono-trigger"], "title": ["Chrono Trigger"]})

    posts, comments = reddit.collect(games, client, limit=5)

    assert client.queries == [("gamecollecting", '"Chrono Trigger" SNES', 5), ("retrogaming", '"Chrono Trigger" SNES', 5)]
    assert posts[["console", "game", "post_id"]].values.tolist() == [["super-nintendo", "chrono-trigger", "p1"]]
    assert comments[["console", "game", "comment_id", "comment_text"]].values.tolist() == [
        ["super-nintendo", "chrono-trigger", "c1", "great"]]


def test_reddit_without_credentials_is_a_clear_error():
    with pytest.raises(PipelineError, match="sentiment is optional"):
        reddit.load_credentials()


def test_reddit_credentials_are_read_from_data_dir():
    paths.reddit_credentials().write_text(json.dumps({"client_id": "id", "secret_key": "secret"}))
    assert reddit.load_credentials() == ("id", "secret")
