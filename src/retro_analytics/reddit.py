"""Collect Reddit posts and comments about each tracked game (optional).

Reddit's API needs approved credentials. Put them in
``data/reddit_credentials.json`` as ``{"client_id": ..., "secret_key": ...}``.
Nothing here runs unless you call ``ra sentiment`` (or ``ra run --sentiment``),
and nothing collected may be committed or shared.
"""

import json

import pandas as pd
import praw

from retro_analytics import PipelineError, paths

SUBREDDITS = ('gamecollecting', 'retrogaming')
LIMIT = 25
USER_AGENT = "python:retro-analytics:0.2 (personal price research)"

# How collectors write each console's name in a post.
CONSOLE_NAMES = {
    'nes': 'NES', 'super-nintendo': 'SNES', 'nintendo-64': 'N64', 'gamecube': 'GameCube',
    'wii': 'Wii', 'nintendo-switch': 'Switch', 'gameboy': 'Game Boy', 'gameboy-color': 'Game Boy Color',
    'gameboy-advance': 'GBA', 'nintendo-ds': 'DS', 'sega-genesis': 'Genesis', 'sega-cd': 'Sega CD',
    'sega-dreamcast': 'Dreamcast', 'playstation': 'PS1', 'playstation-2': 'PS2', 'playstation-3': 'PS3',
    'playstation-4': 'PS4', 'psp': 'PSP', 'xbox': 'Xbox', 'xbox-360': 'Xbox 360', 'xbox-one': 'Xbox One',
}

POST_COLUMNS = ['console', 'game', 'post_id', 'title', 'body', 'score']
COMMENT_COLUMNS = ['console', 'game', 'post_id', 'comment_id', 'score', 'comment_text', 'timestamp']


def load_credentials():
    path = paths.reddit_credentials()
    if not path.exists():
        raise PipelineError(f"Reddit credentials not found at {path}; sentiment is optional, skip `ra sentiment`")
    creds = json.loads(path.read_text())
    return creds['client_id'], creds['secret_key']


def make_client():
    client_id, secret_key = load_credentials()
    return praw.Reddit(client_id=client_id, client_secret=secret_key, user_agent=USER_AGENT)


def search_query(console, title):
    """The game's title as a phrase, plus its console, so ports don't mix."""
    return f'"{title}" {CONSOLE_NAMES.get(console, console.replace("-", " "))}'


def collect(games, reddit, subreddits=SUBREDDITS, limit=LIMIT):
    """Posts and top-level-expanded comments for each game. Returns (posts, comments)."""
    posts, comments = [], []
    for game in games.itertuples(index=False):
        query = search_query(game.console, game.title)
        print(f"Searching Reddit for {query}")
        for sub in subreddits:
            for post in reddit.subreddit(sub).search(query, limit=limit):
                posts.append((game.console, game.game, post.id, post.title,
                              post.selftext if post.is_self else None, post.score))
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    comments.append((game.console, game.game, post.id, comment.id, comment.score,
                                     comment.body, comment.created_utc))
    return pd.DataFrame(posts, columns=POST_COLUMNS), pd.DataFrame(comments, columns=COMMENT_COLUMNS)


def main(reddit=None):
    games = pd.read_csv(paths.require(paths.games(), "seed"))
    posts, comments = collect(games, reddit or make_client())
    posts.to_csv(paths.reddit_posts(), index=False)
    comments.to_csv(paths.reddit_comments(), index=False)
    print(f"Saved {len(posts)} posts and {len(comments)} comments")


if __name__ == "__main__":
    main()
