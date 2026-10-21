"""Locations of every file the pipeline reads or writes.

All data lives under one root: the ``RA_DATA_DIR`` environment variable if set,
otherwise ``data/`` in the project directory. Fitted models live in ``models/``
next to it. Parent directories are created when a path is requested, so a fresh
checkout needs no setup.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("RA_DATA_DIR", PROJECT_DIR / "data"))


def models_dir() -> Path:
    return Path(os.environ.get("RA_MODELS_DIR", PROJECT_DIR / "models"))


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def raw(name: str) -> Path:
    return _ensure(data_dir() / "raw" / name)


def preprocessed(name: str) -> Path:
    return _ensure(data_dir() / "preprocessed" / name)


def output(name: str) -> Path:
    return _ensure(data_dir() / "outputs" / name)


def prices() -> Path:
    return raw("prices.csv")


def consoles() -> Path:
    return raw("consoles.csv")


def reddit_posts() -> Path:
    return raw("reddit_posts.csv")


def reddit_comments() -> Path:
    return raw("comments.csv")


def reddit_credentials() -> Path:
    return _ensure(data_dir() / "reddit_credentials.json")


def historical_prices() -> Path:
    return preprocessed("scraped_data/historical prices.csv")


def preprocessed_posts() -> Path:
    return preprocessed("scraped_data/preprocessed_posts.csv")


def preprocessed_comments() -> Path:
    return preprocessed("scraped_data/preprocessed_comments.csv")


def arima_input() -> Path:
    return preprocessed("model_input/preprocessed_input_ARIMA.pickle")


def arima_models() -> Path:
    return _ensure(models_dir() / "ARIMA" / "models_ARIMA.pickle")


def predictions() -> Path:
    return output("predictions.pickle")


def results() -> Path:
    return output("results.csv")
