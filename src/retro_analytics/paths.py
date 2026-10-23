"""Locations of every file the pipeline reads or writes.

All data lives under one root: the ``RA_DATA_DIR`` environment variable if set,
otherwise ``data/`` in the project directory. Fitted models live in ``models/``
next to it. Parent directories are created when a path is requested, so a fresh
checkout needs no setup.
"""

import os
from pathlib import Path

from retro_analytics import PipelineError

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


def games() -> Path:
    return raw("games.csv")


def reddit_posts() -> Path:
    return raw("reddit_posts.csv")


def reddit_comments() -> Path:
    return raw("reddit_comments.csv")


def reddit_credentials() -> Path:
    return _ensure(data_dir() / "reddit_credentials.json")


def series() -> Path:
    return preprocessed("series.csv")


def sentiment() -> Path:
    return preprocessed("sentiment.csv")


def model_specs() -> Path:
    return _ensure(models_dir() / "model_specs.csv")


def forecasts() -> Path:
    return output("forecasts.csv")


def results() -> Path:
    return output("results.csv")


def report() -> Path:
    return output("report.html")


def require(path: Path, stage: str) -> Path:
    """`path`, or a PipelineError naming the stage that produces it."""
    if not path.exists():
        raise PipelineError(f"{path} not found; run `ra {stage}` first")
    return path
