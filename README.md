# retro-analytics

Ranks retro video games by their forecast complete-in-box (CIB) price trend on
pricecharting.com, combined with Reddit sentiment.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 or newer; development uses 3.13.

```sh
uv sync                      # core pipeline
uv sync --extra sentiment    # also Reddit scraping and NLTK sentiment
uv run pytest                # run the test suite
uv run ra --help             # list pipeline stages
```

Data is written under `data/` (or `$RA_DATA_DIR`) and fitted models under
`models/` (or `$RA_MODELS_DIR`). Directories are created on first use. Both are
gitignored: never commit price data, Reddit data or credentials.

## Pipeline stages

Each stage is a `ra` subcommand and reads the files the previous one wrote.

1. `ra scrape` (`pricecharting.py`): appends today's CIB price for each game on
   the consoles listed in `data/raw/consoles.csv` to `data/raw/prices.csv`.
   Known issue: PriceCharting now serves absolute game links, which this
   scraper does not handle yet, so game fetches currently fail.
2. `ra sentiment` (`reddit.py`, `sentiment.py`, needs the `sentiment` extra and
   `data/reddit_credentials.json`): searches Reddit for each game and scores
   posts and comments with NLTK VADER.
3. `ra prep` (`time_series.py`, `arima_prep.py`): groups prices into one series
   per game.
4. `ra train` (`training.py`): fits an ARIMA model per game, choosing the order
   by AIC.
5. `ra forecast` (`forecast.py`): forecasts each model five steps ahead.
6. `ra rank` (`metrics.py`): combines forecast slope, percent change and
   sentiment into `data/outputs/results.csv`, best first. It needs the
   sentiment stage's output, and games without Reddit data are left out.

`ra run` runs scrape, prep, train, forecast and rank in order. It does not run
the sentiment stage, so run `ra sentiment` first when you want fresh Reddit
data.

## Layout

- `src/retro_analytics/`: the package. `paths.py` holds every file location;
  `cli.py` is the `ra` entry point; `pipeline.py` chains the stages.
- `tests/`: pytest suite on synthetic data, no network.
