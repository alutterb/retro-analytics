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

1. `ra seed` (`pricecharting.py`): chooses which games to track and writes
   `data/raw/games.csv`. By default it reads the console page of each of the 21
   tracked consoles and keeps the 25 most popular games with a CIB price of at
   least $20, skipping hardware and accessories. `--console`, `--top` and
   `--min-price` change that; `--game nes/legend-of-zelda` (repeatable) tracks
   exactly the games given without reading console pages. The file is plain
   CSV, so you can also edit it by hand. If a console page fails, that
   console's previously seeded rows are kept and the command exits non-zero.
   If no game is selected at all, `games.csv` is not written and the command
   exits non-zero; try a lower `--min-price` or other consoles.
2. `ra scrape` (`pricecharting.py`): fetches each seeded game's page once and
   reads its full monthly CIB price history from the chart data embedded in
   the page, merging it into `data/raw/prices.csv` (created on the first run).
   If `games.csv` does not exist yet, as on a fresh checkout, it first seeds it
   with the `ra seed` defaults, so `ra scrape` alone works from an empty data
   directory (about 550 requests, so expect 15-20 minutes); if that seeding
   selects no games, it stops with an error and writes neither file. `--snapshot` also records
   the price shown today. If any page fails or no longer has chart data, the
   other games are still saved and the command exits non-zero with a count of
   failures.
3. `ra sentiment` (`reddit.py`, `sentiment.py`, needs the `sentiment` extra and
   `data/reddit_credentials.json`): searches Reddit for each game and scores
   posts and comments with NLTK VADER.
4. `ra prep` (`time_series.py`, `arima_prep.py`): groups prices into one series
   per game.
5. `ra train` (`training.py`): fits an ARIMA model per game, choosing the order
   by AIC.
6. `ra forecast` (`forecast.py`): forecasts each model five steps ahead.
7. `ra rank` (`metrics.py`): combines forecast slope, percent change and
   sentiment into `data/outputs/results.csv`, best first. It needs the
   sentiment stage's output, and games without Reddit data are left out.

`ra run` runs scrape, prep, train, forecast and rank in order on the games
already seeded. If some games fail to scrape it still ranks the rest, then
exits non-zero. It does not run the sentiment stage, so run `ra sentiment`
first when you want fresh Reddit data.

## PriceCharting etiquette

The scraper is for private use. It sends one request at a time, waits 1-2 s
between requests, identifies itself with a descriptive User-Agent, times out
after 30 s and backs off on errors. PriceCharting's terms treat its current and
historical prices as its property, so keep scraped data local: don't commit
it, publish it or serve it from a public app. Tests run on trimmed copies of
real pages in `tests/fixtures/` whose prices were replaced with made-up values.

## Layout

- `src/retro_analytics/`: the package. `paths.py` holds every file location;
  `cli.py` is the `ra` entry point; `pipeline.py` chains the stages.
- `tests/`: pytest suite on synthetic data, no network.
