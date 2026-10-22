# retro-analytics

Ranks retro video games by how much their complete-in-box (CIB) price on
pricecharting.com is forecast to grow over the next six months. Reddit
sentiment is optional and off by default; when collected, it only breaks ties.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 or newer; development uses 3.13.

```sh
uv sync                      # core pipeline
uv sync --extra sentiment    # also Reddit scraping and NLTK sentiment
uv run pytest                # run the test suite
uv run ra --help             # list pipeline stages
```

Data is written under `data/` (or `$RA_DATA_DIR`) and the chosen model
specifications under `models/` (or `$RA_MODELS_DIR`). Every hand-off between
stages is a CSV file. Directories are created on first use. Both are
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
2. `ra scrape` (`pricecharting.py`): fetches each seeded game's page once and
   reads its full monthly CIB price history from the chart data embedded in
   the page, merging it into `data/raw/prices.csv`. `--snapshot` also records
   the price shown today. If any page fails or no longer has chart data, the
   other games are still saved and the command exits non-zero with a count of
   failures.
3. `ra sentiment` (optional; `reddit.py`, `sentiment.py`, needs the
   `sentiment` extra and `data/reddit_credentials.json` with `client_id` and
   `secret_key`): searches r/gamecollecting and r/retrogaming for each game's
   title plus console name, scores every post title, body and comment with
   NLTK's VADER (its lexicon downloads on first use) and writes the mean score
   per game to `data/preprocessed/sentiment.csv`. Reddit now approves API
   access case by case, so the pipeline never depends on this stage.
4. `ra prep` (`time_series.py`): turns `prices.csv` into one monthly series per
   game covering the last five years, filling months missing between two known
   prices by linear interpolation. Only games listed in `games.csv` are used,
   so games dropped by re-seeding leave the ranking but keep their history in
   `prices.csv`.
5. `ra train` (`training.py`): picks a model per game with at least 12 months
   of prices: an ARIMA(p, 1, q) on log prices with p, q <= 2 chosen by AIC and
   a drift term, so a rising price keeps rising in the forecast. When no ARIMA
   converges it uses a random walk with drift. Writes `models/model_specs.csv`.
6. `ra forecast` (`forecast.py`): refits each chosen model and forecasts six
   months ahead into `data/outputs/forecasts.csv`.
7. `ra rank` (`metrics.py`): ranks games by forecast growth, `forecast price /
   latest price - 1`, into `data/outputs/results.csv`. Games whose growth
   matches to 0.1 percentage points are ordered by sentiment; games without
   Reddit data count as neutral and are never dropped.

`ra run` runs scrape, prep, train, forecast and rank in order on the games
already seeded, price-only. `ra run --sentiment` also runs the sentiment stage
before ranking. If some games fail to scrape it still ranks the rest, then
exits non-zero. Any stage whose input is missing exits non-zero and names the
stage to run first.

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
