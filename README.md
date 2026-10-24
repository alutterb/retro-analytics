# retro-analytics

Ranks retro video games by how much their complete-in-box (CIB) price on
[PriceCharting](https://www.pricecharting.com) is forecast to grow over the
next six months, and writes the ranking as a single HTML page you open in a
browser. Reddit sentiment is optional and off by default; when collected, it
only breaks ties.

It is a personal tool: it runs on your machine, keeps all scraped data local,
and is meant to refresh once a week.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 or newer; development uses 3.13.

```sh
uv sync                        # install the pinned dependencies
uv run ra seed                 # pick the games to track (21 consoles, top 25 each, CIB >= $20)
uv run ra run                  # scrape, forecast, rank and write the report
xdg-open data/outputs/report.html
```

The first `ra seed` reads 21 console pages and the first `ra run` reads one
page per tracked game (about 550 pages, 1-2 s apart, so 15-20 minutes).
`ra seed --console nes --top 10` or `ra seed --game nes/legend-of-zelda`
tracks fewer games.

### Three-game example

`examples/three-games.sh` runs the whole pipeline on NES The Legend of Zelda,
SNES Chrono Trigger and Genesis Sonic the Hedgehog (three page requests) into
`data/example/`, separate from your main tracked set:

```sh
./examples/three-games.sh
```

It prints where `results.csv` and `report.html` were written.

## Output

- `data/outputs/report.html`: one self-contained page with no external files.
  The table is sortable by clicking any column header and filterable by
  console. Each row has the latest price, the six-month forecast, the forecast
  growth, the sentiment score when collected, and a sparkline of the last 24
  monthly prices (solid) followed by the forecast (dashed). Game names link to
  their PriceCharting page.
- `data/outputs/results.csv`: the same ranking as CSV, best first, with
  columns `rank, console, game, title, last_month, last_price,
  forecast_month, forecast_price, growth_pct, sentiment, sentiment_texts`.

Forecasts are statistical extrapolations of past prices, not advice.

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
   If no game is selected at all, `games.csv` is not written and the command
   exits non-zero; try a lower `--min-price` or other consoles.
2. `ra scrape` (`pricecharting.py`): fetches each seeded game's page once and
   reads its full monthly CIB price history from the chart data embedded in
   the page, merging it into `data/raw/prices.csv` (created on the first run).
   If `games.csv` does not exist yet, as on a fresh checkout, it first seeds it
   with the `ra seed` defaults, so `ra scrape` alone works from an empty data
   directory (about 550 requests, so expect 15-20 minutes); if that seeding
   selects no games, it stops with an error and writes neither file.
   `--snapshot` also records the price shown today. If any page fails or no longer has chart data, the
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
8. `ra report` (`report.py`): writes `data/outputs/report.html` from the
   ranking, the monthly series and the forecasts.

`ra run` runs scrape, prep, train, forecast, rank and report in order on the
games already seeded, price-only. `ra run --sentiment` also runs the sentiment
stage before ranking, and `ra run --game console/slug` (repeatable) re-seeds
exactly those games first. If some games fail to scrape it still ranks the rest, then
exits non-zero. Any stage whose input is missing exits non-zero and names the
stage to run first.

## Weekly refresh

PriceCharting's history is monthly, so a weekly run is plenty. Two options:

**systemd user timer** (Linux). The units in `deploy/systemd/` run `ra run`
every Sunday at 06:00 local time, catching up after the machine was off.
Edit `WorkingDirectory` in the service if the repository is not at
`~/Documents/Projects/retro-analytics`, then:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/systemd/retro-analytics.service deploy/systemd/retro-analytics.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now retro-analytics.timer
systemctl --user list-timers retro-analytics.timer   # next run
journalctl --user -u retro-analytics.service          # last run's output
```

To keep it running while you are logged out, also run
`loginctl enable-linger "$USER"` once.

**cron**. Add this line with `crontab -e`, adjusting the path:

```cron
0 6 * * 0  cd "$HOME/Documents/Projects/retro-analytics" && "$HOME/.local/bin/uv" run --frozen ra run >> data/cron.log 2>&1
```

Re-run `ra seed` now and then (for example monthly) if you want the tracked
set to follow PriceCharting's popularity order; the weekly job keeps the games
you already track.

## Reddit sentiment (optional)

Since late 2025 Reddit reportedly requires approval for API access, so the
pipeline never needs it. If you have approved credentials:

```sh
uv sync --extra sentiment
echo '{"client_id": "...", "secret_key": "..."}' > data/reddit_credentials.json
uv run ra run --sentiment
```

Sentiment only reorders games whose forecast growth is equal to 0.1 percentage
points, so it cannot outweigh a real price difference.

## Android app

`android/` holds a personal Android app that shows the rising and crashing
games on your phone. It downloads the results file `ra run` publishes to your
private results repository; see [android/README.md](android/README.md) to
build it, install it and paste in a read-only token.

## PriceCharting etiquette

The scraper is for private use. It sends one request at a time, waits 1-2 s
between requests, identifies itself with a descriptive User-Agent, times out
after 30 s and backs off on errors. PriceCharting's terms treat its current and
historical prices as its property, so keep scraped data local: don't commit
it, publish it or serve the report from a public site. The report credits and
links PriceCharting. Tests run on trimmed copies of real pages in
`tests/fixtures/` whose prices were replaced with made-up values.

## Development

```sh
uv sync --all-extras
uv run pytest
```

Tests use synthetic data and saved pages; the only network they touch is a
local test server. CI runs them on Python 3.12 and 3.13.

- `src/retro_analytics/`: the package. `paths.py` holds every file location;
  `cli.py` is the `ra` entry point; `pipeline.py` chains the stages;
  `report.py` renders the HTML page.
- `tests/`: the pytest suite and its HTML fixtures.
- `examples/`: the three-game end-to-end example.
- `deploy/systemd/`: the weekly timer.
