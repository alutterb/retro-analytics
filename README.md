# retro-analytics

Ranks retro video games by how much their complete-in-box (CIB) price on
[PriceCharting](https://www.pricecharting.com) is forecast to grow over the
next six months, and writes the ranking as a single HTML page you open in a
browser. It also flags which games are rising and which are crashing in price,
and can publish those to a private GitHub repository for a phone app to read.
Reddit sentiment is optional and off by default; when collected, it only breaks
ties.

It is a personal tool: it runs on your machine, keeps all scraped data local,
and is meant to refresh once a week.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 or newer; development uses 3.13.

```sh
uv sync                        # install the pinned dependencies
uv run ra seed                 # pick the games to track (21 consoles, top 25 each, CIB >= $20)
uv run ra run                  # scrape, forecast, rank, write the report and the phone export
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

It prints where `results.csv`, `report.html` and `latest.json` were written.

## Output

- `data/outputs/report.html`: one self-contained page with no external files.
  The table is sortable by clicking any column header and filterable by
  console. Each row has the latest price, the six-month forecast, the forecast
  growth, the sentiment score when collected, and a sparkline of the last 24
  monthly prices (solid) followed by the forecast (dashed). Game names link to
  their PriceCharting page.
- `data/outputs/results.csv`: the same ranking as CSV, best first, with
  columns `rank, console, game, title, last_month, last_price,
  forecast_month, forecast_price, growth_pct, sentiment, sentiment_texts,
  rebound` (`True` for [rebound](#rebound) games).
- `data/outputs/latest.json`: the games flagged rising or crashing, plus their
  rebound flag (see [Rising and crashing](#rising-and-crashing)), for the phone app.

Forecasts are statistical extrapolations of past prices, not advice.

Data is written under `data/` (or `$RA_DATA_DIR`) and the chosen model
specifications under `models/` (or `$RA_MODELS_DIR`). Every hand-off between
stages is a CSV file. Directories are created on first use. Both are
gitignored: never commit price data, Reddit data or credentials. Your settings
(`data/settings.toml`) live there too.

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
9. `ra export` (`flags.py`, `export.py`): flags rising and crashing games and
   writes them to `data/outputs/latest.json`.
10. `ra publish` (`publish.py`): commits `latest.json` to the private results
    repository set in `data/settings.toml`; does nothing when none is set.

`ra run` runs scrape, prep, train, forecast, rank, report, export and publish
in order on the games already seeded, price-only. `ra run --sentiment` also runs the sentiment
stage before ranking, and `ra run --game console/slug` (repeatable) re-seeds
exactly those games first. If some games fail to scrape it still ranks the rest, then
exits non-zero; a failed publish likewise leaves the local export in place
and exits non-zero after every other stage has run. Any stage whose input is missing exits non-zero and names the
stage to run first.

## Rising and crashing

`ra export` flags a game **rising** when its forecast growth, or its actual
CIB price change over the last 1, 3 or 6 months, is at or above that signal's
rising threshold, and **crashing** when one is at or below its crashing
threshold. A game can be both, for example one that fell 12% last month but is
forecast to recover. Each flag keeps its reason ("down 20% in 3 months",
"forecast up 15% in 6 months"). A window the game has too little history for
is skipped, not counted as no change.

The defaults:

| Signal | Rising at or above | Crashing at or below |
|---|---|---|
| Forecast growth (6 months) | +10% | -10% |
| Change over 1 month | +15% | -10% |
| Change over 3 months | +25% | -15% |
| Change over 6 months | +40% | -25% |

To change any of them, copy `examples/settings.toml` to `data/settings.toml`
(or `$RA_DATA_DIR/settings.toml`) and edit it. Every key is optional; an
unknown key is an error so a typo cannot silently keep a default.

```toml
[rising]
forecast_pct = 10
change_1m_pct = 15

[crashing]
change_3m_pct = -20
```

### Rebound

A **rebound** is a game that recently crashed while the model expects it to
climb: it meets a crashing threshold on a realized price change (1, 3 or 6
months) and its forecast growth is at or above `[rebound] forecast_pct`
(default +10%). A forecast-only crash does not count, since that forecast
points down. Rebounds are ranked by forecast growth, biggest first. The
report marks them with a badge and a "Rebound only" filter, `results.csv` has
a `rebound` column, `ra export` prints their count, and the phone app has a
Rebound tab and notifies when a game newly becomes one.

```toml
[rebound]
forecast_pct = 25   # only crashes forecast to recover by 25% or more
```

A settings file without a `[rebound]` table keeps the default.

`data/outputs/latest.json` holds only flagged games. Each has its console,
title, PriceCharting link, latest CIB price and month, its 1/3/6-month
changes, the six-month forecast, the last 24 monthly prices and its flags,
next to the thresholds used and a `schema_version` (currently 1) the app
checks. A rebound game also has a `rebound` field shaped like a flag
(`direction: "rebound"`, `value_pct` the forecast growth, `text` such as "down
25% in 1 month, forecast up 81% in 6 months"), or `null`; `flags` still holds
only rising and crashing entries, so older readers are unaffected.

## Phone export and publishing

A phone app reads `latest.json` from a private GitHub repository. To have `ra run` (and so the weekly timer) publish it there, add a
`[publish]` table to `data/settings.toml`:

```toml
[publish]
repo = "git@github.com:alutterb/retro-analytics-data.git"
branch = "main"                 # default
path = "results/latest.json"    # default
```

`ra publish` does the same on its own. It clones the branch into a temporary
directory under `data/`, replaces the file and pushes one commit, and skips
the commit when only the generation date changed. Before any git command it
asks the GitHub CLI whether the repository is private and refuses to publish
when it is public, is not on GitHub, carries credentials in its URL, or its
visibility cannot be checked. PriceCharting's prices are for personal use, so
keep this repository private.

The weekly timer can run before you log in (with linger, or catching up at
boot), when the desktop keyring that `gh auth login` normally stores its token
in is still locked. Give it credentials that need no session or prompt:

1. A token for the privacy check. Create a fine-grained GitHub token with
   read-only access to `retro-analytics-data` (the same kind the phone app
   uses; Metadata and Contents read are enough) and store it where only you
   can read it:

   ```sh
   mkdir -p ~/.config/retro-analytics
   printf 'GH_TOKEN=%s\n' 'github_pat_...' > ~/.config/retro-analytics/env
   chmod 600 ~/.config/retro-analytics/env
   ```

   then uncomment the `EnvironmentFile=` line in
   `deploy/systemd/retro-analytics.service`.
2. A deploy key for pushing, limited to the results repository:

   ```sh
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/retro_analytics_data
   gh repo deploy-key add ~/.ssh/retro_analytics_data.pub \
     --repo alutterb/retro-analytics-data --allow-write --title retro-analytics
   ```

   then uncomment the `Environment=GIT_SSH_COMMAND=...` line in
   `deploy/systemd/retro-analytics.service` before installing it, so only the
   timer uses that key. Runs from your own shell keep using your usual
   credentials.

Git and gh calls time out (5 minutes and 1 minute) and never prompt, so a
stalled network or a missing credential fails the publish instead of hanging
the timer.

Commits use your git identity (`git config --global user.name` and
`user.email`).

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
historical prices as its property, so keep scraped data to yourself: don't
commit it to this repository, publish it or serve the report from a public
site. The only place `ra publish` sends it is your own private results
repository. The report credits and
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
  `report.py` renders the HTML page; `settings.py` reads `settings.toml`;
  `flags.py`, `export.py` and `publish.py` produce and publish the phone
  export.
- `tests/`: the pytest suite and its HTML fixtures.
- `examples/`: the three-game end-to-end example.
- `deploy/systemd/`: the weekly timer.
