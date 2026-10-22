"""Collect complete-in-box (CIB) prices from PriceCharting's public pages.

For private use only: requests go out one at a time, 1-2 s apart, with a
descriptive User-Agent, and nothing scraped is ever committed or published.

Two steps:

- ``seed`` picks which games to track: the most popular games on each console
  page whose CIB price is above a floor (or an explicit list of games), and
  writes them to ``games.csv``.
- ``scrape`` fetches each seeded game's page once and reads its full monthly
  CIB history from the chart data embedded in it, so a new game has years of
  history from its first run.
"""

import json
import random
import re
import time
from datetime import UTC, date, datetime
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from retro_analytics import PipelineError, paths

BASE_URL = "https://www.pricecharting.com"
USER_AGENT = "retro-analytics/0.2 (personal price research; +https://github.com/alutterb/retro-analytics)"

# The 21 consoles the project has always tracked, as PriceCharting URL slugs.
CONSOLES = (
    'nes', 'super-nintendo', 'nintendo-64', 'gamecube', 'wii', 'nintendo-switch',
    'gameboy', 'gameboy-color', 'gameboy-advance', 'nintendo-ds',
    'sega-genesis', 'sega-cd', 'sega-dreamcast',
    'playstation', 'playstation-2', 'playstation-3', 'playstation-4', 'psp',
    'xbox', 'xbox-360', 'xbox-one',
)

# Console pages also list hardware and accessories; these words in a title mark them.
NON_GAME = re.compile(
    r"\b(console|system|controller|adapt[eo]r|game genie|light gun|zapper|power glove|"
    r"rob the robot|classic edition|memory card|link cable|rumble pak|expansion pak|"
    r"charger|headset|balance board|steering wheel|nunchuk)\b",
    re.IGNORECASE,
)

GAME_COLUMNS = ['console', 'game', 'title', 'url']
PRICE_COLUMNS = ['console', 'game', 'price', 'date']


class ScrapeError(PipelineError):
    pass


class PoliteSession:
    """GETs pages one at a time, spaced out, with retries and backoff on server trouble."""

    def __init__(self, delay=(1.0, 2.0), timeout=(10, 30), retries=3, sleep=time.sleep, clock=time.monotonic):
        self.delay = delay
        self.timeout = timeout
        self.sleep = sleep
        self.clock = clock
        self._last_request = None
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT
        retry = Retry(
            total=retries,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=('GET',),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def get(self, url):
        if self._last_request is not None:
            wait = random.uniform(*self.delay) - (self.clock() - self._last_request)
            if wait > 0:
                self.sleep(wait)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        finally:
            self._last_request = self.clock()
        return response.text


def parse_price(text):
    """'$1,250.50' -> 1250.5; an empty cell -> None."""
    cleaned = re.sub(r'[^0-9.]', '', text or '')
    return float(cleaned) if cleaned else None


def parse_console_page(html, page_url):
    """Every row of a console page's games table, in the page's (popularity) order."""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', attrs={'id': 'games_table'})
    if table is None:
        raise ScrapeError(f"no games table on {page_url}")

    rows = []
    for row in table.select('tbody tr'):
        link = row.select_one('td.title a')
        if link is None:
            continue
        # Links are site-relative today and were absolute before; urljoin handles both.
        url = urljoin(page_url, link['href'])
        price = row.select_one('td.cib_price span.js-price')
        rows.append({
            'game': url.rstrip('/').split('/')[-1],
            'title': link.get_text(strip=True),
            'url': url,
            'cib_price': parse_price(price.get_text()) if price else None,
        })
    return rows


def is_game(title):
    return NON_GAME.search(title) is None


def select_games(rows, top, min_price):
    """The first `top` games on the page whose CIB price is at least `min_price`."""
    picked = [
        row for row in rows
        if is_game(row['title']) and row['cib_price'] is not None and row['cib_price'] >= min_price
    ]
    return picked[:top]


def parse_current_price(html):
    soup = BeautifulSoup(html, 'html.parser')
    price = soup.select_one('#complete_price span.price.js-price')
    return parse_price(price.get_text()) if price else None


def parse_cib_history(html):
    """Monthly CIB prices from the page's embedded chart data, as (date, dollars) pairs.

    The chart stores [epoch milliseconds, cents] pairs, with 0 meaning no data.
    """
    marker = html.find('VGPC.chart_data')
    if marker == -1:
        raise ScrapeError("page has no VGPC.chart_data")
    start = html.find('{', marker)
    try:
        chart, _ = json.JSONDecoder().raw_decode(html, start)
    except ValueError as e:
        raise ScrapeError(f"VGPC.chart_data is not valid JSON: {e}") from e
    if 'cib' not in chart:
        raise ScrapeError("VGPC.chart_data has no 'cib' series")

    return [
        (datetime.fromtimestamp(ms / 1000, UTC).date(), cents / 100)
        for ms, cents in chart['cib']
        if cents
    ]


def game_url(console, game):
    return f"{BASE_URL}/game/{console}/{game}"


def explicit_games(specs):
    """Turn 'console/game-slug' strings into seed rows."""
    games = []
    for spec in specs:
        console, _, game = spec.strip('/').partition('/')
        if not console or not game or '/' in game:
            raise ValueError(f"expected console/game-slug, got {spec!r}")
        games.append({'console': console, 'game': game, 'title': game, 'url': game_url(console, game)})
    return pd.DataFrame(games, columns=GAME_COLUMNS)


def seed_games(session, consoles=CONSOLES, top=25, min_price=20.0):
    """Pick games to track from each console page. Returns (games, failures)."""
    games, failures = [], []
    for console in consoles:
        page_url = f"{BASE_URL}/console/{console}"
        print(f"Reading {page_url}")
        try:
            rows = parse_console_page(session.get(page_url), page_url)
        except (requests.RequestException, ScrapeError) as e:
            failures.append((console, str(e)))
            print(f"FAILED {console}: {e}")
            continue
        for row in select_games(rows, top, min_price):
            games.append({'console': console, 'game': row['game'], 'title': row['title'], 'url': row['url']})
    return pd.DataFrame(games, columns=GAME_COLUMNS), failures


def scrape_prices(games, session, snapshot=False, today=None):
    """Fetch each game's page once. Returns (price rows, failures).

    With `snapshot`, also records the price shown on the page as today's value.
    """
    today = today or date.today()
    rows, failures = [], []
    for game in games.itertuples(index=False):
        print(f"Fetching {game.console}/{game.game}")
        try:
            html = session.get(game.url)
            history = parse_cib_history(html)
            if not history:
                raise ScrapeError("chart data has no CIB prices")
            if snapshot:
                current = parse_current_price(html)
                if current is None:
                    raise ScrapeError("no current CIB price on the page")
                history.append((today, current))
        except (requests.RequestException, ScrapeError) as e:
            failures.append((f"{game.console}/{game.game}", str(e)))
            print(f"FAILED {game.console}/{game.game}: {e}")
            continue
        rows.extend((game.console, game.game, price, day.isoformat()) for day, price in history)
    return pd.DataFrame(rows, columns=PRICE_COLUMNS), failures


def merge_prices(existing, new):
    """Union of old and new rows; a re-scraped (console, game, date) takes the new price."""
    frames = [frame for frame in (existing, new) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['console', 'game', 'date'], keep='last')
    return merged.sort_values(['console', 'game', 'date']).reset_index(drop=True)


def raise_on_failures(failures, total, what):
    if failures:
        details = '; '.join(f"{name}: {error}" for name, error in failures)
        raise ScrapeError(f"{len(failures)} of {total} {what} failed: {details}")


def seed(consoles=CONSOLES, top=25, min_price=20.0, games=None, session=None):
    """Write games.csv, from explicit 'console/game' specs or the console pages."""
    if games:
        selected, failures = explicit_games(games), []
    else:
        selected, failures = seed_games(session or PoliteSession(), consoles, top, min_price)
    if failures and paths.games().exists():
        failed = {console for console, _ in failures}
        previous = pd.read_csv(paths.games())
        selected = pd.concat([selected, previous[previous['console'].isin(failed)]], ignore_index=True)
    if len(selected):
        selected.to_csv(paths.games(), index=False)
        print(f"Seeded {len(selected)} games into {paths.games()}")
    raise_on_failures(failures, len(consoles), "console pages")
    if not len(selected):
        raise ScrapeError("no games selected; try a lower --min-price or other consoles")


def main(snapshot=False, session=None):
    """Scrape every game in games.csv into prices.csv.

    On a fresh checkout neither file exists yet: games.csv is first seeded from
    the default console list, and prices.csv is created.
    """
    session = session or PoliteSession()
    seed_error = None
    if not paths.games().exists():
        print(f"{paths.games()} not found; seeding from the {len(CONSOLES)} default consoles")
        try:
            seed(consoles=CONSOLES, session=session)
        except ScrapeError as e:
            # Some console pages failed: scrape what was seeded, then still fail.
            if not paths.games().exists():
                raise
            seed_error = e
    games = pd.read_csv(paths.games())

    new, failures = scrape_prices(games, session, snapshot=snapshot)
    prices_path = paths.prices()
    existing = pd.read_csv(prices_path) if prices_path.exists() else pd.DataFrame(columns=PRICE_COLUMNS)
    merge_prices(existing, new).to_csv(prices_path, index=False)
    print(f"Wrote {len(new)} price points for {len(games) - len(failures)} games to {prices_path}")
    errors = [str(seed_error)] if seed_error else []
    try:
        raise_on_failures(failures, len(games), "games")
    except ScrapeError as e:
        errors.append(str(e))
    if errors:
        raise ScrapeError('; '.join(errors))


if __name__ == "__main__":
    main()
