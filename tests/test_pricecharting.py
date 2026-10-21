import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest
import requests

from retro_analytics import cli, paths, pricecharting

FIXTURES = Path(__file__).parent / "fixtures"
CONSOLE_URL = "https://www.pricecharting.com/console/nes"
ZELDA_URL = "https://www.pricecharting.com/game/nes/legend-of-zelda"


def fixture(name):
    return (FIXTURES / name).read_text()


class FakeSession:
    """Serves saved pages by URL; anything else fails like a dead link."""

    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        if url not in self.pages:
            raise requests.HTTPError(f"404 for {url}")
        return self.pages[url]


def test_console_page_links_are_absolute_and_well_formed():
    rows = pricecharting.parse_console_page(fixture("console_nes.html"), CONSOLE_URL)
    zelda = rows[0]
    assert zelda == {
        "game": "legend-of-zelda",
        "title": "Legend of Zelda",
        "url": ZELDA_URL,
        "cib_price": 150.0,
    }
    assert all(row["url"].startswith("https://www.pricecharting.com/game/nes/") for row in rows)
    assert not any("pricecharting.comhttps" in row["url"] for row in rows)


def test_console_page_prices_handle_commas_and_blanks():
    rows = {row["game"]: row for row in pricecharting.parse_console_page(fixture("console_nes.html"), CONSOLE_URL)}
    assert rows["mike-tyson%27s-punch-out"]["cib_price"] == 1250.5
    assert rows["zelda-test-cartridge"]["cib_price"] is None


def test_select_games_skips_hardware_cheap_and_unpriced_items():
    rows = pricecharting.parse_console_page(fixture("console_nes.html"), CONSOLE_URL)
    picked = [row["game"] for row in pricecharting.select_games(rows, top=10, min_price=20)]
    # Page order is kept; console, controller, Game Genie, the unpriced test cart and $15 Tetris are out.
    assert picked == ["legend-of-zelda", "super-mario-bros-3", "mike-tyson%27s-punch-out", "metroid"]
    assert [row["game"] for row in pricecharting.select_games(rows, top=2, min_price=20)] == picked[:2]


@pytest.mark.parametrize("title, expected", [
    ("Legend of Zelda", True),
    ("Nintendo NES Console", False),
    ("NES Advantage Controller", False),
    ("Sega Genesis Model 2 System", False),
    ("Game Genie Top Loader Adaptor", False),
])
def test_is_game(title, expected):
    assert pricecharting.is_game(title) is expected


def test_cib_history_from_chart_data():
    history = pricecharting.parse_cib_history(fixture("game_nes_legend-of-zelda.html"))
    # Two leading zero months are dropped; cents become dollars; timestamps become dates.
    assert len(history) == 12
    assert history[0] == (date(2025, 11, 1), 120.0)
    assert history[-1] == (date(2026, 10, 1), 151.25)
    assert [day for day, _ in history] == sorted(day for day, _ in history)


def test_current_price():
    assert pricecharting.parse_current_price(fixture("game_nes_legend-of-zelda.html")) == 151.25


@pytest.mark.parametrize("html, message", [
    ("<html><body>no chart here</body></html>", "no VGPC.chart_data"),
    ("<script>VGPC.chart_data = {broken;</script>", "not valid JSON"),
    ('<script>VGPC.chart_data = {"used": [[1, 100]]};</script>', "no 'cib' series"),
])
def test_cib_history_missing_or_malformed_is_an_error(html, message):
    with pytest.raises(pricecharting.ScrapeError, match=message):
        pricecharting.parse_cib_history(html)


def test_seed_from_console_pages_writes_games_and_reports_failures():
    session = FakeSession({CONSOLE_URL: fixture("console_nes.html")})
    with pytest.raises(pricecharting.ScrapeError, match="1 of 2 console pages failed: snes"):
        pricecharting.seed(consoles=["nes", "snes"], top=2, min_price=20, session=session)

    games = pd.read_csv(paths.games())
    assert list(games.columns) == pricecharting.GAME_COLUMNS
    assert games[["console", "game"]].values.tolist() == [["nes", "legend-of-zelda"], ["nes", "super-mario-bros-3"]]
    assert session.requested == [CONSOLE_URL, "https://www.pricecharting.com/console/snes"]


def test_seed_keeps_previous_rows_for_failed_consoles():
    pricecharting.explicit_games(["snes/chrono-trigger", "nes/old-game"]).to_csv(paths.games(), index=False)
    session = FakeSession({CONSOLE_URL: fixture("console_nes.html")})
    with pytest.raises(pricecharting.ScrapeError, match="snes"):
        pricecharting.seed(consoles=["nes", "snes"], top=2, min_price=20, session=session)

    games = pd.read_csv(paths.games())
    assert games[["console", "game"]].values.tolist() == [
        ["nes", "legend-of-zelda"], ["nes", "super-mario-bros-3"], ["snes", "chrono-trigger"]]


def test_seed_explicit_games_makes_no_requests():
    session = FakeSession({})
    pricecharting.seed(games=["nes/legend-of-zelda", "super-nintendo/chrono-trigger"], session=session)
    games = pd.read_csv(paths.games())
    assert games["url"].tolist() == [ZELDA_URL, "https://www.pricecharting.com/game/super-nintendo/chrono-trigger"]
    assert session.requested == []


@pytest.mark.parametrize("spec", ["nes", "nes/", "/zelda", "nes/game/extra"])
def test_explicit_games_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        pricecharting.explicit_games([spec])


def test_scrape_backfills_history_and_fails_loudly_on_bad_games():
    pricecharting.explicit_games(["nes/legend-of-zelda", "nes/missing-game"]).to_csv(paths.games(), index=False)
    session = FakeSession({ZELDA_URL: fixture("game_nes_legend-of-zelda.html")})

    with pytest.raises(pricecharting.ScrapeError, match="1 of 2 games failed: nes/missing-game"):
        pricecharting.main(session=session)

    # The good game is still written.
    prices = pd.read_csv(paths.prices())
    assert list(prices.columns) == pricecharting.PRICE_COLUMNS
    assert len(prices) == 12
    assert set(prices["game"]) == {"legend-of-zelda"}
    assert prices.iloc[-1].tolist() == ["nes", "legend-of-zelda", 151.25, "2026-10-01"]


def test_scrape_snapshot_adds_todays_price_and_merges_without_duplicates():
    pricecharting.explicit_games(["nes/legend-of-zelda"]).to_csv(paths.games(), index=False)
    session = FakeSession({ZELDA_URL: fixture("game_nes_legend-of-zelda.html")})
    games = pd.read_csv(paths.games())

    first, _ = pricecharting.scrape_prices(games, session, snapshot=True, today=date(2026, 10, 21))
    second, _ = pricecharting.scrape_prices(games, session, snapshot=True, today=date(2026, 10, 28))
    merged = pricecharting.merge_prices(first, second)

    assert len(first) == 13
    assert first.iloc[-1]["date"] == "2026-10-21"
    # Same monthly history twice plus two distinct snapshots.
    assert len(merged) == 14
    assert not merged.duplicated(["console", "game", "date"]).any()


def test_scrape_without_seed_explains_what_to_do():
    with pytest.raises(pricecharting.ScrapeError, match="run `ra seed` first"):
        pricecharting.main(session=FakeSession({}))


def test_cli_turns_scrape_errors_into_exit_code(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["scrape"])
    assert exit_info.value.code == 1
    assert "run `ra seed` first" in capsys.readouterr().err


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status=200, text="ok"):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_polite_session_spaces_requests_and_sets_user_agent_and_timeout(monkeypatch):
    clock = FakeClock()
    session = pricecharting.PoliteSession(delay=(1.0, 2.0), sleep=clock.sleep, clock=clock)
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout, session.session.headers["User-Agent"]))
        return FakeResponse()

    monkeypatch.setattr(session.session, "get", fake_get)
    session.get("https://example.test/a")
    session.get("https://example.test/b")

    assert clock.sleeps and 1.0 <= clock.sleeps[0] <= 2.0
    assert len(clock.sleeps) == 1  # no wait before the first request
    assert all(timeout == (10, 30) for _, timeout, _ in calls)
    assert all(agent.startswith("retro-analytics/") for _, _, agent in calls)


def test_polite_session_raises_on_http_errors(monkeypatch):
    session = pricecharting.PoliteSession()
    monkeypatch.setattr(session.session, "get", lambda url, timeout: FakeResponse(status=404))
    with pytest.raises(requests.HTTPError):
        session.get("https://example.test/missing")


def test_polite_session_retries_server_errors():
    hits = []

    class Flaky(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.headers["User-Agent"])
            status = 503 if len(hits) == 1 else 200
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"recovered" if status == 200 else b"busy")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Flaky)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session = pricecharting.PoliteSession(delay=(0, 0))
        assert session.get(f"http://127.0.0.1:{server.server_port}/game") == "recovered"
    finally:
        server.shutdown()
    assert len(hits) == 2
    assert all(agent.startswith("retro-analytics/") for agent in hits)
