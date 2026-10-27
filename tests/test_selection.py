import json
from datetime import date, timedelta

import pandas as pd
import pytest

from retro_analytics import paths, pricecharting, selection, settings
from test_pricecharting import CONSOLE_URL, FakeSession, choice, fixture

TODAY = date(2026, 10, 26)
GAME_URL = "https://www.pricecharting.com/game/nes/{}"
SLUGS = ["legend-of-zelda", "super-mario-bros-3", "mike-tyson%27s-punch-out", "metroid"]


def volume_page(cib):
    """The saved volume row with the CIB cell's text replaced."""
    return fixture("game_nes_legend-of-zelda_volume.html").replace("1 sale per day", cib)


def pages_with_volumes(**by_slug):
    pages = {CONSOLE_URL: fixture("console_nes.html")}
    for slug, cib in by_slug.items():
        pages[GAME_URL.format(slug.replace("_", "-"))] = volume_page(cib)
    return pages


@pytest.mark.parametrize("text, per_day", [
    ("1 sale per day", 1.0),
    ("5 sales per day", 5.0),
    ("3 sales per week", 3 / 7),
    ("2 sales per month", 2 / 30),
    ("1 sale per year", 1 / 365),
    ("1,200 sales per year", 1200 / 365),
])
def test_cib_volume_is_converted_to_sales_per_day(text, per_day):
    assert selection.parse_cib_volume(volume_page(text)) == pytest.approx(per_day)


def test_cib_volume_reads_the_cib_cell_not_loose_or_new():
    html = fixture("game_nes_legend-of-zelda_volume.html")
    assert "5 sales per day" in html and "1 sale per week" in html   # other conditions differ from CIB
    assert selection.parse_cib_volume(html) == 1.0


@pytest.mark.parametrize("html", ["<html></html>", volume_page("no sales yet"), "<tr class='sales_volume'></tr>"])
def test_missing_cib_volume_is_none(html):
    assert selection.parse_cib_volume(html) is None


def test_seed_keeps_the_games_that_trade_most_often_not_the_first_on_the_page():
    pages = pages_with_volumes(**{"legend_of_zelda": "1 sale per week", "super_mario_bros_3": "4 sales per day",
                                  "mike_tyson%27s_punch_out": "1 sale per month", "metroid": "2 sales per day"})
    session = FakeSession(pages)
    games, failures = pricecharting.seed_games(session, {"nes": choice(top=2)}, today=TODAY)
    assert failures == []
    assert games["game"].tolist() == ["super-mario-bros-3", "metroid"]


def test_only_the_first_candidates_are_ranked():
    pages = pages_with_volumes(legend_of_zelda="1 sale per week", super_mario_bros_3="1 sale per month",
                               metroid="50 sales per day")
    session = FakeSession(pages)
    games, _ = pricecharting.seed_games(session, {"nes": choice(top=1, candidates=2)}, today=TODAY)
    assert games["game"].tolist() == ["legend-of-zelda"]
    assert GAME_URL.format("metroid") not in session.requested   # beyond the candidate pool: never fetched


def test_ties_and_unreadable_pages_fall_back_to_page_order():
    # Zelda's page is dead and Metroid shows no volume: both count as 0, behind Mario's 1 per day.
    pages = pages_with_volumes(super_mario_bros_3="1 sale per day", **{"mike_tyson%27s_punch_out": "1 sale per day"})
    pages[GAME_URL.format("metroid")] = "<html></html>"
    games, _ = pricecharting.seed_games(FakeSession(pages), {"nes": choice(top=4)}, today=TODAY)
    assert games["game"].tolist() == ["super-mario-bros-3", "mike-tyson%27s-punch-out", "legend-of-zelda", "metroid"]


def test_second_seed_reads_volumes_from_the_cache():
    pages = pages_with_volumes(legend_of_zelda="1 sale per day", super_mario_bros_3="2 sales per day",
                               **{"mike_tyson%27s_punch_out": "1 sale per week"}, metroid="1 sale per week")
    first = FakeSession(pages)
    pricecharting.seed_games(first, {"nes": choice(top=2)}, today=TODAY)
    assert len([url for url in first.requested if "/game/" in url]) == 4

    second = FakeSession({CONSOLE_URL: pages[CONSOLE_URL]})
    games, _ = pricecharting.seed_games(second, {"nes": choice(top=2)}, today=TODAY + timedelta(days=30))
    assert second.requested == [CONSOLE_URL]
    assert games["game"].tolist() == ["super-mario-bros-3", "legend-of-zelda"]


def test_stale_cache_entries_are_fetched_again_and_failures_are_not_cached():
    old = (TODAY - timedelta(days=selection.CACHE_DAYS + 1)).isoformat()
    selection.save_cache({GAME_URL.format("metroid"): {"per_day": 9.0, "date": old}})
    assert selection.load_cache(TODAY) == {}

    session = FakeSession({CONSOLE_URL: fixture("console_nes.html")})   # every game page fails
    pricecharting.seed_games(session, {"nes": choice(top=2)}, today=TODAY)
    assert json.loads(paths.volumes().read_text()) == {}


def test_corrupt_cache_file_is_ignored():
    paths.volumes().write_text("{not json", encoding="utf-8")
    assert selection.load_cache(TODAY) == {}


def test_cache_is_saved_even_when_a_later_console_fails():
    pages = pages_with_volumes(legend_of_zelda="1 sale per day")
    pricecharting.seed_games(FakeSession(pages), {"nes": choice(top=1, candidates=1), "snes": choice()}, today=TODAY)
    assert list(json.loads(paths.volumes().read_text())) == [GAME_URL.format("legend-of-zelda")]


def test_seed_applies_the_consoles_price_floor():
    # Only Zelda ($150) and Tyson ($1250.50) clear $100; Tyson trades more often.
    pages = pages_with_volumes(legend_of_zelda="1 sale per day", **{"mike_tyson%27s_punch_out": "3 sales per day"})
    pricecharting.seed(consoles={"nes": choice(top=1, min_price=100)}, session=FakeSession(pages))
    assert pd.read_csv(paths.games())["game"].tolist() == ["mike-tyson%27s-punch-out"]
