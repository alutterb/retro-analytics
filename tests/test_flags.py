import pandas as pd
import pytest

from retro_analytics import flags, settings

DEFAULTS = settings.DEFAULT_THRESHOLDS


def series_of(console, game, prices, end="2026-10-01"):
    months = pd.date_range(end=end, periods=len(prices), freq="MS").strftime("%Y-%m-%d")
    return pd.DataFrame({"console": console, "game": game, "month": months, "price": prices})


def results_of(*rows):
    return pd.DataFrame(rows, columns=["console", "game", "growth_pct"])


def flag(series, results, thresholds=DEFAULTS):
    return {(f["console"], f["game"]): f for f in flags.flag_games(results, series, thresholds)}


def test_realized_changes_compare_latest_price_with_k_months_earlier():
    prices = series_of("nes", "zelda", [100.0, 90.0, 85.0, 75.0, 80.0, 80.0, 60.0]).set_index("month")["price"]
    changes = flags.realized_changes(prices)
    assert changes[1] == pytest.approx(-25.0)
    assert changes[3] == pytest.approx(-20.0)
    assert changes[6] == pytest.approx(-40.0)


def test_recent_crash_flags_crashing_only():
    # Covers AE1: forecast +4%, 100 -> 80 over 3 months.
    series = series_of("nes", "zelda", [100.0, 95.0, 88.0, 80.0])
    flagged = flag(series, results_of(("nes", "zelda", 4.0)))
    reasons = flagged[("nes", "zelda")]["reasons"]
    assert [(r["direction"], r["signal"]) for r in reasons] == [("crashing", "change_3m")]
    assert reasons[0]["text"] == "down 20% in 3 months"
    assert reasons[0]["value_pct"] == pytest.approx(-20.0)
    assert reasons[0]["threshold_pct"] == -15.0


def test_a_game_can_be_rising_and_crashing():
    # Covers AE2: fell 12% last month, forecast +15%.
    series = series_of("snes", "chrono", [100.0, 100.0, 100.0, 88.0])
    reasons = flag(series, results_of(("snes", "chrono", 15.0)))[("snes", "chrono")]["reasons"]
    by_direction = {r["direction"]: r for r in reasons}
    assert by_direction["crashing"]["signal"] == "change_1m"
    assert by_direction["crashing"]["text"] == "down 12% in 1 month"
    assert by_direction["rising"]["signal"] == "forecast"
    assert by_direction["rising"]["text"] == "forecast up 15% in 6 months"


def test_short_history_skips_windows_it_cannot_cover():
    # Covers AE3: four months of prices, so no 6-month change.
    series = series_of("nes", "metroid", [50.0, 50.0, 50.0, 50.0])
    changes = flags.realized_changes(series.set_index("month")["price"])
    assert changes[6] is None
    assert changes[3] == pytest.approx(0.0)


def test_threshold_is_inclusive():
    series = series_of("nes", "zelda", [100.0] * 7)
    flagged = flag(series, results_of(("nes", "zelda", -10.0)))
    assert [r["signal"] for r in flagged[("nes", "zelda")]["reasons"]] == ["forecast"]
    assert flagged[("nes", "zelda")]["reasons"][0]["text"] == "forecast down 10% in 6 months"


def test_quiet_game_is_not_flagged():
    series = series_of("nes", "zelda", [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    assert flag(series, results_of(("nes", "zelda", 3.0))) == {}


def test_realized_rise_flags_rising():
    series = series_of("genesis", "sonic", [100.0, 100.0, 100.0, 130.0])
    reasons = flag(series, results_of(("genesis", "sonic", 0.0)))[("genesis", "sonic")]["reasons"]
    assert [(r["direction"], r["signal"], r["text"]) for r in reasons] == [
        ("rising", "change_1m", "up 30% in 1 month"),
        ("rising", "change_3m", "up 30% in 3 months"),
    ]


def test_custom_thresholds_change_the_outcome():
    series = series_of("nes", "zelda", [100.0] * 7)
    results = results_of(("nes", "zelda", 6.0))
    assert flag(series, results) == {}
    custom = {"rising": dict(DEFAULTS["rising"], forecast_pct=5.0), "crashing": DEFAULTS["crashing"]}
    assert [r["signal"] for r in flag(series, results, custom)[("nes", "zelda")]["reasons"]] == ["forecast"]


def test_flagged_record_carries_changes():
    series = series_of("nes", "zelda", [100.0, 100.0, 100.0, 80.0])
    record = flag(series, results_of(("nes", "zelda", 0.0)))[("nes", "zelda")]
    assert record["changes"][1] == pytest.approx(-20.0)
    assert record["changes"][6] is None
