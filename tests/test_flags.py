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


def test_rising_threshold_is_inclusive_too():
    flagged = flag(series_of("nes", "zelda", [100.0] * 7), results_of(("nes", "zelda", 10.0)))
    assert [(r["direction"], r["signal"]) for r in flagged[("nes", "zelda")]["reasons"]] == [("rising", "forecast")]


def test_six_month_change_alone_can_flag():
    series = series_of("nes", "zelda", [100.0, 120.0, 125.0, 130.0, 135.0, 138.0, 140.0])
    reasons = flag(series, results_of(("nes", "zelda", 0.0)))[("nes", "zelda")]["reasons"]
    assert [(r["signal"], r["text"]) for r in reasons] == [("change_6m", "up 40% in 6 months")]


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
    custom = dict(DEFAULTS, rising=dict(DEFAULTS["rising"], forecast_pct=5.0))
    assert [r["signal"] for r in flag(series, results, custom)[("nes", "zelda")]["reasons"]] == ["forecast"]


def test_flagged_record_carries_changes():
    series = series_of("nes", "zelda", [100.0, 100.0, 100.0, 80.0])
    record = flag(series, results_of(("nes", "zelda", 0.0)))[("nes", "zelda")]
    assert record["changes"][1] == pytest.approx(-20.0)
    assert record["changes"][6] is None


def rebound_of(prices, growth):
    series = series_of("dreamcast", "vmu", prices)
    return flag(series, results_of(("dreamcast", "vmu", growth)))[("dreamcast", "vmu")]["rebound"]


def test_recent_crash_with_upward_forecast_is_a_rebound():
    # Pokemon Sapphire shape: down 27% in a month, forecast +30%.
    rebound = rebound_of([100.0, 100.0, 100.0, 73.0], 30.0)
    assert rebound == {
        "direction": "rebound", "signal": "change_1m", "value_pct": 30.0, "threshold_pct": 10.0,
        "text": "down 27% in 1 month, forecast up 30% in 6 months"}


def test_rebound_cites_the_shortest_window_that_crashed():
    # Dreamcast VMU shape: down in every window; the 1-month drop is the most recent.
    rebound = rebound_of([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 57.0], 81.0)
    assert rebound["signal"] == "change_1m"
    assert rebound["text"] == "down 43% in 1 month, forecast up 81% in 6 months"
    assert rebound["value_pct"] == 81.0


def test_rebound_forecast_threshold_is_inclusive():
    prices = [100.0, 100.0, 100.0, 80.0]
    assert rebound_of(prices, 10.0)["value_pct"] == 10.0
    assert rebound_of(prices, 9.9) is None


def test_flat_forecast_after_a_crash_is_not_a_rebound():
    assert rebound_of([100.0, 100.0, 100.0, 80.0], 0.0) is None


def test_forecast_up_without_a_crash_is_not_a_rebound():
    # Rising on the forecast alone, prices steady.
    assert rebound_of([100.0] * 7, 40.0) is None


def test_drop_short_of_the_crashing_threshold_is_not_a_rebound():
    # Down 9% in a month misses the -10% crashing threshold; the forecast alone flags it rising.
    series = series_of("dreamcast", "vmu", [100.0, 100.0, 100.0, 91.0])
    assert flag(series, results_of(("dreamcast", "vmu", 50.0)))[("dreamcast", "vmu")]["rebound"] is None


def test_long_window_crash_alone_can_rebound():
    series = series_of("dreamcast", "vmu", [100.0, 80.0, 74.0, 74.0, 74.0, 74.0, 74.0])
    reasons = flag(series, results_of(("dreamcast", "vmu", 20.0)))[("dreamcast", "vmu")]
    assert reasons["rebound"]["signal"] == "change_6m"


def test_rebound_threshold_comes_from_settings():
    custom = dict(DEFAULTS, rebound={"forecast_pct": 50.0})
    series = series_of("dreamcast", "vmu", [100.0, 100.0, 100.0, 80.0])
    assert flag(series, results_of(("dreamcast", "vmu", 30.0)), custom)[("dreamcast", "vmu")]["rebound"] is None
