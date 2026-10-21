import pandas as pd
import pytest

from retro_analytics import PipelineError, paths, time_series


def prices_frame():
    return pd.DataFrame({
        "console": ["nes"] * 5 + ["snes"] * 2,
        "game": ["zelda"] * 5 + ["chrono"] * 2,
        "price": [100.0, 110.0, 130.0, 125.0, 140.0, 300.0, None],
        # Zelda: Feb is missing, March has a chart value and a later snapshot.
        "date": ["2026-01-01", "2026-03-01", "2026-04-01", "2026-03-20", "2026-04-15",
                 "2026-01-01", "2026-02-01"],
    })


def test_monthly_series_resamples_interpolates_and_keeps_latest_in_month():
    series = time_series.monthly_series(prices_frame())
    zelda = series[series["game"] == "zelda"]
    assert zelda["month"].tolist() == ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
    # Feb is interpolated between Jan (100) and the last March value (125); April keeps the 15th.
    assert zelda["price"].tolist() == [100.0, 112.5, 125.0, 140.0]


def test_monthly_series_drops_missing_prices_and_does_not_extrapolate():
    chrono = time_series.monthly_series(prices_frame()).query("game == 'chrono'")
    assert chrono["month"].tolist() == ["2026-01-01"]


def test_monthly_series_keeps_only_recent_history():
    long = pd.DataFrame({
        "console": "nes", "game": "zelda", "price": range(1, 101),
        "date": pd.date_range("2010-01-01", periods=100, freq="MS").strftime("%Y-%m-%d"),
    })
    series = time_series.monthly_series(long, history_months=24)
    assert len(series) == 24
    assert series["price"].iloc[-1] == 100


def test_prep_round_trip_through_files():
    prices_frame().to_csv(paths.prices(), index=False)
    time_series.main()
    loaded = time_series.load_series()
    assert set(loaded) == {("nes", "zelda"), ("snes", "chrono")}
    assert loaded[("nes", "zelda")].index.freqstr == "MS"
    assert loaded[("nes", "zelda")].tolist() == [100.0, 112.5, 125.0, 140.0]


def test_prep_without_prices_says_which_stage_to_run():
    with pytest.raises(PipelineError, match="run `ra scrape` first"):
        time_series.main()
