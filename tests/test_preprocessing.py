import pickle

import pandas as pd
import pytest

from retro_analytics import arima_prep, paths, time_series


def prices_frame():
    return pd.DataFrame({
        "console": ["nes", "nes", "nes", "nes", "snes", "snes"],
        "game": ["zelda", "zelda", "zelda", "metroid", "chrono", "chrono"],
        "price": [170.0, 170.0, 172.5, 40.0, 300.0, None],
        "date": ["2026-01-01", "2026-01-01", "2026-02-01", "2026-01-01", "2026-01-01", "2026-02-01"],
    })


def test_group_prices_dedupes_and_collects_lists():
    grouped = time_series.group_prices(prices_frame()).set_index(["console", "game"])
    assert grouped.loc[("nes", "zelda"), "historical prices"] == [170.0, 172.5]
    assert grouped.loc[("nes", "zelda"), "historical dates"] == ["2026-01-01", "2026-02-01"]
    assert grouped.loc[("nes", "metroid"), "historical prices"] == [40.0]


@pytest.mark.parametrize("text, expected", [
    ("[1.0, 2.5, 3.0]", [1.0, 2.5, 3.0]),
    ("[40.0]", [40.0]),
    ("[1.0, nan]", False),
])
def test_clean_timeseries(text, expected):
    assert arima_prep.clean_timeseries(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "[1.0 2.0]", "42.0"])
def test_clean_timeseries_rejects_malformed(text):
    with pytest.raises(ValueError):
        arima_prep.clean_timeseries(text)


def test_clean_timeseries_does_not_execute_code():
    with pytest.raises(ValueError):
        arima_prep.clean_timeseries("__import__('os').getcwd(), 1")


def test_prep_round_trip_through_files():
    prices_frame().to_csv(paths.prices(), index=False)
    time_series.main()
    arima_prep.main()
    with open(paths.arima_input(), "rb") as f:
        data = pickle.load(f)
    assert data[("nes", "zelda")] == [170.0, 172.5]
    assert data[("nes", "metroid")] == [40.0]
    # A gap in the series drops the game from modelling.
    assert data[("snes", "chrono")] is None
