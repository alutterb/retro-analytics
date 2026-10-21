import numpy as np
import pandas as pd
import pytest

from conftest import monthly
from retro_analytics import forecast, paths, time_series, training


def test_select_model_picks_small_arima_with_trend(growing_series):
    spec = training.select_model(growing_series)
    assert spec["method"] == "arima"
    assert 0 <= spec["p"] <= training.MAX_P and 0 <= spec["q"] <= training.MAX_Q
    assert spec["d"] == 1
    assert np.isfinite(spec["aic"])


def test_forecast_of_rising_series_keeps_rising(growing_series):
    # The old ARIMA without drift forecast a flat line for series like this.
    spec = training.select_model(growing_series)
    points = forecast.forecast_prices(growing_series, spec, horizon=6)
    assert points.shape == (6,)
    assert np.all(np.diff(points) > 0)
    assert points[-1] > growing_series.iloc[-1] * 1.05


def test_drift_fallback_extends_average_growth():
    prices = monthly([100, 110, 121, 133.1])
    points = forecast.forecast_prices(prices, {"method": "drift"}, horizon=2)
    assert points == pytest.approx([146.41, 161.051])


def test_flat_series_forecasts_flat():
    prices = monthly([50.0] * 24)
    points = forecast.forecast_prices(prices, {"method": "drift"}, horizon=3)
    assert points == pytest.approx([50.0] * 3)


def test_select_models_skips_short_and_non_positive_series(growing_series):
    series = {
        ("nes", "zelda"): growing_series,
        ("nes", "short"): monthly([10.0] * 5),
        ("nes", "free"): monthly([0.0] + [10.0] * 20),
    }
    specs = training.select_models(series)
    assert specs[["console", "game"]].values.tolist() == [["nes", "zelda"]]
    assert specs.loc[0, "n_points"] == 48
    assert list(specs.columns) == training.SPEC_COLUMNS


def test_select_model_falls_back_to_drift_when_no_fit_works(monkeypatch, growing_series):
    def broken_fit(log_prices, order):
        raise ValueError("singular")

    monkeypatch.setattr(training, "fit_arima", broken_fit)
    assert training.select_model(growing_series)["method"] == "drift"


def test_train_forecast_round_trip_writes_csv_not_pickles(growing_series):
    rows = pd.DataFrame({
        "console": "nes", "game": "zelda",
        "month": growing_series.index.strftime("%Y-%m-%d"), "price": growing_series.to_numpy(),
    })
    rows.to_csv(paths.series(), index=False)

    training.main()
    forecast.main()

    specs = pd.read_csv(paths.model_specs())
    forecasts = pd.read_csv(paths.forecasts())
    assert specs["game"].tolist() == ["zelda"]
    assert forecasts["step"].tolist() == list(range(1, forecast.HORIZON + 1))
    assert forecasts["month"].iloc[0] == "2026-01-01"  # the month after the series ends
    assert np.all(np.isfinite(forecasts["forecast"]))
    assert not list(paths.data_dir().rglob("*.pickle")) and not list(paths.models_dir().rglob("*.pickle"))
    assert set(time_series.load_series()) == {("nes", "zelda")}
