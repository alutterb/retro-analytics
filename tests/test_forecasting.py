import pickle

import numpy as np

from retro_analytics import forecast, paths, training


def test_find_best_order_returns_valid_order(trending_series):
    order = training.find_best_order(trending_series, max_p=1, max_d=1, max_q=1)
    assert order is not None
    p, d, q = order
    assert 0 <= p <= 1 and 0 <= d <= 1 and 0 <= q <= 1
    assert order != (0, 0, 0)


def test_train_arima_skips_missing_series():
    assert training.train_arima((("nes", "zelda"), None)) == (("nes", "zelda"), None)


def test_train_forecast_round_trip(trending_series):
    data = {("nes", "zelda"): trending_series, ("nes", "metroid"): None}
    models = training.train_models(data, num_processes=1)
    assert list(models) == [("nes", "zelda")]

    training.save_models(models, paths.arima_models())
    forecast.save_predictions(forecast.forecast_all(forecast.load_models(paths.arima_models())), paths.predictions())
    with open(paths.predictions(), "rb") as f:
        predictions = pickle.load(f)

    points = np.asarray(predictions[("nes", "zelda")])
    assert points.shape == (5,)
    assert np.all(np.isfinite(points))
    # Forecasts should stay in the neighbourhood of the series it was fitted on.
    assert 40 < points.mean() < 90


def test_forecast_none_model_is_empty():
    assert forecast.forecast((("nes", "zelda"), None)) == []
