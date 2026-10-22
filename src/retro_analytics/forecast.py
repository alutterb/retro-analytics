"""Forecast each game's price a few months ahead with its chosen model."""

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from retro_analytics import paths, time_series, training

HORIZON = 6

FORECAST_COLUMNS = ['console', 'game', 'step', 'month', 'forecast']


def drift_forecast(log_prices, horizon):
    """Random walk with drift: the mean monthly log change, carried forward."""
    drift = np.diff(log_prices).mean() if len(log_prices) > 1 else 0.0
    return log_prices[-1] + drift * np.arange(1, horizon + 1)


def forecast_prices(prices, spec, horizon=HORIZON):
    """Forecast prices for the next `horizon` months, in dollars."""
    log_prices = np.log(np.asarray(prices, dtype=float))
    if spec['method'] == 'arima':
        order = (int(spec['p']), int(spec['d']), int(spec['q']))
        try:
            log_forecast = np.asarray(training.fit_arima(log_prices, order).forecast(steps=horizon))
        except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as e:
            print(f"  refit of ARIMA{order} failed ({e}), using drift")
            log_forecast = drift_forecast(log_prices, horizon)
    else:
        log_forecast = drift_forecast(log_prices, horizon)
    return np.exp(log_forecast)


def forecast_all(series, specs, horizon=HORIZON):
    rows = []
    for spec in specs.to_dict('records'):
        prices = series[(spec['console'], spec['game'])].dropna()
        months = pd.date_range(prices.index[-1], periods=horizon + 1, freq='MS')[1:]
        for step, (month, value) in enumerate(zip(months, forecast_prices(prices, spec, horizon)), start=1):
            rows.append((spec['console'], spec['game'], step, month.strftime('%Y-%m-%d'), value))
    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def main():
    specs = pd.read_csv(paths.require(paths.model_specs(), "train"))
    forecasts = forecast_all(time_series.load_series(), specs)
    forecasts.to_csv(paths.forecasts(), index=False)
    print(f"Wrote {len(specs)} forecasts to {paths.forecasts()}")


if __name__ == "__main__":
    main()
