"""Choose a forecasting model per game.

Prices are modelled on the log scale, so a trend means steady percentage
growth. Each game gets an ARIMA(p, 1, q) with p, q <= 2, picked by AIC, and
every model carries a drift term: a series that has been rising keeps rising in
the forecast instead of flattening out, which is what the old drift-free
models did. If no ARIMA fits, the game falls back to a random walk with drift:
the average monthly log change carried forward.

The chosen specification, not the fitted object, is saved; forecasting refits it.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from retro_analytics import paths, time_series

MIN_POINTS = 12
MAX_P = 2
MAX_Q = 2
D = 1

SPEC_COLUMNS = ['console', 'game', 'n_points', 'method', 'p', 'd', 'q', 'aic']


def fit_arima(log_prices, order):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)
        # With one difference, trend 't' is the drift (the mean monthly log change).
        return ARIMA(log_prices, order=order, trend='t').fit()


def select_model(prices):
    """The best ARIMA by AIC for one price series, or the drift fallback."""
    log_prices = np.log(np.asarray(prices, dtype=float))
    best, failures = None, []
    for p, q in itertools.product(range(MAX_P + 1), range(MAX_Q + 1)):
        try:
            fitted = fit_arima(log_prices, (p, D, q))
        except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as e:
            failures.append(f"({p},{D},{q}): {e}")
            continue
        if np.isfinite(fitted.aic) and (best is None or fitted.aic < best['aic']):
            best = {'method': 'arima', 'p': p, 'd': D, 'q': q, 'aic': fitted.aic}
    if best is None:
        print(f"  no ARIMA fit converged, using drift ({len(failures)} failures; first: {failures[0]})")
        return {'method': 'drift', 'p': None, 'd': None, 'q': None, 'aic': None}
    return best


def select_models(series, min_points=MIN_POINTS):
    """Model specs for every series long enough to model."""
    rows, skipped = [], 0
    for (console, game), prices in series.items():
        prices = prices.dropna()
        if len(prices) < min_points:
            print(f"Skipping {console}/{game}: {len(prices)} months of prices, need {min_points}")
            skipped += 1
            continue
        if (prices <= 0).any():
            print(f"Skipping {console}/{game}: non-positive prices")
            skipped += 1
            continue
        print(f"Selecting model for {console}/{game}")
        rows.append({'console': console, 'game': game, 'n_points': len(prices), **select_model(prices)})
    print(f"Selected {len(rows)} models, skipped {skipped} series")
    return pd.DataFrame(rows, columns=SPEC_COLUMNS)


def main():
    specs = select_models(time_series.load_series())
    specs.to_csv(paths.model_specs(), index=False)


if __name__ == "__main__":
    main()
