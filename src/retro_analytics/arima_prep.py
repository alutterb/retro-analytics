import ast
import pickle

import pandas as pd

from retro_analytics import paths


def clean_timeseries(timeseries):
    if 'nan' in timeseries:
        return False
    timeseries = timeseries.strip()
    if len(timeseries) == 0:
        raise ValueError("Input timeseries string is empty")

    try:
        values = ast.literal_eval(timeseries)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Input timeseries string is not properly formatted: {timeseries!r}") from e
    if not isinstance(values, list):
        raise ValueError(f"Input timeseries string is not a list: {timeseries!r}")
    return values


def build_series(hist_prices_df):
    """Map (console, game) to its list of prices, or None when the series has gaps."""
    raw_values = hist_prices_df['historical prices'].apply(clean_timeseries)
    names = list(zip(hist_prices_df['console'], hist_prices_df['game']))

    data = {}
    for name, timeseries in zip(names, raw_values):
        if timeseries:
            # In the case of ARIMA, we don't need to make series stationary or scale the data.
            # We'll store the original timeseries directly.
            data[name] = timeseries
        else:
            data[name] = None
    return data


def main():
    hist_prices_df = pd.read_csv(paths.historical_prices())
    data = build_series(hist_prices_df)

    with open(paths.arima_input(), 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
