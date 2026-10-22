"""Turn scraped price rows into one clean monthly series per game."""

import pandas as pd

from retro_analytics import paths

# Older prices say little about where a game is heading now.
HISTORY_MONTHS = 60

SERIES_COLUMNS = ['console', 'game', 'month', 'price']


def monthly_series(prices, history_months=HISTORY_MONTHS):
    """One price per calendar month per game, oldest first.

    The last price seen in a month wins, so a snapshot taken today overrides
    that month's chart value. Months missing between two known prices are
    filled by linear interpolation; nothing is invented before the first or
    after the last known month.
    """
    prices = prices.dropna(subset=['price']).copy()
    prices['date'] = pd.to_datetime(prices['date'])
    frames = []
    for (console, game), rows in prices.sort_values('date').groupby(['console', 'game'], sort=True):
        monthly = rows.set_index('date')['price'].resample('MS').last()
        monthly = monthly.interpolate(limit_area='inside').dropna().tail(history_months)
        frames.append(pd.DataFrame({
            'console': console,
            'game': game,
            'month': monthly.index.strftime('%Y-%m-%d'),
            'price': monthly.to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=SERIES_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_series():
    """Series as {(console, game): pd.Series of prices indexed by month}."""
    series = pd.read_csv(paths.require(paths.series(), "prep"), parse_dates=['month'])
    return {
        key: rows.set_index('month')['price'].asfreq('MS')
        for key, rows in series.groupby(['console', 'game'], sort=True)
    }


def main():
    prices = pd.read_csv(paths.require(paths.prices(), "scrape"))
    if paths.games().exists():
        tracked = pd.read_csv(paths.games(), usecols=['console', 'game'])
        prices = prices.merge(tracked.drop_duplicates(), on=['console', 'game'])
    series = monthly_series(prices)
    series.to_csv(paths.series(), index=False)
    print(f"Wrote {series.groupby(['console', 'game']).ngroups} monthly series to {paths.series()}")


if __name__ == "__main__":
    main()
