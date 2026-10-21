import pandas as pd

from retro_analytics import paths


def group_prices(prices_df):
    """Collapse one-row-per-scrape prices into one row of price and date lists per game."""
    prices_df = prices_df.drop_duplicates()  # Remove exact duplicate rows
    grouped_df = prices_df.groupby(['console', 'game'])[['price', 'date']].agg({'price': list, 'date': list}).reset_index()
    grouped_df = grouped_df.rename(columns={'price': 'historical prices', 'date': 'historical dates'})
    return grouped_df


def main():
    prices_df = pd.read_csv(paths.prices())
    group_prices(prices_df).to_csv(paths.historical_prices())


if __name__ == "__main__":
    main()
