"""Rank games by forecast price growth.

A game's score is how much its CIB price is forecast to grow over the forecast
horizon, as a percentage of its latest price. Reddit sentiment, when it has
been collected, only breaks ties: games whose growth matches to the displayed
precision are ordered by sentiment, and a game with no Reddit data counts as
neutral rather than being dropped.

The ranking also marks rebound games: a crashing price change with a forecast
growth at or above the rebound threshold (see ``flags``).
"""

import pandas as pd

from retro_analytics import flags, paths, settings

# Growth figures closer than this (in percentage points) count as a tie.
TIE_PRECISION = 0.1

RESULT_COLUMNS = [
    'rank', 'console', 'game', 'title', 'last_month', 'last_price',
    'forecast_month', 'forecast_price', 'growth_pct', 'sentiment', 'sentiment_texts',
    'rebound',
]


def rank(series, forecasts, sentiment=None, titles=None, tie_precision=TIE_PRECISION, thresholds=None):
    """One row per forecast game, best first.

    `series` and `forecasts` are the long tables written by `ra prep` and
    `ra forecast`. `sentiment` (console, game, sentiment, n_texts) and
    `titles` (console, game, title) are optional; `thresholds` default to the
    built-in ones and decide the `rebound` column.
    """
    keys = ['console', 'game']
    last = (series.sort_values('month').groupby(keys, as_index=False).last()
            .rename(columns={'month': 'last_month', 'price': 'last_price'}))
    final = (forecasts.sort_values('step').groupby(keys, as_index=False).last()
             .rename(columns={'month': 'forecast_month', 'forecast': 'forecast_price'})
             .drop(columns='step'))
    ranked = final.merge(last, on=keys, how='inner')
    ranked['growth_pct'] = (ranked['forecast_price'] / ranked['last_price'] - 1) * 100

    if sentiment is not None and not sentiment.empty:
        ranked = ranked.merge(sentiment[keys + ['sentiment', 'n_texts']], on=keys, how='left')
    else:
        ranked['sentiment'], ranked['n_texts'] = float('nan'), 0
    ranked = ranked.rename(columns={'n_texts': 'sentiment_texts'})
    ranked['sentiment_texts'] = ranked['sentiment_texts'].fillna(0).astype(int)

    if titles is not None and not titles.empty:
        ranked = ranked.merge(titles[keys + ['title']].drop_duplicates(keys), on=keys, how='left')
    else:
        ranked['title'] = None
    ranked['title'] = ranked['title'].fillna(ranked['game'])

    ranked['_growth_bucket'] = (ranked['growth_pct'] / tie_precision).round()
    ranked['_sentiment'] = ranked['sentiment'].fillna(0.0)
    ranked = ranked.sort_values(['_growth_bucket', '_sentiment', 'growth_pct'], ascending=False)
    ranked['rank'] = range(1, len(ranked) + 1)
    rebounds = {(f['console'], f['game']) for f in flags.flag_games(
        ranked, series, thresholds or settings.DEFAULT_THRESHOLDS) if f['rebound']}
    ranked['rebound'] = [key in rebounds for key in zip(ranked['console'], ranked['game'])]
    return ranked[RESULT_COLUMNS].reset_index(drop=True)


def main():
    series = pd.read_csv(paths.require(paths.series(), "prep"))
    forecasts = pd.read_csv(paths.require(paths.forecasts(), "forecast"))
    sentiment = pd.read_csv(paths.sentiment()) if paths.sentiment().exists() else None
    titles = pd.read_csv(paths.games()) if paths.games().exists() else None
    results = rank(series, forecasts, sentiment, titles, thresholds=settings.load().thresholds)
    results.to_csv(paths.results(), index=False)
    print(f"Ranked {len(results)} games ({results['rebound'].sum()} rebound) into {paths.results()}")


if __name__ == "__main__":
    main()
