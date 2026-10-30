"""Write the flagged games as one JSON file a phone can download.

The file holds only the games flagged rising or crashing, each with enough
history and forecast to draw a chart. A game that recently crashed while the
forecast points up also carries a ``rebound`` flag shaped like the entries in
``flags``; it is a separate field so a reader that predates it still sees only
rising and crashing flags. Its shape is versioned by ``schema_version``; the
Android app reads version 1, so later changes add fields or bump the version.

``underrated`` lists, per console, the top games from ``ra underrated`` with
this week's CIB price; it is empty until IGDB has been queried once.
"""

import json
import math
import os
from datetime import date

import pandas as pd

from retro_analytics import flags, forecast, paths, report, settings

SCHEMA_VERSION = 1
HISTORY_MONTHS = report.SPARK_MONTHS
UNDERRATED_TOP = 25  # games per console


def _pct(value):
    return None if value is None or pd.isna(value) else round(float(value), 1)


def _money(value):
    return round(float(value), 2)


def _text(value):
    return None if pd.isna(value) else str(value)


def underrated_rows(underrated, results, games=None):
    """Each console's top underrated games still tracked, with the latest CIB price where ranked."""
    if underrated is None:
        return []
    top = underrated[underrated['rank'] <= UNDERRATED_TOP]
    if games is not None:
        tracked = set(zip(games['console'], games['game']))
        top = top[[key in tracked for key in zip(top['console'], top['game'])]]
    prices = results.set_index(['console', 'game'])
    rows = []
    for u in top.itertuples(index=False):
        if not all(math.isfinite(float(v)) for v in (u.score, u.rating, u.votes, u.popularity)):
            continue  # one bad number would make the whole file unreadable
        key = (u.console, u.game)
        price = prices.loc[key] if key in prices.index else None
        rows.append({
            'console': u.console,
            'game': u.game,
            'title': str(u.title),
            'url': _text(u.url),
            'igdb_url': _text(u.igdb_url),
            'score': round(float(u.score), 2),
            'rank': int(u.rank),
            'rating': round(float(u.rating), 1),
            'votes': int(u.votes),
            'popularity': int(u.popularity),
            'last_price': None if price is None else _money(price['last_price']),
            'last_month': None if price is None else str(price['last_month']),
        })
    return rows


def build(results, series, forecasts, thresholds, games=None, generated=None, underrated=None):
    """The export document for the flagged games, in ranking order, and the underrated games."""
    urls = {}
    if games is not None and 'url' in games:
        urls = {(g.console, g.game): g.url for g in games.itertuples(index=False) if pd.notna(g.url)}
    history = {key: rows.sort_values('month') for key, rows in series.groupby(['console', 'game'])}
    future = {key: rows.sort_values('step') for key, rows in forecasts.groupby(['console', 'game'])}
    ranked = results.set_index(['console', 'game'])
    horizon = int(forecasts['step'].max()) if not forecasts.empty else forecast.HORIZON

    exported = []
    for flagged in flags.flag_games(results, series, thresholds, horizon):
        key = (flagged['console'], flagged['game'])
        r = ranked.loc[key]
        past = history[key].tail(HISTORY_MONTHS)
        ahead = future.get(key, forecasts.iloc[0:0])
        exported.append({
            'console': key[0],
            'game': key[1],
            'title': str(r['title']),
            'url': urls.get(key),
            'last_month': str(r['last_month']),
            'last_price': _money(r['last_price']),
            'changes_pct': {f'{k}m': _pct(v) for k, v in flagged['changes'].items()},
            'forecast': {
                'growth_pct': _pct(r['growth_pct']),
                'months': ahead['month'].tolist(),
                'prices': [_money(v) for v in ahead['forecast']],
            },
            'history': {
                'months': past['month'].tolist(),
                'prices': [_money(v) for v in past['price']],
            },
            'flags': [dict(reason, value_pct=_pct(reason['value_pct'])) for reason in flagged['reasons']],
            'rebound': flagged['rebound'] and dict(flagged['rebound'], value_pct=_pct(flagged['rebound']['value_pct'])),
        })
    return {
        'schema_version': SCHEMA_VERSION,
        'generated': generated or date.today().isoformat(),
        'horizon_months': horizon,
        'thresholds': thresholds,
        'games': exported,
        'underrated': underrated_rows(underrated, results, games),
    }


def write(document, path):
    """Write atomically, so a publish never reads a half-written file."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    results = pd.read_csv(paths.require(paths.results(), "rank"))
    series = pd.read_csv(paths.require(paths.series(), "prep"))
    forecasts = pd.read_csv(paths.require(paths.forecasts(), "forecast"))
    games = pd.read_csv(paths.games()) if paths.games().exists() else None
    underrated = pd.read_csv(paths.underrated()) if paths.underrated().exists() else None
    document = build(results, series, forecasts, settings.load().thresholds, games, underrated=underrated)
    write(document, paths.export())
    count = {direction: sum(any(f['direction'] == direction for f in g['flags']) for g in document['games'])
             for direction in ('rising', 'crashing')}
    rebounds = sum(g['rebound'] is not None for g in document['games'])
    print(f"Flagged {count['rising']} rising, {count['crashing']} crashing and {rebounds} rebound games, "
          f"and listed {len(document['underrated'])} underrated games, into {paths.export()}")
