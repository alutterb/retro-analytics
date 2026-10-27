"""Flag games whose price is rising or crashing.

Two signals, in both directions: the forecast growth from the ranking, and the
realized price change over the last 1, 3 and 6 months. A game is rising when
any signal is at or above its rising threshold and crashing when any is at or
below its crashing threshold, so one game can be both (a recent drop with a
forecast recovery). Every triggered condition is kept as a reason.

A rebound is that case made explicit: a game that meets a crashing threshold
on a realized price change and whose forecast growth is at or above the
rebound threshold, so it recently crashed but the model expects it to climb.
"""

import operator

import pandas as pd

from retro_analytics import forecast

WINDOWS = (1, 3, 6)


def realized_changes(prices):
    """Percentage change from k months before the latest month, per window.

    `prices` is one game's monthly prices indexed by month. A window whose
    starting month is not in the series is None, not zero.
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    latest_month, latest = prices.index[-1], prices.iloc[-1]
    changes = {}
    for k in WINDOWS:
        start = latest_month - pd.DateOffset(months=k)
        # Rounded so an exact move such as 100 -> 140 meets a 40% threshold despite float error.
        changes[k] = round((latest / prices[start] - 1) * 100, 6) if start in prices.index else None
    return changes


def _text(value, months, horizon):
    """`months` is the realized-change window, or None for the forecast."""
    word = 'up' if value >= 0 else 'down'
    if months is None:
        return f"forecast {word} {abs(value):.0f}% in {horizon} months"
    return f"{word} {abs(value):.0f}% in {months} month{'s' if months != 1 else ''}"


def reasons(growth_pct, changes, thresholds, horizon=forecast.HORIZON):
    """Every rising or crashing condition one game meets."""
    signals = {'forecast': (growth_pct, None), **{f'change_{k}m': (changes[k], k) for k in WINDOWS}}
    found = []
    for direction, beyond in (('rising', operator.ge), ('crashing', operator.le)):
        for signal, (value, months) in signals.items():
            if value is None or pd.isna(value):
                continue
            threshold = thresholds[direction][f'{signal}_pct']
            if beyond(float(value), threshold):
                found.append({
                    'direction': direction, 'signal': signal, 'value_pct': float(value),
                    'threshold_pct': threshold, 'text': _text(float(value), months, horizon),
                })
    return found


def rebound(growth_pct, found, thresholds, horizon=forecast.HORIZON):
    """The rebound record for one game, or None.

    `found` is the game's reasons. The drop shown is the shortest realized
    window that crashed, the most recent one; `value_pct` is the forecast
    growth, which ranks rebounds.
    """
    threshold = thresholds['rebound']['forecast_pct']
    if growth_pct is None or pd.isna(growth_pct) or float(growth_pct) < threshold:
        return None
    drops = [r for r in found if r['direction'] == 'crashing' and r['signal'] != 'forecast']
    if not drops:
        return None
    order = [f'change_{k}m' for k in WINDOWS]
    drop = min(drops, key=lambda r: order.index(r['signal']))
    growth = float(growth_pct)
    return {
        'direction': 'rebound', 'signal': drop['signal'], 'value_pct': growth, 'threshold_pct': threshold,
        'text': f"{drop['text']}, {_text(growth, None, horizon)}",
    }


def flag_games(results, series, thresholds, horizon=forecast.HORIZON):
    """Flagged games in ranking order: console, game, changes, reasons and rebound.

    `results` is the ranking (console, game, growth_pct); `series` the long
    monthly table written by `ra prep`.
    """
    history = {key: rows.set_index('month')['price'] for key, rows in series.groupby(['console', 'game'])}
    flagged = []
    for r in results.itertuples(index=False):
        prices = history.get((r.console, r.game))
        if prices is None or prices.empty:
            continue
        changes = realized_changes(prices)
        found = reasons(r.growth_pct, changes, thresholds, horizon)
        if found:
            flagged.append({'console': r.console, 'game': r.game, 'changes': changes, 'reasons': found,
                            'rebound': rebound(r.growth_pct, found, thresholds, horizon)})
    return flagged
