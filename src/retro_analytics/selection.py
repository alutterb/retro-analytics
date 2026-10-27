"""Pick which games to track on a console: the ones that trade most often.

A console page lists games in popularity order but not how often they sell.
PriceCharting shows that on each game's own page ("1 sale per day" for CIB),
so seeding reads the first few candidate pages per console and keeps the
games with the most CIB sales per day. The figures are cached in
``volumes.json`` so a re-seed only asks for games it has not seen lately.
"""

import json
import re
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup

from retro_analytics import paths

# How long a cached sales figure is trusted; trade frequency moves slowly.
CACHE_DAYS = 90

DAYS_PER_UNIT = {'day': 1, 'week': 7, 'month': 30, 'year': 365}
VOLUME = re.compile(r'(\d[\d,]*)\s+sales?\s+per\s+(day|week|month|year)', re.IGNORECASE)


def parse_cib_volume(html):
    """CIB sales per day from a game page ('5 sales per week' -> 0.714), or None if it shows none."""
    cell = BeautifulSoup(html, 'html.parser').select_one('tr.sales_volume td[data-show-tab="completed-auctions-cib"]')
    match = VOLUME.search(cell.get_text(' ', strip=True)) if cell else None
    if match is None:
        return None
    return int(match.group(1).replace(',', '')) / DAYS_PER_UNIT[match.group(2).lower()]


def load_cache(today):
    """Cached {game url: sales per day} that are still fresh."""
    try:
        stored = json.loads(paths.volumes().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    oldest = (today - timedelta(days=CACHE_DAYS)).isoformat()
    return {
        url: entry for url, entry in stored.items()
        if isinstance(entry, dict) and str(entry.get('date', '')) >= oldest and 'per_day' in entry
    }


def save_cache(cache):
    paths.volumes().write_text(json.dumps(cache, indent=1, sort_keys=True), encoding='utf-8')


def rank_by_volume(rows, top, session, cache, today=None):
    """The `top` rows with the most CIB sales per day; page order breaks ties.

    `cache` ({url: {'per_day', 'date'}}) is read and updated in place. A game whose page
    cannot be read, or shows no CIB volume, counts as 0 and so ranks last.
    """
    today = today or date.today()
    volumes = []
    for row in rows:
        entry = cache.get(row['url'])
        if entry is None:
            print(f"Reading volume of {row['url']}")
            try:
                per_day = parse_cib_volume(session.get(row['url']))
            except requests.RequestException as e:
                print(f"FAILED {row['url']}: {e}")
                volumes.append(0.0)
                continue
            entry = cache[row['url']] = {'per_day': per_day, 'date': today.isoformat()}
        volumes.append(entry['per_day'] or 0.0)
    order = sorted(range(len(rows)), key=lambda i: (-volumes[i], i))
    return [rows[i] for i in order[:top]]
