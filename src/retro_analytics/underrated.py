"""Find tracked games that IGDB players rate higher than their popularity predicts.

Each tracked game is matched once to an IGDB game on its console and the match
is cached in ``igdb_matches.csv``. Every run then refreshes the IGDB user
rating and vote count of the matched games and scores them within each
console:

- quality is the user rating shrunk toward the console's median rating,
  ``(votes * rating + K * median) / (votes + K)``, so a handful of votes
  cannot carry a game;
- popularity is the game's votes plus those of its parent entry, since ports
  and editions are often rated on the original;
- the score is how far a game's quality sits above the line through its
  console's games (quality against log popularity), in standard deviations.

Games need at least 10 votes to be listed. Bundles, compilations and
hardware are left out, and each IGDB game appears once per console. Results go
to ``underrated.csv``; without IGDB credentials the stage is skipped and the
previous file, if any, is kept.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from retro_analytics import igdb, igdb_match, paths

K = 10  # votes' worth of weight on the console median
MIN_VOTES = 10
POOLED_BELOW = 20  # rated games a console needs for its own slope
BUNDLE_TYPES = (3, 13)  # IGDB game_type: bundles and packs
PACKAGED_OVERRIDES = Path(__file__).with_name('igdb_overrides.csv')

# PriceCharting console slug -> IGDB platform ids, siblings included where PriceCharting merges them.
PLATFORMS = {
    'nes': (18,), 'super-nintendo': (19,), 'nintendo-64': (4,), 'gamecube': (21,), 'wii': (5,),
    'nintendo-switch': (130, 508), 'gameboy': (33,), 'gameboy-color': (22,), 'gameboy-advance': (24,),
    'nintendo-ds': (20, 159), 'sega-genesis': (29,), 'sega-cd': (78, 482), 'sega-dreamcast': (23,),
    'playstation': (7,), 'playstation-2': (8,), 'playstation-3': (9,), 'playstation-4': (48,), 'psp': (38,),
    'xbox': (11,), 'xbox-360': (12,), 'xbox-one': (49,), 'sega-saturn': (32,), 'turbografx-16': (86, 150),
    'neo-geo-aes': (80, 79, 136), '3do': (50,), 'jaguar': (62, 410), 'virtual-boy': (87,),
    'atari-2600': (59,), 'atari-7800': (60,), 'atari-lynx': (61,), 'sega-master-system': (64,),
    'sega-game-gear': (35,), 'sega-32x': (30, 482), 'cd-i': (117,),
}

# Hardware and accessories that `ra seed` lets through (console pages list them among the games).
HARDWARE = re.compile(
    r"Panasonic 3DO|CX78 Joypad|7800 GameDrive|Travel (?:Pouch|Case)|Battery Pack|^Atari Lynx \[|Game Boy Printer|"
    r"Game ?Boy Pocket|Gameboy Light|Analogue|Gameboy Advance SP|^Game Boy Color|Gameboy Color$|"
    r"Gameboy Player|Atari Jaguar Composite|JagLink|Neo SD|Nintendo DS Lite|Nintendo DSi XL|Switch OLED|"
    r"^PSP (?:2000|3010 \[|Go)|Sega 32x Genesis|32X Unit|Backup RAM Cart|TV Tuner|Game Gear Handheld|"
    r"^Sega Game Gear \[|Super Wide Gear|Genesis Mini|3D Glasses|SNES Super Set|Super Gameboy|"
    r"Turbo (?:Express|Tap|Stick|Booster)|TurboGrafx 16 Mini|HyperFlash32|Hyperboy|Wii Remote|Action Replay|"
    r"^Xbox One [XS]|CDi Mouse|Gameboy Camera",
    re.IGNORECASE,
)

MATCH_FIELDS = 'name,alternative_names.name,game_localizations.name,game_type,total_rating_count'
RATING_FIELDS = 'url,rating,rating_count,game_type,parent_game,version_parent'
MATCH_COLUMNS = ['console', 'game', 'igdb_id', 'match_score']
COLUMNS = ['console', 'game', 'title', 'url', 'igdb_id', 'igdb_url', 'rating', 'votes', 'popularity',
           'shrunk', 'score', 'rank']


# --- Scoring ---------------------------------------------------------------------------------------

def _slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 1 and x.var() > 0 else 0.0


def standardized_residuals(frame, y, x):
    """Within-console residual of `y` on `x`, in console standard deviations.

    A console with fewer than POOLED_BELOW games uses the slope pooled across
    consoles; one with two or fewer games, or no spread, scores 0.
    """
    centered = lambda column: frame[column] - frame.groupby('console')[column].transform('mean')
    pooled = _slope(centered(x), centered(y))
    out = pd.Series(0.0, index=frame.index)
    for _, group in frame.groupby('console'):
        slope = _slope(group[x], group[y]) if len(group) >= POOLED_BELOW else pooled
        residual = group[y] - (group[y].mean() + slope * (group[x] - group[x].mean()))
        spread = residual.std(ddof=1) if len(group) > 2 else 0.0
        if spread > 0:
            out[group.index] = residual / spread
    return out


def score(games):
    """Ranked eligible games per console from one row per tracked game with its IGDB ratings.

    `games` has console, game, title, url, igdb_id, igdb_url, game_type, rating,
    votes, parent_votes and per_day (CIB sales per day, for picking between
    listings of the same IGDB game).
    """
    games = games[~games['title'].str.contains(HARDWARE) & ~games['game_type'].isin(BUNDLE_TYPES)].copy()
    # One row per console and IGDB game: the plain listing over "[Greatest Hits]" and the like, then the busiest.
    games['variant'] = games['title'].str.contains('[', regex=False)
    games = games.sort_values(['variant', 'per_day'], ascending=[True, False], na_position='last')
    games = games.drop_duplicates(['console', 'igdb_id'])
    rated = games[(games['votes'] > 0) & games['rating'].notna()].copy()
    median = rated.groupby('console')['rating'].transform('median')
    rated['shrunk'] = (rated['votes'] * rated['rating'] + K * median) / (rated['votes'] + K)
    rated['popularity'] = rated['votes'] + rated['parent_votes']
    rated['log_popularity'] = np.log1p(rated['popularity'])
    rated['score'] = standardized_residuals(rated, 'shrunk', 'log_popularity')
    eligible = rated[rated['votes'] >= MIN_VOTES].sort_values(['console', 'score'], ascending=[True, False])
    eligible['rank'] = eligible.groupby('console').cumcount() + 1
    return eligible[COLUMNS].reset_index(drop=True)


# --- Matching and its cache ------------------------------------------------------------------------

def load_matches():
    path = paths.igdb_matches()
    if not path.exists():
        return pd.DataFrame(columns=MATCH_COLUMNS)
    return pd.read_csv(path, dtype={'igdb_id': 'Int64'})


def load_overrides():
    """{(console, game): IGDB id or None}: the packaged corrections, then the owner's file over them."""
    overrides = {}
    for path in (PACKAGED_OVERRIDES, paths.igdb_overrides()):
        if path.exists():
            for row in pd.read_csv(path, comment='#', dtype={'igdb_id': 'Int64'}).itertuples(index=False):
                overrides[(row.console, row.game)] = None if pd.isna(row.igdb_id) else int(row.igdb_id)
    return overrides


def match_new(client, games, matches):
    """Cache rows for the tracked games on mapped consoles that have none yet; one platform pull per console."""
    known = set(zip(matches['console'], matches['game']))
    new = games[[(c, g) not in known for c, g in zip(games['console'], games['game'])]]
    rows = []
    for console, todo in new.groupby('console', sort=False):
        platforms = ', '.join(map(str, PLATFORMS[console]))
        pulled = client.query_all('games', f'fields {MATCH_FIELDS}; where platforms = ({platforms}); sort id asc;')
        names = igdb_match.candidates(pulled)
        found = 0
        for game in todo.itertuples(index=False):
            hit = igdb_match.match(game.title, names, console)
            found += hit is not None
            rows.append({'console': console, 'game': game.game, 'igdb_id': hit and hit[1]['id'],
                         'match_score': hit and round(hit[0], 1)})
        print(f"Matched {found} of {len(todo)} new {console} games to IGDB")
    added = pd.DataFrame(rows, columns=MATCH_COLUMNS).astype({'igdb_id': 'Int64'})
    return pd.concat([matches, added], ignore_index=True) if rows else matches


def fetch(client, ids, fields):
    """{id: IGDB game} for `ids`, 500 to a request."""
    ids = sorted(set(ids))
    found = {}
    for start in range(0, len(ids), igdb.PAGE):
        chunk = ', '.join(map(str, ids[start:start + igdb.PAGE]))
        for game in client.query('games', f'fields {fields}; where id = ({chunk}); limit {igdb.PAGE};'):
            found[game['id']] = game
    return found


def per_day():
    """{PriceCharting url: CIB sales per day} from the seed's volume cache."""
    if not paths.volumes().exists():
        return {}
    stored = json.loads(paths.volumes().read_text(encoding='utf-8'))
    return {url: entry.get('per_day') for url, entry in stored.items() if isinstance(entry, dict)}


def update(client, games):
    """Match new games, refresh ratings and return the scored list."""
    mapped = games['console'].isin(PLATFORMS)
    for console in sorted(set(games.loc[~mapped, 'console'])):
        print(f"No IGDB platform known for {console}; its games are not scored")
    games = games[mapped]
    tracked = pd.MultiIndex.from_frame(games[['console', 'game']])
    matches = load_matches()
    matches = matches[pd.MultiIndex.from_frame(matches[['console', 'game']]).isin(tracked)]
    matches = match_new(client, games, matches)

    overrides = load_overrides()
    ids = {(m.console, m.game): None if pd.isna(m.igdb_id) else int(m.igdb_id)
           for m in matches.itertuples(index=False)}
    ids.update((key, value) for key, value in overrides.items() if key in ids)
    rated = fetch(client, [i for i in ids.values() if i is not None], RATING_FIELDS)
    # A cached id IGDB no longer returns (merged or removed) is matched afresh next run.
    overridden = pd.Series([key in overrides for key in zip(matches['console'], matches['game'])], index=matches.index)
    gone = matches['igdb_id'].notna() & ~matches['igdb_id'].isin(list(rated)) & ~overridden
    matches[~gone].to_csv(paths.igdb_matches(), index=False)

    parent_of = {i: g.get('parent_game') or g.get('version_parent') for i, g in rated.items()}
    parents = fetch(client, [p for p in parent_of.values() if p], 'rating_count')
    volume = per_day()
    rows = []
    for game in games.itertuples(index=False):
        entry = rated.get(ids.get((game.console, game.game)))
        if entry is None:
            continue
        parent = parents.get(parent_of[entry['id']]) or {}
        rows.append({
            'console': game.console, 'game': game.game, 'title': game.title, 'url': game.url,
            'igdb_id': entry['id'], 'igdb_url': entry.get('url'), 'game_type': entry.get('game_type'),
            'rating': entry.get('rating'), 'votes': entry.get('rating_count') or 0,
            'parent_votes': parent.get('rating_count') or 0, 'per_day': volume.get(game.url),
        })
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return score(pd.DataFrame(rows))


def main():
    games = pd.read_csv(paths.require(paths.games(), "seed"))
    credentials = igdb.credentials()
    if credentials is None:
        kept = "keeping the previous list" if paths.underrated().exists() else "no list yet"
        print(f"warning: TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET are not set; skipping underrated games ({kept})")
        return
    scored = update(igdb.Client(*credentials), games)
    scored.to_csv(paths.underrated(), index=False)
    print(f"Scored {len(scored)} underrated candidates on {scored['console'].nunique()} consoles "
          f"into {paths.underrated()}")
