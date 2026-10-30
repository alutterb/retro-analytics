"""Match PriceCharting titles to IGDB games on the same console.

Titles are normalized (ASCII, lowercase, '&' to 'and', Roman numerals to
digits, a few synonyms, filler words dropped), then compared with every name,
alternative name, localized name and colon subtitle of the console's IGDB
games. An exact match scores 100; numbers must agree ("Silent Hill 2" never
matches "Silent Hill 4"); otherwise the score is a fuzzy ratio, less small
penalties for weaker kinds of match. A match is accepted at 90 or more, the
cutoff below which hand-checked matches were mostly wrong.
"""

import re
import unicodedata

from rapidfuzz import fuzz

ACCEPT = 90

# II-XVI only: a bare I, V or X is usually a letter ("Mega Man X" is not "Mega Man 10").
ROMAN = {'ii': '2', 'iii': '3', 'iv': '4', 'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'xi': '11',
         'xii': '12', 'xiii': '13', 'xiv': '14', 'xv': '15', 'xvi': '16'}
SYNONYMS = {'brothers': 'bros', 'versus': 'vs', 'n': 'and', 'disneys': 'disney', 'doctor': 'dr', 'mister': 'mr',
            'spiderman': 'spider man', 'warioware': 'wario ware', 'saint': 'st', 'one': '1', 'two': '2',
            'three': '3'}
FILLER = {'the', 'a', 'an', 'version', 'game'}

# IGDB game_type: mods, forks and updates only ever produce wrong matches.
SKIPPED_TYPES = {5, 12, 14}
# Lower is preferred on a tie: main games, then remakes and remasters, then ports, then bundles.
TYPE_PREFERENCE = {0: 0, 8: 1, 9: 1, 10: 1, 4: 2, 11: 2, 3: 3}
# A console's own number in a title is not a sequel number ("Mario Kart 64").
CONSOLE_NUMBERS = {'nintendo-64': {'64'}}

BRACKETS = re.compile(r'\s*\[[^\]]*\]')
SUBTITLE = re.compile(r'\s*:\s*|\s+-\s+')


def norm(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    text = text.replace('&', ' and ').replace("'", '')
    words = (ROMAN.get(word, SYNONYMS.get(word, word)) for word in re.sub(r'[^a-z0-9]+', ' ', text).split())
    return ' '.join(word for word in words if word not in FILLER)


def candidates(games):
    """(normalized name, game, is alternative, is subtitle) for every name of every matchable game."""
    out = []
    for game in games:
        if game.get('game_type') in SKIPPED_TYPES:
            continue
        names = [(game['name'], False)] + [
            (other['name'], True)
            for field in ('alternative_names', 'game_localizations')
            for other in game.get(field) or () if other.get('name')
        ]
        for name, alternative in names:
            if normalized := norm(name):
                out.append((normalized, game, alternative, False))
            for part in SUBTITLE.split(name)[1:]:
                if normalized := norm(part):
                    out.append((normalized, game, alternative, True))
    return out


def _preference(game):
    return TYPE_PREFERENCE.get(game.get('game_type'), 4), -(game.get('total_rating_count') or 0)


def _numbers(name, console):
    ignored = CONSOLE_NUMBERS.get(console, set())
    return sorted(word for word in name.split() if word.isdigit() and word not in ignored)


def _queries(title):
    """The title with its [bracket] text kept, then without it, then Zelda titles with 'legend of' put back."""
    queries = [norm(title.replace('[', ' ').replace(']', ' '))]
    base = norm(BRACKETS.sub('', title))
    if base not in queries:
        queries.append(base)
    queries += ['legend of ' + query for query in queries if query.startswith('zelda')]
    return queries


def _score(name, query, console):
    if name == query:
        return 100.0
    if name.replace(' ', '') == query.replace(' ', ''):
        return 99.5
    if _numbers(name, console) != _numbers(query, console):
        return None
    if len(query) >= 5 and (name.startswith(query + ' ') or name.endswith(' ' + query)):
        return 94.0  # a subtitle or a publisher prefix left out
    if abs(len(name) - len(query)) > 0.5 * max(len(name), len(query)) + 4:
        return None
    return max(fuzz.ratio(name, query), fuzz.token_sort_ratio(name, query))


def match(title, names, console):
    """(score, IGDB game) for the best match of `title` among `names` (from `candidates`), or None below 90."""
    best = None
    for penalty, query in enumerate(_queries(title)):
        for name, game, alternative, subtitle in names:
            score = _score(name, query, console)
            if score is None:
                continue
            score -= 4 * subtitle + 1 * alternative + penalty
            if best is None or score > best[0] + 1e-9 or (
                    abs(score - best[0]) < 1e-9 and _preference(game) < _preference(best[1])):
                best = (score, game)
    return best if best is not None and best[0] >= ACCEPT else None
