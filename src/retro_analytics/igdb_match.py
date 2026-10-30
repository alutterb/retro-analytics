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
from typing import NamedTuple

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


class Name(NamedTuple):
    """One candidate name, with what every comparison needs worked out once."""
    text: str
    game: dict
    penalty: int  # 1 for an alternative name, 4 for a subtitle
    spaceless: str
    numbers: list
    sorted_words: str


def _numbers(name, console):
    ignored = CONSOLE_NUMBERS.get(console, set())
    return sorted(word for word in name.split() if word.isdigit() and word not in ignored)


def _name(text, game, penalty, console):
    return Name(text, game, penalty, text.replace(' ', ''), _numbers(text, console), ' '.join(sorted(text.split())))


def candidates(games, console):
    """Every name, alternative name, localized name and colon subtitle of the matchable games."""
    out = []
    for game in games:
        if game.get('game_type') in SKIPPED_TYPES:
            continue
        names = [(game['name'], 0)] + [
            (other['name'], 1)
            for field in ('alternative_names', 'game_localizations')
            for other in game.get(field) or () if other.get('name')
        ]
        for name, alternative in names:
            if normalized := norm(name):
                out.append(_name(normalized, game, alternative, console))
            for part in SUBTITLE.split(name)[1:]:
                if normalized := norm(part):
                    out.append(_name(normalized, game, alternative + 4, console))
    return out


def _preference(game):
    return TYPE_PREFERENCE.get(game.get('game_type'), 4), -(game.get('total_rating_count') or 0)


def _queries(title):
    """The title with its [bracket] text kept, then without it, then Zelda titles with 'legend of' put back."""
    queries = [norm(title.replace('[', ' ').replace(']', ' '))]
    base = norm(BRACKETS.sub('', title))
    if base not in queries:
        queries.append(base)
    queries += ['legend of ' + query for query in queries if query.startswith('zelda')]
    return queries


def _score(name, query, cutoff):
    """The match score of candidate `name` for `query` (a Name), or None when it cannot reach `cutoff`."""
    if name.text == query.text:
        return 100.0
    if name.spaceless == query.spaceless:
        return 99.5
    if name.numbers != query.numbers:
        return None
    if len(query.text) >= 5 and (name.text.startswith(query.text + ' ') or name.text.endswith(' ' + query.text)):
        return 94.0  # a subtitle or a publisher prefix left out
    if abs(len(name.text) - len(query.text)) > 0.5 * max(len(name.text), len(query.text)) + 4:
        return None
    # ratio on sorted words is token_sort_ratio; the cutoff lets rapidfuzz stop early.
    score = max(fuzz.ratio(name.text, query.text, score_cutoff=cutoff),
                fuzz.ratio(name.sorted_words, query.sorted_words, score_cutoff=cutoff))
    return score if score >= cutoff else None


def match(title, names, console):
    """(score, IGDB game) for the best match of `title` among `names` (from `candidates`), or None below 90."""
    best = None
    for penalty, text in enumerate(_queries(title)):
        query = _name(text, None, penalty, console)
        for name in names:
            # Anything that cannot score 90 after its penalties can never be the accepted match.
            score = _score(name, query, ACCEPT + name.penalty + penalty - 1e-9)
            if score is None:
                continue
            score -= name.penalty + penalty
            if best is None or score > best[0] + 1e-9 or (
                    abs(score - best[0]) < 1e-9 and _preference(name.game) < _preference(best[1])):
                best = (score, name.game)
    return best if best is not None and best[0] >= ACCEPT else None
