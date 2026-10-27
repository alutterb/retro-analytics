"""Owner settings: tracked consoles, rising/crashing/rebound thresholds and the publish target.

Settings live in ``settings.toml`` in the data directory (gitignored), so the
weekly timer picks up a change without editing the systemd unit. A missing
file means every default below and no publishing. Unknown tables or keys are
errors, so a typo cannot silently fall back to a default.
"""

import re
import tomllib
from dataclasses import dataclass, fields

from retro_analytics import PipelineError, paths

# Percentages. Rising flags at or above, crashing at or below. A rebound is a
# crashing price change with a forecast growth at or above rebound.forecast_pct.
DEFAULT_THRESHOLDS = {
    'rising': {'forecast_pct': 10.0, 'change_1m_pct': 15.0, 'change_3m_pct': 25.0, 'change_6m_pct': 40.0},
    'crashing': {'forecast_pct': -10.0, 'change_1m_pct': -10.0, 'change_3m_pct': -15.0, 'change_6m_pct': -25.0},
    'rebound': {'forecast_pct': 10.0},
}


# What `ra seed` keeps per console unless [selection] or the console says otherwise: the `top` games
# that trade most often (CIB sales per day) among the first `candidates` games priced at or above `min_price`.
DEFAULT_SELECTION = {'top': 25, 'min_price': 20.0, 'candidates': 50}

# The tracked consoles, as PriceCharting URL slugs, with per-console overrides of DEFAULT_SELECTION.
# The obscure consoles list fewer games, trade rarely and sell cheaper, so they keep fewer from a shorter list.
_OBSCURE = {'candidates': 30}
DEFAULT_CONSOLES = {
    'nes': {}, 'super-nintendo': {}, 'nintendo-64': {}, 'gamecube': {}, 'wii': {}, 'nintendo-switch': {},
    'gameboy': {}, 'gameboy-color': {}, 'gameboy-advance': {}, 'nintendo-ds': {},
    'sega-genesis': {}, 'sega-cd': {}, 'sega-dreamcast': {},
    'playstation': {}, 'playstation-2': {}, 'playstation-3': {}, 'playstation-4': {}, 'psp': {},
    'xbox': {}, 'xbox-360': {}, 'xbox-one': {},
    'sega-saturn': {**_OBSCURE, 'top': 15}, 'turbografx-16': {**_OBSCURE, 'top': 15},
    'neo-geo-aes': {**_OBSCURE, 'top': 15}, '3do': {**_OBSCURE, 'top': 10, 'min_price': 15},
    'jaguar': {**_OBSCURE, 'top': 10}, 'virtual-boy': {**_OBSCURE, 'top': 10},
    'atari-2600': {**_OBSCURE, 'top': 15, 'min_price': 10}, 'atari-7800': {**_OBSCURE, 'top': 10, 'min_price': 15},
    'atari-lynx': {**_OBSCURE, 'top': 10, 'min_price': 15},
    'sega-master-system': {**_OBSCURE, 'top': 15, 'min_price': 15},
    'sega-game-gear': {**_OBSCURE, 'top': 15, 'min_price': 15}, 'sega-32x': {**_OBSCURE, 'top': 10},
    'cd-i': {**_OBSCURE, 'top': 10, 'min_price': 10},
}

SLUG = re.compile(r'[a-z0-9]+(-[a-z0-9]+)*')


@dataclass(frozen=True)
class ConsoleSelection:
    top: int
    min_price: float
    candidates: int


@dataclass(frozen=True)
class Publish:
    repo: str
    branch: str = 'main'
    path: str = 'results/latest.json'


@dataclass(frozen=True)
class Settings:
    thresholds: dict
    publish: Publish | None
    selection: ConsoleSelection  # for a console that is not in `consoles`
    consoles: dict  # slug -> ConsoleSelection, in tracking order


def load():
    """Settings from ``settings.toml``, merged over the defaults."""
    path = paths.settings()
    try:
        raw = tomllib.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except tomllib.TOMLDecodeError as e:
        raise PipelineError(f"{path} is not valid TOML: {e}") from e

    def fail(message):
        raise PipelineError(f"{path}: {message}")

    unknown = set(raw) - set(DEFAULT_THRESHOLDS) - {'publish', 'selection', 'consoles'}
    if unknown:
        fail(f"unknown table(s) {', '.join(sorted(unknown))}; expected rising, crashing, rebound, selection, consoles, publish")

    thresholds = {}
    for direction, defaults in DEFAULT_THRESHOLDS.items():
        given = raw.get(direction, {})
        for key, value in given.items():
            if key not in defaults:
                fail(f"unknown key {direction}.{key}; expected one of {', '.join(defaults)}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                fail(f"{direction}.{key} must be a number, got {value!r}")
        thresholds[direction] = {key: float(given.get(key, default)) for key, default in defaults.items()}

    publish = None
    if 'publish' in raw:
        given = raw['publish']
        unknown = set(given) - {f.name for f in fields(Publish)}
        if unknown:
            fail(f"unknown key(s) publish.{', publish.'.join(sorted(unknown))}")
        if not isinstance(given.get('repo'), str) or not given['repo'].strip():
            fail("publish.repo is required (for example git@github.com:owner/name.git)")
        publish = Publish(**{key: str(value) for key, value in given.items()})
    selection, consoles = load_consoles(raw, fail)
    return Settings(thresholds=thresholds, publish=publish, selection=selection, consoles=consoles)


def load_selection(given, base, where, fail):
    """`base` with the selection keys in `given` applied, as (top, min_price, candidates)."""
    unknown = set(given) - set(DEFAULT_SELECTION)
    if unknown:
        fail(f"unknown key(s) {where}.{', '.join(sorted(unknown))}; expected one of {', '.join(DEFAULT_SELECTION)}")
    merged = {**base, **given}
    for key in ('top', 'candidates'):
        if isinstance(merged[key], bool) or not isinstance(merged[key], int) or merged[key] < 1:
            fail(f"{where}.{key} must be a whole number of at least 1, got {merged[key]!r}")
    if isinstance(merged['min_price'], bool) or not isinstance(merged['min_price'], (int, float)) \
            or merged['min_price'] < 0:
        fail(f"{where}.min_price must be a number of at least 0, got {merged['min_price']!r}")
    return merged


def load_consoles(raw, fail):
    """The [selection] defaults and the per-console selections, as (ConsoleSelection, {slug: ConsoleSelection})."""
    defaults = load_selection(raw.get('selection', {}), DEFAULT_SELECTION, 'selection', fail)
    fallback = ConsoleSelection(defaults['top'], float(defaults['min_price']), defaults['candidates'])
    consoles = raw.get('consoles', DEFAULT_CONSOLES)
    if not isinstance(consoles, dict) or not consoles:
        fail("[consoles] must list at least one console, for example nes = {}")
    selections = {}
    for slug, given in consoles.items():
        if not SLUG.fullmatch(slug):
            fail(f"console {slug!r} is not a PriceCharting slug such as sega-saturn")
        if not isinstance(given, dict):
            fail(f"consoles.{slug} must be a table such as {{ top = 10 }}, got {given!r}")
        merged = load_selection(given, defaults, f'consoles.{slug}', fail)
        if 'candidates' not in given:
            merged['candidates'] = max(merged['candidates'], merged['top'])
        if merged['candidates'] < merged['top']:
            fail(f"consoles.{slug}.candidates ({merged['candidates']}) must be at least top ({merged['top']})")
        selections[slug] = ConsoleSelection(merged['top'], float(merged['min_price']), merged['candidates'])
    return fallback, selections
