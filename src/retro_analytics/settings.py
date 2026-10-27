"""Owner settings: rising/crashing/rebound thresholds and the optional publish target.

Settings live in ``settings.toml`` in the data directory (gitignored), so the
weekly timer picks up a change without editing the systemd unit. A missing
file means every default below and no publishing. Unknown tables or keys are
errors, so a typo cannot silently fall back to a default.
"""

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


@dataclass(frozen=True)
class Publish:
    repo: str
    branch: str = 'main'
    path: str = 'results/latest.json'


@dataclass(frozen=True)
class Settings:
    thresholds: dict
    publish: Publish | None


def load():
    """Settings from ``settings.toml``, merged over the defaults."""
    path = paths.settings()
    try:
        raw = tomllib.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except tomllib.TOMLDecodeError as e:
        raise PipelineError(f"{path} is not valid TOML: {e}") from e

    def fail(message):
        raise PipelineError(f"{path}: {message}")

    unknown = set(raw) - set(DEFAULT_THRESHOLDS) - {'publish'}
    if unknown:
        fail(f"unknown table(s) {', '.join(sorted(unknown))}; expected rising, crashing, rebound, publish")

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
    return Settings(thresholds=thresholds, publish=publish)
