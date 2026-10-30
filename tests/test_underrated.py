import math
import re

import numpy as np
import pandas as pd
import pytest

from retro_analytics import igdb, paths, pipeline, underrated


def row(game, votes, rating, console='nes', parent_votes=0, game_type=0, igdb_id=None, title=None, per_day=1.0):
    return {'console': console, 'game': game, 'title': title or game, 'url': f'https://pc/{console}/{game}',
            'igdb_id': igdb_id if igdb_id is not None else hash((console, game)) % 10**6,
            'igdb_url': f'https://igdb/{game}', 'game_type': game_type, 'rating': rating, 'votes': votes,
            'parent_votes': parent_votes, 'per_day': per_day}


def scored(*rows):
    return underrated.score(pd.DataFrame(rows)).set_index('game')


def test_shrinks_toward_console_median_with_k_10():
    out = scored(row('a', 10, 90.0), row('b', 30, 70.0), row('c', 50, 80.0))
    # Median rating 80: a = (10*90 + 10*80) / 20.
    assert out.loc['a', 'shrunk'] == pytest.approx(85.0)
    assert out.loc['b', 'shrunk'] == pytest.approx((30 * 70 + 10 * 80) / 40)


def test_few_votes_are_fitted_but_not_listed():
    out = scored(row('a', 10, 90.0), row('b', 30, 70.0), row('c', 50, 80.0), row('thin', 9, 99.0))
    assert set(out.index) == {'a', 'b', 'c'}


def test_bundles_and_hardware_are_dropped():
    out = scored(row('a', 10, 90.0), row('b', 30, 70.0), row('c', 50, 80.0),
                 row('pack', 500, 95.0, game_type=3), row('collection', 500, 95.0, game_type=13),
                 row('lite', 40, 95.0, title='Black Nintendo DS Lite'))
    assert set(out.index) == {'a', 'b', 'c'}


def test_one_row_per_igdb_game_prefers_plain_then_busiest_listing():
    out = scored(row('plain', 20, 80.0, igdb_id=7, per_day=0.1),
                 row('hits', 20, 80.0, igdb_id=7, title='plain [Greatest Hits]', per_day=5.0),
                 row('busy', 20, 60.0, igdb_id=8, per_day=3.0), row('quiet', 20, 60.0, igdb_id=8, per_day=0.2),
                 row('other', 40, 70.0))
    assert set(out.index) == {'plain', 'busy', 'other'}


def test_popularity_counts_parent_votes():
    out = scored(row('port', 20, 80.0, parent_votes=980), row('b', 30, 70.0), row('c', 50, 80.0))
    assert out.loc['port', 'popularity'] == 1000


def test_score_is_standardized_residual_on_log_popularity():
    rng = np.random.default_rng(1)
    rows = [row(f'g{i}', int(v), float(r)) for i, (v, r) in
            enumerate(zip(rng.integers(10, 2000, 40), rng.uniform(50, 95, 40)))]
    out = scored(*rows)
    frame = out.assign(x=np.log1p(out['popularity']))
    slope, intercept = np.polyfit(frame['x'], frame['shrunk'], 1)
    residual = frame['shrunk'] - (intercept + slope * frame['x'])
    assert out['score'].to_numpy() == pytest.approx((residual / residual.std(ddof=1)).to_numpy())
    assert out['score'].std(ddof=1) == pytest.approx(1.0)
    assert list(out['rank']) == list(range(1, 41))
    assert out['score'].is_monotonic_decreasing


def test_small_console_uses_pooled_slope_and_keeps_its_short_list():
    big = [row(f'g{i}', 10 * (i + 1), 60.0 + i, console='nes') for i in range(25)]
    small = [row('x', 10, 70.0, console='virtual-boy'), row('y', 100, 70.0, console='virtual-boy'),
             row('z', 1000, 70.0, console='virtual-boy')]
    out = underrated.score(pd.DataFrame(big + small))
    vb = out[out['console'] == 'virtual-boy'].set_index('game')
    assert list(vb['rank']) == [1, 2, 3]
    # Equal ratings, but the pooled slope says popular games should rate higher: the least voted wins.
    assert list(vb.index) == ['x', 'y', 'z']


def test_console_with_one_rated_game_scores_zero_not_nan():
    out = scored(row('alone', 15, 88.0, console='virtual-boy'))
    assert out.loc['alone', 'score'] == 0.0 and not math.isnan(out.loc['alone', 'score'])


# --- The stage, against a fake IGDB -----------------------------------------------------------------

IGDB_GAMES = [
    {'id': 1, 'name': 'Legend of Zelda', 'platforms': [18], 'game_type': 0, 'rating': 85.0, 'rating_count': 900,
     'url': 'https://www.igdb.com/games/the-legend-of-zelda'},
    {'id': 2, 'name': 'Clash at Demonhead', 'platforms': [18], 'game_type': 0, 'rating': 88.7, 'rating_count': 13},
    {'id': 3, 'name': 'StarTropics', 'platforms': [18], 'game_type': 0, 'rating': 86.6, 'rating_count': 36},
    {'id': 4, 'name': 'Sonic Chaos', 'platforms': [35], 'game_type': 0, 'rating': 86.6, 'rating_count': 37},
    {'id': 397, 'name': 'Final Fantasy: 20th Anniversary Edition', 'platforms': [38], 'game_type': 9,
     'rating': 80.0, 'rating_count': 24, 'parent_game': 6},
    {'id': 6, 'name': 'Final Fantasy', 'platforms': [18], 'game_type': 0, 'rating': 79.0, 'rating_count': 700},
]


class FakeIgdb:
    """Answers the two query shapes the stage sends: games on platforms, and games by id."""

    def __init__(self, games=IGDB_GAMES):
        self.games = games
        self.queries = []

    def _ids(self, body, field):
        found = re.search(rf'where {field} = \(([^)]*)\)', body)
        return {int(i) for i in found.group(1).split(',')} if found else None

    def query(self, endpoint, body):
        self.queries.append(body)
        ids = self._ids(body, 'id')
        return [g for g in self.games if g['id'] in ids]

    def query_all(self, endpoint, body):
        self.queries.append(body)
        platforms = self._ids(body, 'platforms')
        return [g for g in self.games if platforms & set(g['platforms'])]


def track(*games):
    pd.DataFrame([{'console': c, 'game': g, 'title': t, 'url': f'https://www.pricecharting.com/game/{c}/{g}'}
                  for c, g, t in games]).to_csv(paths.games(), index=False)
    return pd.read_csv(paths.games())


GAMES = [('nes', 'legend-of-zelda', 'Legend of Zelda'), ('nes', 'clash-at-demonhead', 'Clash at Demonhead'),
         ('nes', 'star-tropics', 'Star Tropics'), ('sega-game-gear', 'sonic-chaos', 'Sonic Chaos'),
         ('psp', 'final-fantasy', 'Final Fantasy'), ('nes', 'nes-zapper', 'NES Zapper Light Gun')]


def test_update_matches_caches_and_scores():
    fake = FakeIgdb()
    out = underrated.update(fake, track(*GAMES)).set_index('game')
    assert set(out.index) == {'legend-of-zelda', 'clash-at-demonhead', 'star-tropics', 'sonic-chaos', 'final-fantasy'}
    assert out.loc['legend-of-zelda', 'igdb_url'] == 'https://www.igdb.com/games/the-legend-of-zelda'
    # The packaged override points PSP Final Fantasy at the anniversary edition; its parent's votes count.
    assert out.loc['final-fantasy', 'igdb_id'] == 397 and out.loc['final-fantasy', 'popularity'] == 724
    cache = pd.read_csv(paths.igdb_matches()).set_index('game')
    assert cache.loc['clash-at-demonhead', 'igdb_id'] == 2
    assert pd.isna(cache.loc['nes-zapper', 'igdb_id'])


def test_cached_games_and_misses_are_not_rematched():
    games = track(*GAMES)
    underrated.update(FakeIgdb(), games)
    fake = FakeIgdb()
    underrated.update(fake, games)
    assert not any('platforms' in q for q in fake.queries)


def test_an_id_igdb_dropped_is_rematched_next_run():
    games = track(*GAMES[:3])
    underrated.update(FakeIgdb(), games)
    without_clash = [g for g in IGDB_GAMES if g['id'] != 2]
    underrated.update(FakeIgdb(without_clash), games)
    assert 'clash-at-demonhead' not in set(pd.read_csv(paths.igdb_matches())['game'])
    fake = FakeIgdb()
    underrated.update(fake, games)
    assert sum('platforms' in q for q in fake.queries) == 1


def test_owner_override_beats_packaged_and_blank_removes():
    paths.igdb_overrides().write_text("console,game,igdb_id\npsp,final-fantasy,\nnes,star-tropics,6\n")
    out = underrated.update(FakeIgdb(), track(*GAMES)).set_index('game')
    assert 'final-fantasy' not in out.index
    assert out.loc['star-tropics', 'igdb_id'] == 6


def test_console_without_platform_is_skipped(capsys):
    out = underrated.update(FakeIgdb(), track(('nes', 'star-tropics', 'Star Tropics'), ('pc-games', 'doom', 'Doom')))
    assert list(out['game']) == ['star-tropics']
    assert "No IGDB platform known for pc-games" in capsys.readouterr().out


def test_without_credentials_the_stage_warns_and_keeps_the_last_list(monkeypatch, capsys):
    monkeypatch.delenv('TWITCH_CLIENT_ID', raising=False)
    monkeypatch.delenv('TWITCH_CLIENT_SECRET', raising=False)
    track(*GAMES)
    paths.underrated().write_text("last week\n")
    underrated.main()
    assert paths.underrated().read_text() == "last week\n"
    assert "warning: TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET are not set" in capsys.readouterr().out


def test_run_without_credentials_still_exports_and_publishes(monkeypatch):
    monkeypatch.delenv('TWITCH_CLIENT_ID', raising=False)
    ran = []
    for module, name in [(pipeline.pricecharting, 'scrape'), (pipeline.time_series, 'prep'),
                         (pipeline.training, 'train'), (pipeline.forecast, 'forecast'),
                         (pipeline.metrics, 'rank'), (pipeline.report, 'report'),
                         (pipeline.export, 'export'), (pipeline.publish, 'publish')]:
        monkeypatch.setattr(module, 'main', lambda *a, name=name, **k: ran.append(name))
    track(*GAMES)
    pipeline.run()
    assert ran[-2:] == ['export', 'publish']


def test_run_exports_and_publishes_before_reporting_an_igdb_failure(monkeypatch):
    ran = []
    for module, name in [(pipeline.pricecharting, 'scrape'), (pipeline.time_series, 'prep'),
                         (pipeline.training, 'train'), (pipeline.forecast, 'forecast'),
                         (pipeline.metrics, 'rank'), (pipeline.report, 'report'),
                         (pipeline.export, 'export'), (pipeline.publish, 'publish')]:
        monkeypatch.setattr(module, 'main', lambda *a, name=name, **k: ran.append(name))

    def broken():
        raise igdb.IgdbError("IGDB games query failed: HTTP 503")

    monkeypatch.setattr(pipeline.underrated, 'main', broken)
    with pytest.raises(igdb.IgdbError, match="503"):
        pipeline.run()
    assert ran[-2:] == ['export', 'publish']
