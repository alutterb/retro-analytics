import pytest

from retro_analytics import cli, settings


def test_cli_dispatches_to_stage(monkeypatch):
    called = []
    monkeypatch.setitem(cli.COMMANDS, "rank", (lambda args: called.append("rank"), "help", None))
    cli.main(["rank"])
    assert called == ["rank"]


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_seed_flags_override_settings_for_the_chosen_consoles(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pricecharting, "seed", lambda **kwargs: seen.update(kwargs))
    cli.main(["seed", "--console", "nes", "--console", "sega-saturn", "--top", "60", "--min-price", "30"])
    assert seen["games"] is None
    assert list(seen["consoles"]) == ["nes", "sega-saturn"]
    # --top above the candidate pool widens the pool so there is something to rank.
    assert seen["consoles"]["nes"] == settings.ConsoleSelection(top=60, min_price=30.0, candidates=60)
    assert seen["consoles"]["sega-saturn"] == settings.ConsoleSelection(top=60, min_price=30.0, candidates=60)


def test_cli_seed_defaults_to_the_consoles_in_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pricecharting, "seed", lambda **kwargs: seen.update(kwargs))
    cli.main(["seed", "--game", "nes/legend-of-zelda"])
    assert seen["consoles"] == settings.load().consoles
    assert len(seen["consoles"]) == 34
    assert seen["games"] == ["nes/legend-of-zelda"]


def test_cli_seed_uses_the_default_selection_for_a_console_not_in_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pricecharting, "seed", lambda **kwargs: seen.update(kwargs))
    cli.main(["seed", "--console", "atari-5200"])
    assert seen["consoles"] == {"atari-5200": settings.ConsoleSelection(25, 20.0, 50)}
