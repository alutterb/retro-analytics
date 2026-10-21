import pytest

from retro_analytics import cli


def test_cli_dispatches_to_stage(monkeypatch):
    called = []
    monkeypatch.setitem(cli.COMMANDS, "rank", (lambda args: called.append("rank"), "help", None))
    cli.main(["rank"])
    assert called == ["rank"]


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_passes_seed_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pricecharting, "seed", lambda **kwargs: seen.update(kwargs))
    cli.main(["seed", "--console", "nes", "--console", "sega-genesis", "--top", "5", "--min-price", "30"])
    assert seen == {"consoles": ["nes", "sega-genesis"], "top": 5, "min_price": 30.0, "games": None}


def test_cli_seed_defaults_to_tracked_consoles(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.pricecharting, "seed", lambda **kwargs: seen.update(kwargs))
    cli.main(["seed", "--game", "nes/legend-of-zelda"])
    assert seen["consoles"] == cli.pricecharting.CONSOLES
    assert len(seen["consoles"]) == 21
    assert seen["games"] == ["nes/legend-of-zelda"]
