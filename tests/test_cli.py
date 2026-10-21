import pytest

from retro_analytics import cli


def test_cli_dispatches_to_stage(monkeypatch):
    called = []
    monkeypatch.setitem(cli.COMMANDS, "rank", (lambda: called.append("rank"), "help"))
    cli.main(["rank"])
    assert called == ["rank"]


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])
