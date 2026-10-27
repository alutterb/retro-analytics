from pathlib import Path

import pytest

from retro_analytics import PipelineError, paths, settings

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "settings.toml"


def write_settings(text):
    paths.settings().write_text(text, encoding="utf-8")


def test_no_settings_file_means_defaults_and_no_publishing():
    loaded = settings.load()
    assert loaded.thresholds == settings.DEFAULT_THRESHOLDS
    assert loaded.thresholds["rising"] == {
        "forecast_pct": 10.0, "change_1m_pct": 15.0, "change_3m_pct": 25.0, "change_6m_pct": 40.0}
    assert loaded.thresholds["crashing"] == {
        "forecast_pct": -10.0, "change_1m_pct": -10.0, "change_3m_pct": -15.0, "change_6m_pct": -25.0}
    assert loaded.thresholds["rebound"] == {"forecast_pct": 10.0}
    assert loaded.publish is None


def test_one_override_keeps_the_other_defaults():
    write_settings("[crashing]\nchange_3m_pct = -30\n")
    loaded = settings.load()
    assert loaded.thresholds["crashing"]["change_3m_pct"] == -30.0
    assert loaded.thresholds["crashing"]["change_1m_pct"] == -10.0
    assert loaded.thresholds["rising"] == settings.DEFAULT_THRESHOLDS["rising"]


def test_rebound_threshold_can_be_overridden_and_older_files_keep_working():
    write_settings("[rebound]\nforecast_pct = 25\n")
    assert settings.load().thresholds["rebound"] == {"forecast_pct": 25.0}
    write_settings("[crashing]\nchange_1m_pct = -20\n")  # no [rebound] table at all
    assert settings.load().thresholds["rebound"] == {"forecast_pct": 10.0}


def test_unknown_rebound_key_is_an_error():
    write_settings("[rebound]\nchange_1m_pct = 25\n")
    with pytest.raises(PipelineError, match="rebound.change_1m_pct"):
        settings.load()


def test_unknown_key_is_an_error_naming_key_and_file():
    write_settings("[crashing]\nchnage_1m_pct = -5\n")
    with pytest.raises(PipelineError, match=r"chnage_1m_pct.*settings\.toml|settings\.toml.*chnage_1m_pct"):
        settings.load()


def test_unknown_table_is_an_error():
    write_settings("[falling]\nforecast_pct = -5\n")
    with pytest.raises(PipelineError, match="falling"):
        settings.load()


def test_non_numeric_threshold_is_an_error():
    write_settings('[rising]\nforecast_pct = "ten"\n')
    with pytest.raises(PipelineError, match="forecast_pct"):
        settings.load()


def test_invalid_toml_is_an_error_naming_the_file():
    write_settings("[rising\n")
    with pytest.raises(PipelineError, match="settings.toml"):
        settings.load()


def test_publish_defaults_branch_and_path():
    write_settings('[publish]\nrepo = "git@github.com:alutterb/retro-analytics-data.git"\n')
    publish = settings.load().publish
    assert publish == settings.Publish(
        repo="git@github.com:alutterb/retro-analytics-data.git", branch="main", path="results/latest.json")


def test_publish_without_repo_is_an_error():
    write_settings('[publish]\nbranch = "main"\n')
    with pytest.raises(PipelineError, match="repo"):
        settings.load()


def test_example_file_is_the_defaults_plus_the_documented_target():
    write_settings(EXAMPLE.read_text(encoding="utf-8"))
    loaded = settings.load()
    assert loaded.thresholds == settings.DEFAULT_THRESHOLDS
    assert loaded.selection == settings.ConsoleSelection(top=25, min_price=20.0, candidates=50)
    assert loaded.consoles == settings.load_consoles({}, pytest.fail)[1]
    assert loaded.publish == settings.Publish(
        repo="git@github.com:alutterb/retro-analytics-data.git", branch="main", path="results/latest.json")


def test_default_consoles_are_the_21_old_ones_plus_13_obscure_ones_with_their_own_selections():
    loaded = settings.load()
    assert len(loaded.consoles) == 34
    assert loaded.selection == settings.ConsoleSelection(top=25, min_price=20.0, candidates=50)
    assert loaded.consoles["nes"] == loaded.selection
    assert loaded.consoles["sega-saturn"] == settings.ConsoleSelection(top=15, min_price=20.0, candidates=30)
    assert loaded.consoles["atari-2600"] == settings.ConsoleSelection(top=15, min_price=10.0, candidates=30)


def test_listing_consoles_replaces_the_default_list():
    write_settings("[consoles]\nnes = {}\nsega-saturn = { top = 5, min_price = 40 }\n")
    loaded = settings.load()
    assert list(loaded.consoles) == ["nes", "sega-saturn"]
    assert loaded.consoles["sega-saturn"] == settings.ConsoleSelection(top=5, min_price=40.0, candidates=50)


def test_selection_table_sets_the_default_for_listed_consoles():
    write_settings("[selection]\ntop = 10\nmin_price = 5\ncandidates = 20\n\n[consoles]\nnes = {}\nsnes = { top = 30 }\n")
    consoles = settings.load().consoles
    assert consoles["nes"] == settings.ConsoleSelection(top=10, min_price=5.0, candidates=20)
    # Raising `top` past the candidate pool widens the pool so there is something to rank.
    assert consoles["snes"] == settings.ConsoleSelection(top=30, min_price=5.0, candidates=30)


@pytest.mark.parametrize("text, message", [
    ("[selection]\ntopp = 5\n", "selection.topp"),
    ("[selection]\ntop = 0\n", "selection.top"),
    ("[selection]\ntop = 2.5\n", "selection.top"),
    ('[selection]\nmin_price = "20"\n', "selection.min_price"),
    ("[selection]\nmin_price = -1\n", "selection.min_price"),
    ("[consoles]\n", "at least one console"),
    ("[consoles]\nSega_Saturn = {}\n", "Sega_Saturn"),
    ("[consoles]\nnes = 5\n", "consoles.nes"),
    ("[consoles]\nnes = { tpo = 5 }\n", "consoles.nes.tpo"),
    ("[consoles]\nnes = { top = 40, candidates = 20 }\n", "candidates"),
])
def test_bad_selection_settings_are_errors(text, message):
    write_settings(text)
    with pytest.raises(PipelineError, match=message):
        settings.load()


def test_existing_settings_without_the_new_keys_keep_working():
    write_settings('[rising]\nforecast_pct = 12\n\n[publish]\nrepo = "git@github.com:a/b.git"\n')
    loaded = settings.load()
    assert loaded.thresholds["rising"]["forecast_pct"] == 12.0
    assert len(loaded.consoles) == 34


def test_non_numeric_console_top_is_a_settings_error():
    write_settings('[consoles]\nnes = { top = "5" }\n')
    with pytest.raises(PipelineError, match="consoles.nes.top"):
        settings.load()
