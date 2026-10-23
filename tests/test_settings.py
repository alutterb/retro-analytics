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
    assert loaded.publish is None


def test_one_override_keeps_the_other_defaults():
    write_settings("[crashing]\nchange_3m_pct = -30\n")
    loaded = settings.load()
    assert loaded.thresholds["crashing"]["change_3m_pct"] == -30.0
    assert loaded.thresholds["crashing"]["change_1m_pct"] == -10.0
    assert loaded.thresholds["rising"] == settings.DEFAULT_THRESHOLDS["rising"]


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
    assert loaded.publish == settings.Publish(
        repo="git@github.com:alutterb/retro-analytics-data.git", branch="main", path="results/latest.json")
