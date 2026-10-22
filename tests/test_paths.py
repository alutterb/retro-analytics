from retro_analytics import paths


def test_paths_follow_env_and_create_parents(data_dirs):
    prices = paths.prices()
    assert prices == data_dirs / "data" / "raw" / "prices.csv"
    assert prices.parent.is_dir()
    assert paths.model_specs() == data_dirs / "models" / "model_specs.csv"
    assert paths.model_specs().parent.is_dir()


def test_default_data_dir_is_project_data(monkeypatch):
    monkeypatch.delenv("RA_DATA_DIR")
    assert paths.data_dir() == paths.PROJECT_DIR / "data"
