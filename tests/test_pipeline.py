import pytest

from retro_analytics import cli, paths, pipeline, pricecharting


@pytest.fixture
def stages(monkeypatch):
    ran = []
    for module, name in [(pipeline.time_series, "prep"), (pipeline.training, "train"),
                         (pipeline.forecast, "forecast"), (pipeline.metrics, "rank")]:
        monkeypatch.setattr(module, "main", lambda name=name: ran.append(name))
    monkeypatch.setattr(pipeline, "sentiment", lambda: ran.append("sentiment"))
    return ran


@pytest.fixture
def scrape_ok(monkeypatch):
    monkeypatch.setattr(pipeline.pricecharting, "main", lambda snapshot: None)


def test_run_is_price_only_by_default(scrape_ok, stages):
    pipeline.run()
    assert stages == ["prep", "train", "forecast", "rank"]


def test_run_with_sentiment_collects_it_before_ranking(scrape_ok, stages):
    cli.main(["run", "--sentiment"])
    assert stages == ["sentiment", "prep", "train", "forecast", "rank"]


def test_run_ranks_partial_scrape_then_fails(monkeypatch, stages):
    def partial_scrape(snapshot):
        paths.prices().write_text("console,game,price,date\n")
        raise pricecharting.ScrapeError("1 of 2 games failed")

    monkeypatch.setattr(pipeline.pricecharting, "main", partial_scrape)
    with pytest.raises(pricecharting.ScrapeError, match="1 of 2"):
        pipeline.run()
    assert stages == ["prep", "train", "forecast", "rank"]


def test_run_stops_when_nothing_was_scraped(monkeypatch, stages):
    def no_seed(snapshot):
        raise pricecharting.ScrapeError("run `ra seed` first")

    monkeypatch.setattr(pipeline.pricecharting, "main", no_seed)
    with pytest.raises(pricecharting.ScrapeError, match="ra seed"):
        pipeline.run()
    assert stages == []


def test_cli_reports_missing_inputs_with_exit_code(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["rank"])
    assert exit_info.value.code == 1
    assert "run `ra prep` first" in capsys.readouterr().err
