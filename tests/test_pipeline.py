import pytest

from retro_analytics import paths, pipeline, pricecharting


@pytest.fixture
def stages(monkeypatch):
    ran = []
    for module, name in [(pipeline.time_series, "prep"), (pipeline.arima_prep, "arima"),
                         (pipeline.training, "train"), (pipeline.forecast, "forecast"),
                         (pipeline.metrics, "rank")]:
        monkeypatch.setattr(module, "main", lambda name=name: ran.append(name))
    return ran


def test_run_ranks_partial_scrape_then_fails(monkeypatch, stages):
    def partial_scrape(snapshot):
        paths.prices().write_text("console,game,price,date\n")
        raise pricecharting.ScrapeError("1 of 2 games failed")

    monkeypatch.setattr(pipeline.pricecharting, "main", partial_scrape)
    with pytest.raises(pricecharting.ScrapeError, match="1 of 2"):
        pipeline.run()
    assert stages == ["prep", "arima", "train", "forecast", "rank"]


def test_run_stops_when_nothing_was_scraped(monkeypatch, stages):
    def no_seed(snapshot):
        raise pricecharting.ScrapeError("run `ra seed` first")

    monkeypatch.setattr(pipeline.pricecharting, "main", no_seed)
    with pytest.raises(pricecharting.ScrapeError, match="ra seed"):
        pipeline.run()
    assert stages == []
