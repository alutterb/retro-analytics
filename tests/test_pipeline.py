import pytest

from retro_analytics import cli, paths, pipeline, pricecharting


@pytest.fixture
def stages(monkeypatch):
    ran = []
    for module, name in [(pipeline.time_series, "prep"), (pipeline.training, "train"),
                         (pipeline.forecast, "forecast"), (pipeline.metrics, "rank"),
                         (pipeline.report, "report"), (pipeline.underrated, "underrated"),
                         (pipeline.export, "export"),
                         (pipeline.publish, "publish")]:
        monkeypatch.setattr(module, "main", lambda name=name: ran.append(name))
    monkeypatch.setattr(pipeline, "sentiment", lambda: ran.append("sentiment"))
    return ran


@pytest.fixture
def scrape_ok(monkeypatch):
    monkeypatch.setattr(pipeline.pricecharting, "main", lambda snapshot: None)


def test_run_is_price_only_by_default(scrape_ok, stages):
    pipeline.run()
    assert stages == ["prep", "train", "forecast", "rank", "report", "underrated", "export", "publish"]


def test_run_with_sentiment_collects_it_before_ranking(scrape_ok, stages):
    cli.main(["run", "--sentiment"])
    assert stages == ["sentiment", "prep", "train", "forecast", "rank", "report", "underrated", "export", "publish"]


def test_run_ranks_partial_scrape_then_fails(monkeypatch, stages):
    def partial_scrape(snapshot):
        paths.prices().write_text("console,game,price,date\n")
        raise pricecharting.ScrapeError("1 of 2 games failed")

    monkeypatch.setattr(pipeline.pricecharting, "main", partial_scrape)
    with pytest.raises(pricecharting.ScrapeError, match="1 of 2"):
        pipeline.run()
    assert stages == ["prep", "train", "forecast", "rank", "report", "underrated", "export", "publish"]


def failing_publish(stages):
    def publish():
        stages.append("publish")
        raise pipeline.publish.PublishError("cannot confirm o/n is private")
    return publish


def test_run_finishes_every_stage_then_reports_a_publish_failure(scrape_ok, stages, monkeypatch):
    monkeypatch.setattr(pipeline.publish, "main", failing_publish(stages))
    with pytest.raises(pipeline.publish.PublishError, match="private"):
        pipeline.run()
    assert stages == ["prep", "train", "forecast", "rank", "report", "underrated", "export", "publish"]


def test_scrape_failure_wins_over_publish_failure(monkeypatch, stages):
    def partial_scrape(snapshot):
        paths.prices().write_text("console,game,price,date\n")
        raise pricecharting.ScrapeError("1 of 2 games failed")

    monkeypatch.setattr(pipeline.pricecharting, "main", partial_scrape)
    monkeypatch.setattr(pipeline.publish, "main", failing_publish(stages))
    with pytest.raises(pricecharting.ScrapeError, match="1 of 2"):
        pipeline.run()
    assert stages[-1] == "publish"


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


def test_run_with_games_seeds_them_first(scrape_ok, stages):
    cli.main(["run", "--game", "nes/legend-of-zelda", "--game", "sega-genesis/sonic-the-hedgehog"])
    assert paths.games().read_text().splitlines()[1:] == [
        "nes,legend-of-zelda,Legend Of Zelda,https://www.pricecharting.com/game/nes/legend-of-zelda",
        "sega-genesis,sonic-the-hedgehog,Sonic The Hedgehog,https://www.pricecharting.com/game/sega-genesis/sonic-the-hedgehog",
    ]
    assert stages[-1] == "publish"
