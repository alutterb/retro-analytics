import json

import pandas as pd
import pytest

from retro_analytics import PipelineError, cli, export, metrics, paths


def monthly(console, game, prices, end="2026-10-01"):
    months = pd.date_range(end=end, periods=len(prices), freq="MS").strftime("%Y-%m-%d")
    return pd.DataFrame({"console": console, "game": game, "month": months, "price": prices})


def forecast_rows(console, game, final, last=100.0, start="2026-11-01"):
    months = pd.date_range(start, periods=6, freq="MS").strftime("%Y-%m-%d")
    values = [last + (final - last) * step / 6 for step in range(1, 7)]
    return pd.DataFrame({"console": console, "game": game, "step": range(1, 7), "month": months, "forecast": values})


@pytest.fixture
def pipeline_outputs():
    """Three ranked games: zelda rising (forecast), sonic crashing (last month), metroid quiet."""
    series = pd.concat([
        monthly("nes", "zelda", [100.0] * 60),
        monthly("sega-genesis", "sonic", [100.0] * 59 + [80.0]),
        monthly("nes", "metroid", [100.0] * 60),
    ], ignore_index=True)
    forecasts = pd.concat([
        forecast_rows("nes", "zelda", 120.0),
        forecast_rows("sega-genesis", "sonic", 80.0, last=80.0),
        forecast_rows("nes", "metroid", 102.0),
    ], ignore_index=True)
    series.to_csv(paths.series(), index=False)
    forecasts.to_csv(paths.forecasts(), index=False)
    metrics.rank(series, forecasts).to_csv(paths.results(), index=False)


def read_export():
    return json.loads(paths.export().read_text(encoding="utf-8"))


def test_export_holds_only_flagged_games(pipeline_outputs):
    export.main()
    document = read_export()
    assert document["schema_version"] == 1
    assert document["horizon_months"] == 6
    assert document["thresholds"]["crashing"]["change_1m_pct"] == -10.0
    assert [g["game"] for g in document["games"]] == ["zelda", "sonic"]


def test_exported_game_carries_every_field(pipeline_outputs):
    export.main()
    sonic = next(g for g in read_export()["games"] if g["game"] == "sonic")
    assert sonic["console"] == "sega-genesis"
    assert sonic["title"] == "sonic"
    assert sonic["url"] is None
    assert sonic["last_month"] == "2026-10-01"
    assert sonic["last_price"] == 80.0
    assert sonic["changes_pct"] == {"1m": -20.0, "3m": -20.0, "6m": -20.0}
    assert sonic["forecast"]["growth_pct"] == 0.0
    assert sonic["forecast"]["months"][0] == "2026-11-01"
    assert len(sonic["forecast"]["prices"]) == 6
    assert sonic["rebound"] is None
    assert sonic["flags"][0] == {
        "direction": "crashing", "signal": "change_1m", "value_pct": -20.0, "threshold_pct": -10.0,
        "text": "down 20% in 1 month"}


def test_history_is_the_last_24_months(pipeline_outputs):
    export.main()
    zelda = read_export()["games"][0]
    assert len(zelda["history"]["months"]) == 24
    assert len(zelda["history"]["prices"]) == 24
    assert zelda["history"]["months"][-1] == "2026-10-01"


def test_titles_and_urls_come_from_games_csv(pipeline_outputs):
    pd.DataFrame({"console": ["nes"], "game": ["zelda"], "title": ["The Legend of Zelda"],
                  "url": ["https://www.pricecharting.com/game/nes/zelda"]}).to_csv(paths.games(), index=False)
    metrics.main()  # the ranking picks titles up from games.csv
    export.main()
    zelda = read_export()["games"][0]
    assert zelda["title"] == "The Legend of Zelda"
    assert zelda["url"] == "https://www.pricecharting.com/game/nes/zelda"


def test_nothing_flagged_still_writes_a_valid_file(pipeline_outputs):
    paths.settings().write_text("[rising]\nforecast_pct = 50\n[crashing]\nchange_1m_pct = -50\n"
                                "change_3m_pct = -50\nchange_6m_pct = -50\n", encoding="utf-8")
    export.main()
    assert read_export()["games"] == []


def test_missing_ranking_names_the_stage():
    with pytest.raises(PipelineError, match="run `ra rank` first"):
        export.main()


def test_cli_registers_export(monkeypatch):
    called = []
    monkeypatch.setattr(cli.export, "main", lambda: called.append("export"))
    cli.main(["export"])
    assert called == ["export"]


def test_recent_crash_with_upward_forecast_exports_a_rebound(pipeline_outputs):
    series = pd.concat([pd.read_csv(paths.series()), monthly("nes", "vmu", [100.0] * 59 + [75.0])], ignore_index=True)
    forecasts = pd.concat([pd.read_csv(paths.forecasts()), forecast_rows("nes", "vmu", 135.0, last=75.0)],
                          ignore_index=True)
    series.to_csv(paths.series(), index=False)
    forecasts.to_csv(paths.forecasts(), index=False)
    metrics.rank(series, forecasts).to_csv(paths.results(), index=False)
    export.main()
    document = read_export()
    assert document["schema_version"] == 1
    assert document["thresholds"]["rebound"] == {"forecast_pct": 10.0}
    vmu = next(g for g in document["games"] if g["game"] == "vmu")
    assert vmu["rebound"] == {
        "direction": "rebound", "signal": "change_1m", "value_pct": 80.0, "threshold_pct": 10.0,
        "text": "down 25% in 1 month, forecast up 80% in 6 months"}
    # The rebound is its own field: flags still hold only rising and crashing entries.
    assert {f["direction"] for f in vmu["flags"]} == {"rising", "crashing"}
    assert [g["game"] for g in document["games"] if g["rebound"]] == ["vmu"]


def test_export_summary_counts_rebounds(pipeline_outputs, capsys):
    export.main()
    assert "1 rising, 1 crashing and 0 rebound games" in capsys.readouterr().out


def underrated_row(game, rank, console="nes", score=1.5, **extra):
    return {"console": console, "game": game, "title": game.title(), "url": f"https://pc/{game}",
            "igdb_id": rank, "igdb_url": f"https://www.igdb.com/games/{game}", "rating": 86.64, "votes": 36,
            "popularity": 40, "shrunk": 84.0, "score": score, "rank": rank, **extra}


def test_without_underrated_scores_the_list_is_empty(pipeline_outputs):
    export.main()
    assert read_export()["underrated"] == []


def test_underrated_rows_carry_scores_and_latest_price(pipeline_outputs):
    pd.DataFrame([underrated_row("zelda", 1, score=2.04321), underrated_row("clash", 2)]).to_csv(
        paths.underrated(), index=False)
    export.main()
    zelda, clash = read_export()["underrated"]
    assert zelda == {"console": "nes", "game": "zelda", "title": "Zelda", "url": "https://pc/zelda",
                     "igdb_url": "https://www.igdb.com/games/zelda", "score": 2.04, "rank": 1, "rating": 86.6,
                     "votes": 36, "popularity": 40, "last_price": 100.0, "last_month": "2026-10-01"}
    assert clash["last_price"] is None and clash["last_month"] is None


def test_underrated_game_missing_from_results_still_gets_its_latest_price(pipeline_outputs):
    series = pd.concat([pd.read_csv(paths.series()), monthly("nes", "clash", [40.0, 55.0], end="2026-09-01")],
                       ignore_index=True)
    series.to_csv(paths.series(), index=False)
    pd.DataFrame([underrated_row("clash", 1)]).to_csv(paths.underrated(), index=False)
    export.main()
    clash, = read_export()["underrated"]
    assert clash["last_price"] == 55.0 and clash["last_month"] == "2026-09-01"


def test_underrated_keeps_each_consoles_top_25_of_tracked_games(pipeline_outputs):
    rows = [underrated_row(f"g{rank}", rank) for rank in range(1, 30)] + [underrated_row("vb", 1, "virtual-boy")]
    pd.DataFrame(rows).to_csv(paths.underrated(), index=False)
    pd.DataFrame([{"console": r["console"], "game": r["game"], "title": r["title"], "url": r["url"]}
                  for r in rows if r["game"] != "g3"]).to_csv(paths.games(), index=False)
    export.main()
    listed = [(g["console"], g["game"]) for g in read_export()["underrated"]]
    assert ("nes", "g25") in listed and ("nes", "g26") not in listed
    assert ("nes", "g3") not in listed  # no longer tracked
    assert ("virtual-boy", "vb") in listed
    assert len(listed) == 25


def test_a_non_finite_underrated_row_is_left_out(pipeline_outputs):
    pd.DataFrame([underrated_row("zelda", 1), underrated_row("clash", 2, score=float("nan"))]).to_csv(
        paths.underrated(), index=False)
    export.main()
    text = paths.export().read_text(encoding="utf-8")
    assert "NaN" not in text
    assert [g["game"] for g in json.loads(text)["underrated"]] == ["zelda"]
