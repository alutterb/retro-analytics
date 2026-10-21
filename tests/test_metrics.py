import pandas as pd
import pytest

from retro_analytics import PipelineError, metrics, paths


def series_frame():
    return pd.DataFrame({
        "console": ["nes", "nes", "super-nintendo", "super-nintendo", "sega-genesis", "sega-genesis", "nes", "nes"],
        "game": ["zelda", "zelda", "chrono", "chrono", "sonic", "sonic", "metroid", "metroid"],
        "month": ["2026-09-01", "2026-10-01"] * 4,
        "price": [95.0, 100.0, 290.0, 300.0, 21.0, 20.0, 49.0, 50.0],
    })


def forecasts_frame(final):
    rows = []
    for (console, game), (first, last) in final.items():
        rows += [(console, game, 1, "2026-11-01", first), (console, game, 2, "2026-12-01", last)]
    return pd.DataFrame(rows, columns=["console", "game", "step", "month", "forecast"])


FINAL = {
    ("nes", "zelda"): (105.0, 110.0),            # +10%
    ("super-nintendo", "chrono"): (310.0, 330.0),  # +10%
    ("sega-genesis", "sonic"): (19.0, 18.0),       # -10%
    ("nes", "metroid"): (52.0, 55.0),              # +10%
}


def test_rank_orders_by_forecast_growth_without_sentiment():
    final = dict(FINAL)
    final[("nes", "metroid")] = (52.0, 60.0)  # +20%
    ranked = metrics.rank(series_frame(), forecasts_frame(final))
    assert list(ranked.columns) == metrics.RESULT_COLUMNS
    assert ranked["game"].tolist()[0] == "metroid"
    assert ranked["game"].tolist()[-1] == "sonic"
    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    metroid = ranked.iloc[0]
    assert metroid["last_price"] == 50.0
    assert metroid["forecast_price"] == 60.0
    assert metroid["forecast_month"] == "2026-12-01"
    assert metroid["growth_pct"] == pytest.approx(20.0)
    assert ranked["sentiment"].isna().all()
    assert (ranked["sentiment_texts"] == 0).all()
    assert ranked["title"].tolist()[0] == "metroid"


def test_sentiment_breaks_ties_and_missing_sentiment_is_neutral():
    sentiment = pd.DataFrame({
        "console": ["nes", "super-nintendo", "sega-genesis"],
        "game": ["zelda", "chrono", "sonic"],
        "sentiment": [-0.4, 0.6, 0.9],
        "n_texts": [12, 30, 4],
    })
    ranked = metrics.rank(series_frame(), forecasts_frame(FINAL), sentiment)
    # Three games tie at +10%: positive chrono, neutral (no data) metroid, negative zelda.
    assert ranked["game"].tolist() == ["chrono", "metroid", "zelda", "sonic"]
    assert ranked.set_index("game").loc["metroid", "sentiment_texts"] == 0
    assert pd.isna(ranked.set_index("game").loc["metroid", "sentiment"])


def test_sentiment_never_outranks_real_growth_differences():
    final = dict(FINAL)
    final[("nes", "zelda")] = (105.0, 110.5)  # +10.5%, beats +10% despite bad sentiment
    sentiment = pd.DataFrame({"console": ["nes"], "game": ["zelda"], "sentiment": [-1.0], "n_texts": [5]})
    ranked = metrics.rank(series_frame(), forecasts_frame(final), sentiment)
    assert ranked["game"].iloc[0] == "zelda"


def test_titles_are_used_when_known():
    titles = pd.DataFrame({"console": ["nes"], "game": ["zelda"], "title": ["Legend of Zelda"], "url": ["x"]})
    ranked = metrics.rank(series_frame(), forecasts_frame(FINAL), titles=titles).set_index("game")
    assert ranked.loc["zelda", "title"] == "Legend of Zelda"
    assert ranked.loc["chrono", "title"] == "chrono"


def test_main_works_price_only_and_picks_up_sentiment_when_present():
    series_frame().to_csv(paths.series(), index=False)
    forecasts_frame(FINAL).to_csv(paths.forecasts(), index=False)

    metrics.main()
    assert len(pd.read_csv(paths.results())) == 4

    pd.DataFrame({"console": ["nes"], "game": ["zelda"], "sentiment": [0.9], "n_texts": [3]}).to_csv(
        paths.sentiment(), index=False)
    metrics.main()
    assert pd.read_csv(paths.results())["game"].iloc[0] == "zelda"


def test_main_without_forecasts_says_which_stage_to_run():
    series_frame().to_csv(paths.series(), index=False)
    with pytest.raises(PipelineError, match="run `ra forecast` first"):
        metrics.main()
