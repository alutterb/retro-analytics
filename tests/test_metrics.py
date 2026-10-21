import pickle

import numpy as np
import pandas as pd
import pytest

from retro_analytics import metrics, paths


def predictions_frame():
    return pd.DataFrame({
        ("nes", "zelda"): [100.0, 102.0, 104.0, 106.0, 108.0],
        ("snes", "chrono"): [300.0, 300.0, 300.0, 300.0, 300.0],
        ("genesis", "sonic"): [20.0, 19.0, 18.0, 17.0, 16.0],
    })


def sentiment(pos, neg):
    return {"neg": neg, "neu": 1 - pos - neg, "pos": pos, "compound": pos - neg}


def comments_frame():
    return pd.DataFrame({
        "game": ["zelda", "chrono", "sonic"],
        "score": [10, 4, 2],
        "comment sentiment": [sentiment(0.5, 0.0), sentiment(0.2, 0.1), sentiment(0.0, 0.5)],
    })


def posts_frame():
    return pd.DataFrame({
        "game": ["zelda", "chrono", "sonic"],
        "score": [3, 1, 1],
        "title sentiment": [sentiment(0.4, 0.0), sentiment(0.0, 0.0), sentiment(0.0, 0.2)],
        "body sentiment": [sentiment(0.1, 0.0), metrics.default, sentiment(0.0, 0.0)],
    })


def test_reg_slopes_match_linear_trend():
    slopes = metrics.reg_slopes(predictions_frame())
    assert slopes[("nes", "zelda")] == pytest.approx(2.0)
    assert slopes[("snes", "chrono")] == pytest.approx(0.0)
    assert slopes[("genesis", "sonic")] == pytest.approx(-1.0)


def test_rank_combines_metrics_and_sorts():
    ranked = metrics.rank(predictions_frame(), comments_frame(), posts_frame())

    assert list(ranked.columns) == [
        "console", "game", "comments metric", "posts metric", "slope", "percent_change", "combined_metric",
    ]
    assert list(ranked["game"]) == ["zelda", "chrono", "sonic"]

    zelda = ranked.set_index("game").loc["zelda"]
    assert zelda["comments metric"] == pytest.approx(5.0)
    assert zelda["posts metric"] == pytest.approx(1.5)
    assert zelda["percent_change"] == pytest.approx(8.0)
    assert zelda["combined_metric"] == pytest.approx(0.1 * 5.0 + 0.1 * 1.5 + 0.4 * 2.0 + 0.4 * 8.0)
    assert np.isfinite(ranked["combined_metric"]).all()


def test_main_reads_pipeline_files_and_writes_results():
    with open(paths.predictions(), "wb") as f:
        pickle.dump(predictions_frame().to_dict("list"), f)
    comments = comments_frame()
    comments["comment sentiment"] = comments["comment sentiment"].astype(str)
    comments.to_csv(paths.preprocessed_comments())
    posts = posts_frame()
    posts["title sentiment"] = posts["title sentiment"].astype(str)
    posts["body sentiment"] = ["Missing" if s == metrics.default else str(s) for s in posts["body sentiment"]]
    posts.to_csv(paths.preprocessed_posts())

    metrics.main()

    results = pd.read_csv(paths.results())
    assert list(results["game"]) == ["zelda", "chrono", "sonic"]
