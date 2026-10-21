import pandas as pd
import pytest
from bs4 import BeautifulSoup

from retro_analytics import PipelineError, paths, report


def results_frame():
    return pd.DataFrame({
        "rank": [1, 2],
        "console": ["super-nintendo", "nes"],
        "game": ["chrono-trigger", "legend-of-zelda"],
        "title": ["Chrono Trigger", "Zelda <script>alert(1)</script>"],
        "last_month": ["2026-10-01", "2026-10-01"],
        "last_price": [300.0, 1500.5],
        "forecast_month": ["2027-04-01", "2027-04-01"],
        "forecast_price": [330.0, 1450.0],
        "growth_pct": [10.0, -3.37],
        "sentiment": [0.42, None],
        "sentiment_texts": [12, 0],
    })


def series_frame():
    months = pd.date_range("2024-01-01", periods=34, freq="MS").strftime("%Y-%m-%d")
    return pd.concat([
        pd.DataFrame({"console": "super-nintendo", "game": "chrono-trigger", "month": months,
                      "price": [200.0 + 3 * i for i in range(34)]}),
        pd.DataFrame({"console": "nes", "game": "legend-of-zelda", "month": months, "price": 1500.0}),
    ])


def forecasts_frame():
    rows = []
    for console, game, start in [("super-nintendo", "chrono-trigger", 305.0), ("nes", "legend-of-zelda", 1490.0)]:
        rows += [(console, game, step, f"2026-{10 + step:02d}-01" if step < 3 else "2027-01-01", start + step)
                 for step in range(1, 7)]
    return pd.DataFrame(rows, columns=["console", "game", "step", "month", "forecast"])


def games_frame():
    return pd.DataFrame({
        "console": ["super-nintendo"], "game": ["chrono-trigger"], "title": ["Chrono Trigger"],
        "url": ["https://www.pricecharting.com/game/super-nintendo/chrono-trigger"],
    })


def page():
    html = report.render(results_frame(), series_frame(), forecasts_frame(), games_frame(), generated="2026-10-21")
    return BeautifulSoup(html, "html.parser")


def test_report_is_self_contained():
    soup = page()
    assert soup.title.string == "Retro game price outlook"
    assert not soup.find_all("script", src=True)
    assert not soup.find_all("link")
    assert "6 months" in soup.find("p").get_text()


def test_report_table_rows_in_rank_order_with_sort_keys():
    rows = page().select("#ranking tbody tr")
    assert [row["data-console"] for row in rows] == ["super-nintendo", "nes"]
    first = [td.get("data-sort") for td in rows[0].find_all("td")]
    assert first[:7] == ["1", "chrono trigger", "super-nintendo", "300.0", "330.0", "10.0", "0.42"]
    assert rows[0]["class"] == ["up"] and rows[1]["class"] == ["down"]
    cells = [td.get_text(" ", strip=True) for td in rows[1].find_all("td")]
    assert cells[3] == "$1,500.50 2026-10"
    assert cells[5] == "-3.4%"
    assert cells[6] == "–"  # no sentiment collected


def test_report_links_known_games_and_escapes_titles():
    rows = page().select("#ranking tbody tr")
    link = rows[0].find("a")
    assert link["href"] == "https://www.pricecharting.com/game/super-nintendo/chrono-trigger"
    assert rows[1].find("a") is None
    assert rows[1].find("script") is None
    assert "<script>alert(1)</script>" in rows[1].get_text()


def test_console_filter_lists_each_console_once():
    options = [o["value"] for o in page().select("#console option")]
    assert options == ["", "nes", "super-nintendo"]


def test_sparkline_draws_recent_history_then_forecast():
    svg = page().select("#ranking tbody tr")[0].find("svg")
    past = svg.find("polyline", class_="past")["points"].split()
    future = svg.find("polyline", class_="future")["points"].split()
    assert len(past) == report.SPARK_MONTHS
    assert len(future) == 7  # joins at the last known price, then six forecast months
    assert future[0] == past[-1]
    # A rising series ends higher on the page (smaller y) than it starts.
    assert float(past[-1].split(",")[1]) < float(past[0].split(",")[1])


def test_flat_sparkline_does_not_divide_by_zero():
    svg = report.sparkline([10.0, 10.0], [10.0])
    assert "nan" not in svg and "inf" not in svg


def test_main_writes_report_file():
    results_frame().to_csv(paths.results(), index=False)
    series_frame().to_csv(paths.series(), index=False)
    forecasts_frame().to_csv(paths.forecasts(), index=False)
    report.main()
    assert "Chrono Trigger" in paths.report().read_text()


def test_main_without_results_says_which_stage_to_run():
    with pytest.raises(PipelineError, match="run `ra rank` first"):
        report.main()
