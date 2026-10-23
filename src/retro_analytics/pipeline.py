"""Run each pipeline stage in order, from scraping to the published results."""

from retro_analytics import export, forecast, metrics, paths, pricecharting, publish, report, time_series, training


def prep():
    time_series.main()


def sentiment():
    # Reddit scraping and NLTK are an optional extra: uv sync --extra sentiment
    try:
        from retro_analytics import reddit
        from retro_analytics import sentiment as sentiment_scores
    except ImportError as e:
        raise ImportError(f"{e}; install the optional extra with `uv sync --extra sentiment`") from e

    reddit.main()
    sentiment_scores.main()


def run(snapshot=False, with_sentiment=False, games=None):
    """Scrape, model, rank, report, export and publish. `games` ('console/slug') re-seeds first."""
    print("executing update")
    if games:
        pricecharting.seed(games=games)
    scrape_error = None
    try:
        pricecharting.main(snapshot=snapshot)
    except pricecharting.ScrapeError as e:
        # Some games failed: rank the rest, then still exit non-zero.
        if not paths.prices().exists():
            raise
        scrape_error = e
    if with_sentiment:
        sentiment()
    prep()
    training.main()
    forecast.main()
    metrics.main()
    report.main()
    export.main()
    publish_error = None
    try:
        publish.main()
    except publish.PublishError as e:
        # The export is written; report the failure after the run, like a partial scrape.
        print(f"publish failed: {e}")
        publish_error = e
    error = scrape_error or publish_error
    if error:
        raise error
    print("complete")
