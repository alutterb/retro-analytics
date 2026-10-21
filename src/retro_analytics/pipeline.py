"""Run each pipeline stage in order, from scraping to the ranked results."""

from retro_analytics import forecast, metrics, paths, pricecharting, time_series, training


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


def run(snapshot=False, with_sentiment=False):
    print("executing update")
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
    if scrape_error:
        raise scrape_error
    print("complete")
