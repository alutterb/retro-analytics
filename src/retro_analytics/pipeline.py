"""Run each pipeline stage in order, from scraping to the ranked results."""

from retro_analytics import arima_prep, forecast, metrics, paths, pricecharting, time_series, training


def prep():
    time_series.main()
    arima_prep.main()


def run(snapshot=False):
    print("executing update")
    scrape_error = None
    try:
        pricecharting.main(snapshot=snapshot)
    except pricecharting.ScrapeError as e:
        # Some games failed: rank the rest, then still exit non-zero.
        if not paths.prices().exists():
            raise
        scrape_error = e
    prep()
    training.main()
    forecast.main()
    metrics.main()
    if scrape_error:
        raise scrape_error
    print("complete")
