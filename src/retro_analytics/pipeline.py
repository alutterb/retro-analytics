"""Run each pipeline stage in order, from scraping to the ranked results."""

from retro_analytics import arima_prep, forecast, metrics, pricecharting, time_series, training


def prep():
    time_series.main()
    arima_prep.main()


def run():
    print("executing update")
    pricecharting.main()
    prep()
    training.main()
    forecast.main()
    metrics.main()
    print("complete")
