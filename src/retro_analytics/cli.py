"""Command-line entry point: ``ra <stage>``."""

import argparse

from retro_analytics import forecast, metrics, pipeline, pricecharting, training


def sentiment():
    # Reddit scraping and NLTK are an optional extra: uv sync --extra sentiment
    from retro_analytics import reddit
    from retro_analytics import sentiment as sentiment_prep

    reddit.main()
    sentiment_prep.main()


COMMANDS = {
    "scrape": (pricecharting.main, "fetch current CIB prices from PriceCharting"),
    "sentiment": (sentiment, "search Reddit and score sentiment (needs the sentiment extra)"),
    "prep": (pipeline.prep, "turn raw prices into model input"),
    "train": (training.main, "fit an ARIMA model per game"),
    "forecast": (forecast.main, "forecast each fitted model forward"),
    "rank": (metrics.main, "combine forecasts and sentiment into results.csv"),
    "run": (pipeline.run, "scrape, prep, train, forecast and rank in one go"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ra", description="Rank retro video games by forecast price trend.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (_, help_text) in COMMANDS.items():
        commands.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    COMMANDS[args.command][0]()


if __name__ == "__main__":
    main()
