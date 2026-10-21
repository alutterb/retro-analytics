"""Command-line entry point: ``ra <stage>``."""

import argparse
import sys

from retro_analytics import forecast, metrics, pipeline, pricecharting, training


def sentiment(args):
    # Reddit scraping and NLTK are an optional extra: uv sync --extra sentiment
    from retro_analytics import reddit
    from retro_analytics import sentiment as sentiment_prep

    reddit.main()
    sentiment_prep.main()


def add_seed_args(parser):
    parser.add_argument('--console', action='append', dest='consoles', metavar='SLUG',
                        help="console to seed from (repeatable; default: the 21 tracked consoles)")
    parser.add_argument('--top', type=int, default=25, help="games per console (default: 25)")
    parser.add_argument('--min-price', type=float, default=20.0,
                        help="lowest CIB price in dollars to include (default: 20)")
    parser.add_argument('--game', action='append', dest='games', metavar='CONSOLE/SLUG',
                        help="track exactly these games instead of reading console pages (repeatable)")


def add_scrape_args(parser):
    parser.add_argument('--snapshot', action='store_true',
                        help="also record the price shown today, not only the monthly history")


def seed(args):
    pricecharting.seed(
        consoles=args.consoles or pricecharting.CONSOLES,
        top=args.top,
        min_price=args.min_price,
        games=args.games,
    )


COMMANDS = {
    "seed": (seed, "choose which games to track (writes games.csv)", add_seed_args),
    "scrape": (lambda args: pricecharting.main(snapshot=args.snapshot),
               "fetch CIB price history for the seeded games", add_scrape_args),
    "sentiment": (sentiment, "search Reddit and score sentiment (needs the sentiment extra)", None),
    "prep": (lambda args: pipeline.prep(), "turn raw prices into model input", None),
    "train": (lambda args: training.main(), "fit an ARIMA model per game", None),
    "forecast": (lambda args: forecast.main(), "forecast each fitted model forward", None),
    "rank": (lambda args: metrics.main(), "combine forecasts and sentiment into results.csv", None),
    "run": (lambda args: pipeline.run(snapshot=args.snapshot),
            "scrape, prep, train, forecast and rank in one go", add_scrape_args),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ra", description="Rank retro video games by forecast price trend.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (_, help_text, add_args) in COMMANDS.items():
        command = commands.add_parser(name, help=help_text)
        if add_args:
            add_args(command)
    args = parser.parse_args(argv)
    try:
        COMMANDS[args.command][0](args)
    except pricecharting.ScrapeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
