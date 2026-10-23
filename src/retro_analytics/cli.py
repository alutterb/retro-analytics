"""Command-line entry point: ``ra <stage>``."""

import argparse
import os
import sys

# The models are small; BLAS threads only fight each other. Must be set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from retro_analytics import PipelineError, export, forecast, metrics, pipeline, pricecharting, report, training


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


def add_run_args(parser):
    add_scrape_args(parser)
    parser.add_argument('--sentiment', action='store_true',
                        help="also collect Reddit sentiment as a tie-breaker (needs the sentiment extra and credentials)")
    parser.add_argument('--game', action='append', dest='games', metavar='CONSOLE/SLUG',
                        help="track exactly these games (re-seeds games.csv first; repeatable)")


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
    "sentiment": (lambda args: pipeline.sentiment(),
                  "optional: score Reddit sentiment per game (needs the sentiment extra)", None),
    "prep": (lambda args: pipeline.prep(), "turn raw prices into one monthly series per game", None),
    "train": (lambda args: training.main(), "choose a forecasting model per game", None),
    "forecast": (lambda args: forecast.main(), "forecast each game's price six months ahead", None),
    "rank": (lambda args: metrics.main(), "rank games by forecast price growth into results.csv", None),
    "report": (lambda args: report.main(), "write the ranking as a static HTML page (report.html)", None),
    "export": (lambda args: export.main(),
               "write the rising and crashing games as JSON for the phone app (latest.json)", None),
    "run": (lambda args: pipeline.run(snapshot=args.snapshot, with_sentiment=args.sentiment, games=args.games),
            "scrape, prep, train, forecast, rank and report in one go", add_run_args),
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
    except (PipelineError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
