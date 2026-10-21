#!/usr/bin/env bash
# End-to-end example on three games: seeds them, fetches their price history
# from PriceCharting (three requests, 1-2 s apart), forecasts, ranks and writes
# the HTML report. Output goes to a separate data directory so it does not mix
# with your main tracked set.
set -euo pipefail

cd "$(dirname "$0")/.."
export RA_DATA_DIR="${RA_DATA_DIR:-$PWD/data/example}"
export RA_MODELS_DIR="${RA_MODELS_DIR:-$PWD/models/example}"

uv run ra run \
  --game nes/legend-of-zelda \
  --game super-nintendo/chrono-trigger \
  --game sega-genesis/sonic-the-hedgehog

echo "Ranking: $RA_DATA_DIR/outputs/results.csv"
echo "Report:  $RA_DATA_DIR/outputs/report.html"
