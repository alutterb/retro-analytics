#!/home/akagi/Documents/Projects/.venv/bin python3

import sys
# Add the paths containing the other scripts
sys.path.append('/home/akagi/Documents/Projects/retro-analytics/src/data_collection/')
sys.path.append('/home/akagi/Documents/Projects/retro-analytics/src/data_preprocessing')
sys.path.append('/home/akagi/Documents/Projects/retro-analytics/src/modeling')
sys.path.append('/home/akagi/Documents/Projects/retro-analytics/src/evaluation')

import pricecharting_scraper
import time_series_preprocessing
import ARIMA_preprocessing
import training
import evaluate
import metrics
import os

# Define the output path for the final metrics
script_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    # Run the individual scripts
    pricecharting_scraper.main()
    time_series_preprocessing.main()
    ARIMA_preprocessing.main()
    training.main()
    evaluate.main()
    metrics.main()

if __name__ == "__main__":
    main()
