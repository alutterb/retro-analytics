# retro-analytics

## DIRECTORY STRUCTURE
### Data Collection
**Price Charting Scraper**

Summary: Collects historical complete in box prices for games on pricecharting.com

**Reddit Scraper**

Summary: Collects comments and posts on several gaming subreddits

### Data Preprocessing
**Time Series Preprocessing**

Summary: Converts scraped prices data from pricecharting into a format that can be used for model preprocessing

**ARIMA Preprocessing**

Summary: Preprocessing raw time series data into a format suitable for ARIMA model training

**LSTM Preprocessing**

NOTE: Currently not implement due to processing power

**Sentiment Preprocessing**

Summary: Converts Reddit posts and comments into numeric values representing pos/neutral/neg sentiment

### Deployment
**Daily Update**

Summary: Runs the pricecharting scraper to update daily historical prices and saves results. Trains models on updated data and updates new predictions.

**Results App**

Summary: Displays the results of model analyses

### Evaluation
**Evaulate**

Summary: Generates and saves ARIMA model forecasts

**Metrics**

Summary: Generates metrics to be used to rank video games


### Modeling
**Training**

Summary: Trains ARIMA models on preprocessed time series data

## Order of Operations

1. Data is scraped from reddit with `reddit_scraper.py` and pricecharting.com with `pricecharting_scraper.py`
2. The raw reddit data is preprocessed to generate sentiment scores for each comment and post in `sentiment_preprocessing.py` and the raw historical price values are preprocessed in `time_series_preprocessing.py` and then later transformed to be ARIMA input in `ARIMA_preprocessing.py`
3. The ARIMA models are trained in `training.py` on the preprocessed time series data
4. Forecasts of the trained models are generated and saved in `evaluate.py`
5. The sentiment analysis and forecasted prices are combined in `metrics.py` to create a ranking of retro video games
6. The results are displayed in a GUI in `results_app.py`

Repeated: `daily_update.py` scrapes pricecharting.com daily to get up to date prices. The models are then retrained on this new data.