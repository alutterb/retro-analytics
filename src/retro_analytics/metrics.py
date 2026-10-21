## Combine forecasting results with sentiment analysis of each game
## Forecasted increase in game price + positive sentiment = good pick
### Follow up qn: which came first - the increasing game price or the positive sentiment? 
### Did positive sentiment cause this increase in price?

## Other scenarios: price increase while negative sentiment, price decrease with positve sentiment
## price decrease and negative sentiment
## Clearly, the last option should be ignored, but how would we interpret the two former options?
## Perhaps price increase with negative sentiment indicates that there soon will be a price decrease
## due to negative sentiment
## whereas for the other option, the price may be decreasing currently; however, positive sentiment
## may push it towards and upward trend
## These questions require further investigation and are outside the scope of the project. For now,
## the project will focus on increase in game price + positive sentiment to flag good purchases


import ast
import pickle

import numpy as np
import pandas as pd

from retro_analytics import paths

## Weights for each metric
comments_metric_weight = 0.1
posts_metric_weight = 0.1
slope_weight = 0.4
percentage_change_weight = 0.4


# use if missing sentiment
default = {'neg' : 0, 'neu' : 0, 'pos' : 0, 'compound' : 0}


def parse_sentiment(value):
    return ast.literal_eval(value) if value != 'Missing' else default


def load_datasets():
    # ARIMA forecasts
    with open(paths.predictions(), 'rb') as f:
        predictions = pickle.load(f)
    predictions = pd.DataFrame(predictions)

    # comments sentiment
    comments  = pd.read_csv(paths.preprocessed_comments())
    comments = comments.drop('Unnamed: 0', axis=1)
    comments['comment sentiment'] = comments['comment sentiment'].apply(parse_sentiment)

    # posts sentiment
    posts = pd.read_csv(paths.preprocessed_posts())
    posts = posts.drop('Unnamed: 0', axis=1)
    posts['title sentiment'] = posts['title sentiment'].apply(parse_sentiment)
    posts['body sentiment'] = posts['body sentiment'].apply(parse_sentiment)

    return predictions, comments, posts

def reg_slopes(timeseries):
    """Least-squares slope of each column against its row position."""
    x = np.arange(timeseries.shape[0])
    slopes = [np.polyfit(x, timeseries[game].to_numpy(dtype=float), 1)[0] for game in timeseries.columns]
    return pd.Series(slopes, index=timeseries.columns, name='slope')

def rank(predictions, comments, posts):
    """Combine forecast trend and Reddit sentiment into one score per game, best first.

    `predictions` has one column per (console, game) holding its forecast points.
    """
    # multiply score by sentiment for posts and comments and group by game
    comments['comments metric'] = comments.apply(lambda x : x['score'] * (x['comment sentiment']['pos'] - x['comment sentiment']['neg']), axis = 1)
    grouped_comments = comments.groupby('game')['comments metric'].sum()

    # now for posts
    posts['posts metric'] = posts.apply(lambda x : x['score']*(x['title sentiment']['pos'] - x['title sentiment']['neg'] + x['body sentiment']['pos'] - x['body sentiment']['neg']), axis = 1)
    grouped_posts = posts.groupby('game')['posts metric'].sum()

    # and merge these dataframes together
    grouped_df = pd.merge(grouped_comments, grouped_posts, how='inner', on='game')

    # now calculate metric based on predictions
    slope_series = reg_slopes(predictions)

    # Also compare the first and last values, then sorting by the highest percent increase
    percent_changes = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0] * 100
    percent_changes.name = 'percent_change'

    # Create a new dataframe with the aggregated data
    aggregated_df = pd.concat([slope_series, percent_changes], axis=1).reset_index()
    aggregated_df.columns = ['console', 'game', 'slope', 'percent_change']

    # Next, merge the aggregated_df dataframe with the grouped_df dataframe
    merged_df = grouped_df.merge(aggregated_df, on='game')

    # Define the new column order
    column_order = ['console', 'game', 'comments metric', 'posts metric', 'slope', 'percent_change']

    # Reorganize the columns based on the new order
    merged_df = merged_df[column_order].copy()

    # Calculate the combined metric
    merged_df['combined_metric'] = (
        comments_metric_weight * merged_df['comments metric'] +
        posts_metric_weight * merged_df['posts metric'] +
        slope_weight * merged_df['slope'] +
        percentage_change_weight * merged_df['percent_change']
    )

    # Sort the DataFrame based on the combined metric in descending order
    merged_df = merged_df.sort_values(by='combined_metric', ascending=False)

    '''
    CONSOLES: 
    {'xbox-360', 'sega-genesis', 'nintendo-64', 'wii', 'playstation-3', 'sega-dreamcast', 'gamecube', 'sega-cd', 
    'gameboy-color', 'nintendo-ds', 'super-nintendo', 'nintendo-switch', 'playstation', 'nes', 
    'xbox', 'psp', 'playstation-4', 'gameboy', 'gameboy-advance', 'xbox-one', 'playstation-2'}
    '''

    return merged_df


def main():
    rank(*load_datasets()).to_csv(paths.results(), index=False)

if __name__ == "__main__":
    main()