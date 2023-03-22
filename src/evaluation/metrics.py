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


import pickle
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression

script_dir = os.path.dirname(os.path.realpath(__file__))
predictions_path = os.path.join(script_dir, '../../data/outputs/predictions.pickle')
comments_path = os.path.join(script_dir, '../../data/preprocessed/scraped_data/preprocessed_comments.csv')
posts_path = os.path.join(script_dir,'../../data/preprocessed/scraped_data/preprocessed_posts.csv')

# use if missing sentiment
default = {'neg' : 0, 'neu' : 0, 'pos' : 0, 'compound' : 0}

def load_datasets():
    # ARIMA forecasts
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    predictions = pd.DataFrame(predictions)

    # comments sentiment
    comments  = pd.read_csv(comments_path)
    comments = comments.drop('Unnamed: 0', axis=1)
    comments['comment sentiment'] = comments['comment sentiment'].apply(lambda x : eval(x) if x != 'Missing' else default)

    # posts sentiment
    posts = pd.read_csv(posts_path)
    posts = posts.drop('Unnamed: 0', axis=1)
    posts['title sentiment'] = posts['title sentiment'].apply(lambda x : eval(x) if x != 'Missing' else default)
    posts['body sentiment'] = posts['body sentiment'].apply(lambda x : eval(x) if x != 'Missing' else default)

    return predictions, comments, posts

def reg_slopes(timeseries):
    slopes = []
    for game in timeseries.columns:
        X = np.arange(timeseries.shape[0]).reshape(-1,1)
        y = timeseries[game].values.reshape(-1,1)

        reg = LinearRegression()
        reg.fit(X,y)

        slopes.append(reg.coef_[0][0])

    slopes_series = pd.Series(slopes, index=timeseries.columns, name='slope')
    return slopes_series

def main():
    predictions, comments, posts = load_datasets()

    # multiply score by sentiment for posts and comments and group by game
    comments['comments metric'] = comments.apply(lambda x : x['score'] * x['comment sentiment']['pos'], axis = 1)
    grouped_comments = comments.groupby('game')['comments metric'].sum()

    # now for posts
    posts['posts metric'] = posts.apply(lambda x : x['score']*(x['title sentiment']['pos'] + x['body sentiment']['pos']), axis = 1)
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

    # Define the weights
    comments_metric_weight = 0.1
    posts_metric_weight = 0.3
    slope_weight = 0.4
    percentage_change_weight = 0.2

    # Calculate the combined metric
    merged_df['combined_metric'] = (
        comments_metric_weight * merged_df['comments metric'] +
        posts_metric_weight * merged_df['posts metric'] +
        slope_weight * merged_df['slope'] +
        percentage_change_weight * merged_df['percent_change']
    )

    # Sort the DataFrame based on the combined metric in descending order
    merged_df = merged_df.sort_values(by='combined_metric', ascending=False)

    # Print the top 10 games most likely to increase in value in the future
    print(merged_df.head(10))

if __name__ == "__main__":
    main()