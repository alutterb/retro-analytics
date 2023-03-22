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

def main():
    predictions, comments, posts = load_datasets()

    # multiply score by sentiment for posts and comments and group by game
    comments['metric'] = comments.apply(lambda x : x['score'] * x['comment sentiment']['pos'], axis = 1)
    grouped_comments = comments.groupby('game')['metric'].sum()

    # now for posts
    posts['metric'] = posts.apply(lambda x : x['score']*(x['title sentiment']['pos'] + x['body sentiment']['pos']), axis = 1)
    grouped_posts = posts.groupby('game')['metric'].sum()

if __name__ == "__main__":
    main()