import requests
import re
import csv
import os
import praw
import pandas as pd
import json

from datetime import datetime

script_dir = os.path.dirname(os.path.realpath(__file__))
posts_path = os.path.join(script_dir, '../../data/raw/reddit_posts.csv')
subreddits_path = os.path.join(script_dir, '../../data/raw/reddit_posts.csv')
creds_path = os.path.join(script_dir, '../../data/reddit_credentials.json')
ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"

def return_reddit_credentials():
    f = open(creds_path)
    creds = json.load(f)
    f.close()
    client_id = creds['client_id']
    secret_key = creds['secret_key']

    return client_id, secret_key


# searches specified subreddit with keyword search    
def search_subreddit(subname, search, limit):
    client_id, secret_key = return_reddit_credentials()
    reddit_read_only = praw.Reddit(client_id=client_id,
        client_secret=secret_key,
        user_agent=ua)
    subreddit = reddit_read_only.subreddit(subname)
    posts = subreddit.search(search, limit=limit)

    return posts

def return_post_comments(post, search):
    comments = post.comments.list()
    relevant_comments = []
    for comment in comments:
        if search in comment.body:
            relevant_comments.append([post.id, comment.id, comment.score, comment.body, comment.created_utc])
    df = pd.DataFrame(relevant_comments, columns=['post_id', 'comment_id', 'score', 'comment_text', 'timestamp'])
    return df

def return_post_info(post):
    pass

def main():
    posts = search_subreddit('gamecollecting', 'zelda', limit=10)
    for post in posts:
        df = return_post_comments(post, 'zelda')
        print(df)
        break


if __name__ == "__main__":
    main()

