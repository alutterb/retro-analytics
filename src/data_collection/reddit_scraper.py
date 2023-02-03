import requests
import re
import csv
import os
import praw
import pandas as pd

from datetime import datetime

'''
look into threading 
'''

script_dir = os.path.dirname(os.path.realpath(__file__))
posts_path = os.path.join(script_dir, '../../data/raw/reddit_posts.csv')
subreddits_path = os.path.join(script_dir, '../../data/raw/reddit_posts.csv')
ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    
def search_subreddit(self, subname, search):
    reddit_read_only = praw.Reddit(client_id=self.client_id,
        client_secret=self.secret_key,
        user_agent=self.ua)
    
    subreddit = reddit_read_only.subreddit(subname)
    posts = subreddit.search(search, limit=50)

    return posts

