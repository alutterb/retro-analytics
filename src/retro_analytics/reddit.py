import multiprocessing
import praw
import pandas as pd
import json

from retro_analytics import paths

ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"

def return_reddit_credentials():
    with open(paths.reddit_credentials()) as f:
        creds = json.load(f)
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

def return_post_comments(post, game):
    post.comments.replace_more(limit=None)
    comments = post.comments.list()
    relevant_comments = []
    for comment in comments:
        relevant_comments.append([game, post.id, comment.id, comment.score, comment.body, comment.created_utc])
    df = pd.DataFrame(relevant_comments, columns=['game', 'post_id', 'comment_id', 'score', 'comment_text', 'timestamp'])
    return df

def return_posts_info(posts, game):
    data = []
    for post in posts:
        body = None
        if post.is_self: # if text post
            body = post.selftext
        title = post.title
        id = post.id
        score = post.score
        data.append([game, id, title, body, score])
    df = pd.DataFrame(data, columns=['game','ID', 'title', 'body', 'score'])
    return df

def return_game_data(game):
    subreddits = ['gamecollecting','retrogaming']
    LIMIT = 50
    print("Searching for game: %s" % game)
    post_df = pd.DataFrame(columns=['game', 'ID', 'title', 'body', 'score'])
    comments_df = pd.DataFrame(columns=['game', 'post_id', 'comment_id', 'score', 'comment_text', 'timestamp'])
    for sub in subreddits:
        print("Searching in sub: %s" % sub)
        posts = list(search_subreddit(subname=sub, search=game, limit=LIMIT))
        post_df = pd.concat([post_df, return_posts_info(posts, game)], ignore_index=True)
        if posts:
            for post in posts:
                print("Searching in post: %s" % post.title)
                comments_df = pd.concat([comments_df, return_post_comments(post, game)], ignore_index=True)
        else:
            comments_df = None
    return post_df, comments_df

def main():
    games = pd.read_csv(paths.prices())['game'].unique()
    pool = multiprocessing.Pool(processes=6)
    results = pool.map(return_game_data, games)
    posts_list = []
    comments_list = []
    for result in results:
        # assuming each result is a tuple of posts and comments dataframes
        posts, comments = result
        posts_list.append(posts)
        comments_list.append(comments)
    
    # create final dataframes from the lists
    posts_df = pd.concat(posts_list, ignore_index=True)
    comments_df = pd.concat(comments_list, ignore_index=True)
    
    # save dataframes
    posts_df.to_csv(paths.reddit_posts(), index=False)
    comments_df.to_csv(paths.reddit_comments(), index=False)


if __name__ == "__main__":
    main()

