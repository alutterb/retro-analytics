import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
preprocessed_path = os.path.join(script_dir, '../../data/preprocessed/scraped_data')
posts_path = os.path.join(script_dir, '../../data/raw/posts.csv')
comments_path = os.path.join(script_dir, '../../data/raw/comments.csv')


def calculate_sentiment(text):
    print("Calculating sentiment...")
    if type(text) == str:
        sid = SentimentIntensityAnalyzer()
        tokens = word_tokenize(text)

        # remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]

        # lemmatize the text
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # join the tokens back into a string
        processed_text = ' '.join(tokens)

        # use the sentiment analyzer to get the sentiment score
        score = sid.polarity_scores(processed_text)

        print("Completed sentiment calcuation")
    else:
        return "Missing"

    return score

def main():
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path)

    posts_df['title sentiment'] = posts_df['title'].apply(calculate_sentiment)
    posts_df['body sentiment'] = posts_df['body'].apply(calculate_sentiment)
    
    comments_df['comment sentiment'] = comments_df['comment_text'].apply(calculate_sentiment)

    preprocessed_posts_path = os.path.join(preprocessed_path, 'preprocessed_posts.csv')
    posts_df.to_csv(preprocessed_posts_path)
    
    preprocessed_comments_path = os.path.join(preprocessed_path, 'preprocessed_comments.csv')
    comments_df.to_csv(preprocessed_comments_path)
    

if __name__ == "__main__":
    main()