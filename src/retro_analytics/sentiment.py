import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from retro_analytics import paths


def calculate_sentiment(text):
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
    else:
        return "Missing"

    return score

def main():
    posts_df = pd.read_csv(paths.reddit_posts())
    comments_df = pd.read_csv(paths.reddit_comments())

    posts_df['title sentiment'] = posts_df['title'].apply(calculate_sentiment)
    posts_df['body sentiment'] = posts_df['body'].apply(calculate_sentiment)
    
    comments_df['comment sentiment'] = comments_df['comment_text'].apply(calculate_sentiment)

    posts_df.to_csv(paths.preprocessed_posts())
    comments_df.to_csv(paths.preprocessed_comments())

if __name__ == "__main__":
    main()