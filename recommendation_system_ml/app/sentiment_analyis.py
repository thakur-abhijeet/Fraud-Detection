import pandas as pd
from textblob import TextBlob

def clean_review(text):
    return str(text).replace("\n", " ").strip()

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def apply_sentiment(df, review_col="review_text"):
    df["review_cleaned"] = df[review_col].fillna("").apply(clean_review)
    df["sentiment_score"] = df["review_cleaned"].apply(get_sentiment)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/reviews.csv")
    df = apply_sentiment(df)
    df.to_csv("data/reviews_with_sentiment.csv", index=False)
