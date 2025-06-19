
import pandas as pd
import re
from textblob import TextBlob
import nltk
import joblib

nltk.download("stopwords")
from nltk.corpus import stopwords


class SentimentBasedRecommender:
    def __init__(self, model_path="model/sentiment_model.pkl"):
        self.model_path = model_path
        self.product_sentiments = None

    def clean_review(self, text):
        """
        Lowercase, remove punctuation and stopwords.
        """
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        words = text.split()
        return " ".join([w for w in words if w not in set(stopwords.words("english"))])

    def get_sentiment_score(self, text):
        """
        Returns polarity score: -1 (negative) to 1 (positive)
        """
        return TextBlob(text).sentiment.polarity

    def preprocess_reviews(self, df):
        """
        Cleans and scores all reviews.
        Expected columns: product_id, review_text
        """
        df = df.copy()
        df["review_text"] = df["review_text"].fillna("")
        df["cleaned_review"] = df["review_text"].apply(self.clean_review)
        df["sentiment_score"] = df["cleaned_review"].apply(self.get_sentiment_score)
        return df

    def aggregate_sentiment_scores(self, df):
        """
        Aggregates sentiment scores at the product level.
        """
        product_sentiment = (
            df.groupby("product_id")["sentiment_score"]
            .mean()
            .reset_index()
            .rename(columns={"sentiment_score": "avg_sentiment_score"})
        )
        self.product_sentiments = product_sentiment
        return product_sentiment

    def recommend_top_by_sentiment(self, top_n=5):
        """
        Returns top-N most positively reviewed products.
        """
        if self.product_sentiments is None:
            raise ValueError("Run train() first.")

        top = self.product_sentiments.sort_values(
            by="avg_sentiment_score", ascending=False
        ).head(top_n)

        return top.to_dict(orient="records")

    def train(self, df_reviews):
        """
        Full pipeline from raw review data to aggregated sentiment scores.
        """
        cleaned_df = self.preprocess_reviews(df_reviews)
        return self.aggregate_sentiment_scores(cleaned_df)

    def save_model(self):
        joblib.dump(self.product_sentiments, self.model_path)

    def load_model(self):
        self.product_sentiments = joblib.load(self.model_path)


# For standalone test/demo
if __name__ == "__main__":
    # Example: Load and run on review data
    reviews_df = pd.read_csv("data/reviews.csv")  # Requires: product_id, review_text

    sentiment_recommender = SentimentBasedRecommender()
    sentiment_recommender.train(reviews_df)

    top_reviews = sentiment_recommender.recommend_top_by_sentiment(top_n=5)
    print("Top positively reviewed products:")
    for product in top_reviews:
        print(product)
