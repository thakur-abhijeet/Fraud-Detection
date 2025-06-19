
import pandas as pd
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender
from sentiment_based_filtering import SentimentBasedRecommender

def train_all():
    print("📥 Loading data...")
    ratings_df = pd.read_csv("data/user_ratings.csv")
    products_df = pd.read_csv("data/products.csv")
    reviews_df = pd.read_csv("data/reviews.csv")

    print("📈 Training Collaborative Filtering...")
    cf = CollaborativeFilteringRecommender()
    cf.train(ratings_df)

    print("🧠 Training Content-Based Filtering...")
    cbf = ContentBasedRecommender()
    cbf.train(products_df)

    print("❤️‍🔥 Processing Sentiment Scores...")
    sentiment = SentimentBasedRecommender()
    sentiment.train(reviews_df)
    sentiment.save_model()

    print("✅ All models trained and saved!")

if __name__ == "__main__":
    train_all()
