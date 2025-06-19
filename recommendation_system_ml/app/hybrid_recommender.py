
import pandas as pd
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender
from sentiment_based_filtering import SentimentBasedRecommender


class HybridRecommender:
    def __init__(
        self,
        weights={"cf": 0.4, "cbf": 0.4, "sentiment": 0.2},
        model_paths={
            "cf": "model/collab_model.pkl",
            "cbf": "model/content_model.pkl",
            "sentiment": "model/sentiment_model.pkl",
        },
    ):
        self.cf_model = CollaborativeFilteringRecommender(model_path=model_paths["cf"])
        self.cb_model = ContentBasedRecommender(model_path=model_paths["cbf"])
        self.sb_model = SentimentBasedRecommender(model_path=model_paths["sentiment"])
        self.weights = weights

        # Load all models
        self.cf_model.load_model()
        self.cb_model.load_model()
        self.sb_model.load_model()

    def get_combined_recommendations(self, user_id, user_df, product_df, top_n=5):
        # === Step 1: CF Recommendations ===
        try:
            cf_recs = self.cf_model.recommend_top_n(user_df, user_id, n=50)
        except Exception:
            cf_recs = []

        cf_scores = {r["product_id"]: r["predicted_rating"] for r in cf_recs}

        # === Step 2: CBF for those CF items (or fallback if CF empty) ===
        candidate_products = list(cf_scores.keys())
        if not candidate_products:
            candidate_products = product_df["product_id"].tolist()

        cbf_scores = {}
        for pid in candidate_products:
            try:
                similar_products = self.cb_model.get_similar_products(pid, n=1)
                cbf_scores[pid] = similar_products[0]["similarity_score"]
            except Exception:
                cbf_scores[pid] = 0.0

        # === Step 3: Sentiment Score ===
        sentiment_df = self.sb_model.product_sentiments
        sentiment_scores = sentiment_df.set_index("product_id")["avg_sentiment_score"].to_dict()

        # === Step 4: Combine Scores ===
        final_scores = {}
        for pid in candidate_products:
            cf = cf_scores.get(pid, 0)
            cbf = cbf_scores.get(pid, 0)
            sent = sentiment_scores.get(pid, 0)

            final_score = (
                self.weights["cf"] * cf
                + self.weights["cbf"] * cbf
                + self.weights["sentiment"] * sent
            )
            final_scores[pid] = round(final_score, 4)

        top_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [{"product_id": pid, "hybrid_score": score} for pid, score in top_products]


# For testing
if __name__ == "__main__":
    user_df = pd.read_csv("data/user_ratings.csv")     # Must contain: user_id, product_id, rating
    product_df = pd.read_csv("data/products.csv")      # Must contain: product_id
    user_id = user_df["user_id"].iloc[0]

    hybrid = HybridRecommender()
    recs = hybrid.get_combined_recommendations(user_id, user_df, product_df, top_n=5)

    print(f"\nTop 5 Hybrid Recommendations for User {user_id}:")
    for rec in recs:
        print(rec)
