
import pandas as pd
from hybrid_recommender import HybridRecommender

class RecommenderAPI:
    def __init__(self):
        self.ratings_df = pd.read_csv("data/user_ratings.csv")
        self.hybrid = HybridRecommender()

    def recommend_for_user(self, user_id: str, top_n: int = 5):
        return self.hybrid.recommend(user_id, self.ratings_df, n=top_n)
