
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate


class CollaborativeFilteringRecommender:
    def __init__(self, model_path="model/collab_model.pkl"):
        self.model = None
        self.model_path = model_path
        self.reader = Reader(rating_scale=(0, 5))

    def prepare_data(self, df: pd.DataFrame):
        """
        Convert DataFrame to Surprise dataset format.
        Required columns: user_id, product_id, rating
        """
        if not set(["user_id", "product_id", "rating"]).issubset(df.columns):
            raise ValueError("DataFrame must contain user_id, product_id, and rating columns.")
        return Dataset.load_from_df(df[["user_id", "product_id", "rating"]], self.reader)

    def train(self, df: pd.DataFrame, save_model=True):
        """
        Train the SVD model using Surprise.
        """
        data = self.prepare_data(df)
        trainset, testset = train_test_split(data, test_size=0.2)
        self.model = SVD()
        self.model.fit(trainset)

        # Optional evaluation
        results = cross_validate(self.model, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

        if save_model:
            self.save_model()

        return results

    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def predict_rating(self, user_id, product_id):
        """
        Predict rating for a given user-product pair.
        """
        return self.model.predict(user_id, product_id).est

    def recommend_top_n(self, df, user_id, n=5):
        """
        Recommend top N products to the user not already rated by them.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train().")

        all_products = df["product_id"].unique()
        user_products = df[df["user_id"] == user_id]["product_id"].unique()
        unrated_products = list(set(all_products) - set(user_products))

        predictions = [
            (pid, self.predict_rating(user_id, pid)) for pid in unrated_products
        ]
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        return [{"product_id": pid, "predicted_rating": round(score, 2)} for pid, score in top_n]


# For standalone execution or testing
if __name__ == "__main__":
    # Example
    df = pd.read_csv("data/user_ratings.csv")  # Ensure it contains: user_id, product_id, rating

    recommender = CollaborativeFilteringRecommender()
    metrics = recommender.train(df)

    print("Evaluation Metrics:", metrics)

    # Example recommendation
    user_id = "U1234"
    top_products = recommender.recommend_top_n(df, user_id, n=5)
    print(f"\nTop 5 product recommendations for User {user_id}:")
    for rec in top_products:
        print(rec)
