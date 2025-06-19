
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

class ContentBasedRecommender:
    def __init__(self, model_path="model/content_model.pkl"):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=1000, ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.product_ids = []
        self.model_path = model_path
        self.df = None

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = " ".join(
            word for word in text.split() if word not in set(stopwords.words("english"))
        )
        return text

    def combine_features(self, row):
        return f"{row['title']} {row['category']} {row['attributes']}"

    def preprocess(self, df):
        df = df.copy()
        df["title"] = df["title"].fillna("")
        df["category"] = df["category"].fillna("")
        df["attributes"] = df["attributes"].fillna("")
        df["combined_text"] = df.apply(self.combine_features, axis=1)
        df["combined_text"] = df["combined_text"].apply(self.clean_text)
        return df

    def train(self, df, save_model=True):
        self.df = self.preprocess(df)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])
        self.product_ids = self.df["product_id"].tolist()
        if save_model:
            self.save_model()

    def save_model(self):
        joblib.dump((self.vectorizer, self.tfidf_matrix, self.product_ids, self.df), self.model_path)

    def load_model(self):
        self.vectorizer, self.tfidf_matrix, self.product_ids, self.df = joblib.load(self.model_path)

    def get_similar_products(self, product_id, n=5):
        if self.tfidf_matrix is None:
            raise ValueError("Model not trained or loaded.")

        try:
            idx = self.product_ids.index(product_id)
        except ValueError:
            raise ValueError(f"Product ID {product_id} not found.")

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()
        similar_indices = similarity_scores.argsort()[::-1][1 : n + 1]

        similar_products = [
            {
                "product_id": self.product_ids[i],
                "similarity_score": round(similarity_scores[i], 3),
            }
            for i in similar_indices
        ]
        return similar_products


# For testing / direct run
if __name__ == "__main__":
    df = pd.read_csv("data/products.csv")  # Requires: product_id, title, category, attributes

    recommender = ContentBasedRecommender()
    recommender.train(df)

    sample_product_id = df["product_id"].iloc[0]
    print(f"Top recommendations for product: {sample_product_id}")
    for rec in recommender.get_similar_products(sample_product_id):
        print(rec)
