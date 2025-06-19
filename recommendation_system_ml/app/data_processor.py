import pandas as pd
import joblib
import nltk
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

nltk.download("stopwords")
from nltk.corpus import stopwords

class ProductDataPreprocessor:
    def __init__(self):
        self.numeric_features = ["price", "rating"]
        self.categorical_features = ["brand", "category"]
        self.text_features = "attributes"

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        text_transformer = TfidfVectorizer(
            stop_words="english", max_features=100, ngram_range=(1, 2)
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
                ("text", text_transformer, self.text_features),
            ]
        )

    def basic_cleaning(self, df):
        df = df.copy()
        df["attributes"] = df["attributes"].fillna("").apply(self.clean_text)
        df["brand"] = df["brand"].fillna("Unknown")
        df["category"] = df["category"].fillna("Other")
        df["price"] = df["price"].fillna(df["price"].median())
        df["rating"] = df["rating"].fillna(df["rating"].mean())
        return df

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = " ".join(
            word
            for word in text.split()
            if word not in set(stopwords.words("english"))
        )
        return text

    def fit(self, df):
        df_clean = self.basic_cleaning(df)
        self.pipeline.fit(df_clean)
        return self

    def transform(self, df):
        df_clean = self.basic_cleaning(df)
        return self.pipeline.transform(df_clean)

    def fit_transform(self, df):
        df_clean = self.basic_cleaning(df)
        return self.pipeline.fit_transform(df_clean)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
