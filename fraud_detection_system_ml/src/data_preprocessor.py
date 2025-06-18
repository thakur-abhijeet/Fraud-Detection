# combined_preprocessor.py

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    def __init__(self):
        self.numeric_features = [
            "amount",
            "time_since_last_transaction",
            "prev_timestamp_hour",
            "prev_timestamp_day",
            "prev_timestamp_weekday",
        ]
        self.categorical_features = ["payment_method", "device_id"]

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

    def basic_cleaning(self, df):
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["prev_timestamp_hour"] = df["timestamp"].dt.hour
            df["prev_timestamp_day"] = df["timestamp"].dt.day
            df["prev_timestamp_weekday"] = df["timestamp"].dt.weekday
            df.drop(columns=["timestamp"], inplace=True)
        else:
            df["prev_timestamp_hour"] = 0
            df["prev_timestamp_day"] = 0
            df["prev_timestamp_weekday"] = 0
            # Drop unused columns (ignore if missing)
            df.drop(
                columns=["shipping_address", "ip_address"],
                errors="ignore",
                inplace=True,
            )

        return df

    # df = df.copy()
    # df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # df["prev_timestamp_hour"] = df["timestamp"].dt.hour
    # df["prev_timestamp_day"] = df["timestamp"].dt.day
    # df["prev_timestamp_weekday"] = df["timestamp"].dt.weekday
    # df.drop(columns=["timestamp"], inplace=True)
    # df.drop(
    #   columns=["shipping_address", "ip_address"], errors="ignore", inplace=True
    # )
    # return df

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
