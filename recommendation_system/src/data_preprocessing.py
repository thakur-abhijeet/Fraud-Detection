# Data Preprocessing Module for E-commerce Recommendation System
# This module provides utilities for loading, cleaning, and preprocessing data
# for content-based, collaborative, and sentiment-based filtering approaches.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os

class DataPreprocessor:
    def __init__(self, data_dir='/home/masubhaat/ML/recommendation_system/data/'):
        """
        Initialize the DataPreprocessor.

        Parameters:
        -----------
        data_dir : str
            Directory containing the data files.
        """
        self.data_dir = data_dir
        self.products_df = None
        self.users_df = None
        self.interactions_df = None
        self.reviews_df = None

        # Initialize NLP tools
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except (LookupError, OSError) as e:
            print(f"Warning: NLTK resources not available. Text preprocessing will be limited. Error: {e}")
            self.stop_words = set()
            self.stemmer = None
            self.lemmatizer = None

    def load_data(self, products_file=None, users_file=None, interactions_file=None, reviews_file=None):
        """
        Load data from CSV or JSON files.

        Parameters:
        -----------
        products_file : str, optional
            Filename for products data.
        users_file : str, optional
            Filename for users data.
        interactions_file : str, optional
            Filename for user-item interactions data.
        reviews_file : str, optional
            Filename for product reviews data.

        Returns:
        --------
        self : DataPreprocessor
            Returns self for method chaining.
        """
        if products_file:
            file_path = os.path.join(self.data_dir, products_file)
            if os.path.exists(file_path):
                if file_path.endswith('.csv'):
                    self.products_df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    self.products_df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file format for {products_file}")
            else:
                print(f"Warning: {products_file} not found in {self.data_dir}")

        if users_file:
            file_path = os.path.join(self.data_dir, users_file)
            if os.path.exists(file_path):
                if file_path.endswith('.csv'):
                    self.users_df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    self.users_df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file format for {users_file}")
            else:
                print(f"Warning: {users_file} not found in {self.data_dir}")

        if interactions_file:
            file_path = os.path.join(self.data_dir, interactions_file)
            if os.path.exists(file_path):
                if file_path.endswith('.csv'):
                    self.interactions_df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    self.interactions_df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file format for {interactions_file}")
            else:
                print(f"Warning: {interactions_file} not found in {self.data_dir}")

        if reviews_file:
            file_path = os.path.join(self.data_dir, reviews_file)
            if os.path.exists(file_path):
                if file_path.endswith('.csv'):
                    self.reviews_df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    self.reviews_df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file format for {reviews_file}")
            else:
                print(f"Warning: {reviews_file} not found in {self.data_dir}")

        return self

    def clean_data(self, handle_missing=True, remove_duplicates=True):

        # Clean the loaded data by handling missing values and removing duplicates.
        # Returns:
        # --------
        # self : DataPreprocessor
        #     Returns self for method chaining

        if self.products_df is not None:
            if handle_missing:
                num_cols = self.products_df.select_dtypes(include=['number']).columns
                for col in num_cols:
                    self.products_df[col].fillna(self.products_df[col].mean(), inplace=True)
                cat_cols = self.products_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.products_df[col].fillna('Unknown', inplace=True)
            if remove_duplicates and 'product_id' in self.products_df.columns:
                self.products_df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)
        
        if self.users_df is not None:
            if handle_missing:
                cat_cols = self.users_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.users_df[col].fillna('Unknown', inplace=True)
            if remove_duplicates and 'user_id' in self.users_df.columns:
                self.users_df.drop_duplicates(subset=['user_id'], keep='first', inplace=True)
        
        if self.interactions_df is not None:
            if handle_missing and 'rating' in self.interactions_df.columns:
                self.interactions_df['rating'].fillna(self.interactions_df['rating'].median(), inplace=True)
            if remove_duplicates and ('user_id' in self.interactions_df.columns and 'product_id' in self.interactions_df.columns):
                self.interactions_df.drop_duplicates(subset=['user_id', 'product_id'], keep='last', inplace=True)
        
        if self.reviews_df is not None:
            if handle_missing:
                if 'review_text' in self.reviews_df.columns:
                    self.reviews_df['review_text'].fillna('', inplace=True)
                if 'rating' in self.reviews_df.columns:
                    self.reviews_df['rating'].fillna(self.reviews_df['rating'].median(), inplace=True)
            if remove_duplicates:
                id_cols = [col for col in ['user_id', 'product_id', 'timestamp'] if col in self.reviews_df.columns]
                if id_cols:
                    self.reviews_df.drop_duplicates(subset=id_cols, keep='last', inplace=True)
        
        return self
    
    def preprocess_text(self, text):

        # Preprocess text data by removing special characters, stopwords, and applying stemming/lemmatization.
        # Parameters:
        # -----------
        # text : str
        #     Text to preprocess
        # Returns:
        # --------
        # str
        #     Preprocessed text

        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_content_features(self, text_columns=None, categorical_columns=None, numerical_columns=None):

        # Extract features for content-based filtering.
        # Returns:
        # --------
        # pd.DataFrame
        #     DataFrame with extracted features

        if self.products_df is None:
            raise ValueError("Products data not loaded. Call load_data() first.")
        
        features_df = self.products_df.copy()
        
        if text_columns:
            for col in text_columns:
                if col in features_df.columns:
                    features_df[f'{col}_processed'] = features_df[col].apply(self.preprocess_text)
            for col in text_columns:
                processed_col = f'{col}_processed'
                if processed_col in features_df.columns:
                    vectorizer = TfidfVectorizer(max_features=100)
                    try:
                        tfidf_matrix = vectorizer.fit_transform(features_df[processed_col].fillna(''))
                        tfidf_df = pd.DataFrame(
                            tfidf_matrix.toarray(),
                            columns=[f'{col}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                        )
                        if 'product_id' in features_df.columns:
                            tfidf_df['product_id'] = features_df['product_id'].values
                            features_df = features_df.merge(tfidf_df, on='product_id', how='left')
                    except:
                        print(f"Warning: Could not create TF-IDF features for {col}")
        
        if categorical_columns:
            for col in categorical_columns:
                if col in features_df.columns:
                    top_categories = features_df[col].value_counts().nlargest(10).index
                    for category in top_categories:
                        features_df[f'{col}_{category}'] = (features_df[col] == category).astype(int)
        
        if numerical_columns:
            scaler = MinMaxScaler()
            for col in numerical_columns:
                if col in features_df.columns:
                    features_df[f'{col}_scaled'] = scaler.fit_transform(
                        features_df[[col]].fillna(features_df[col].mean())
                    )
        return features_df
    
    def prepare_collaborative_data(self, min_interactions=5):

        # Prepare data for collaborative filtering.
        # Returns:
        # --------
        # pd.DataFrame
        #     DataFrame with user-item interactions

        if self.interactions_df is None:
            raise ValueError("Interactions data not loaded. Call load_data() first.")
        
        cf_df = self.interactions_df.copy()
        required_cols = ['user_id', 'product_id']
        if not all(col in cf_df.columns for col in required_cols):
            raise ValueError(f"Interactions data must contain columns: {required_cols}")
        
        if 'rating' not in cf_df.columns:
            cf_df['rating'] = 1.0
        
        if min_interactions > 1:
            user_counts = cf_df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            cf_df = cf_df[cf_df['user_id'].isin(valid_users)]
        
        user_ids = cf_df['user_id'].unique()
        product_ids = cf_df['product_id'].unique()
        user_id_map = {id: i for i, id in enumerate(user_ids)}
        product_id_map = {id: i for i, id in enumerate(product_ids)}
        cf_df['user_idx'] = cf_df['user_id'].map(user_id_map)
        cf_df['product_idx'] = cf_df['product_id'].map(product_id_map)
        
        self.user_id_map = user_id_map
        self.product_id_map = product_id_map
        self.reverse_user_id_map = {v: k for k, v in user_id_map.items()}
        self.reverse_product_id_map = {v: k for k, v in product_id_map.items()}
        
        return cf_df
    
    def prepare_sentiment_data(self):

        # Prepare data for sentiment-based filtering.
        # Returns:
        # --------
        # pd.DataFrame
        #     DataFrame with product reviews and preprocessed text

        if self.reviews_df is None:
            raise ValueError("Reviews data not loaded. Call load_data() first.")
        
        sentiment_df = self.reviews_df.copy()
        if 'review_text' not in sentiment_df.columns:
            raise ValueError("Reviews data must contain 'review_text' column")
        
        sentiment_df['processed_text'] = sentiment_df['review_text'].apply(self.preprocess_text)
        
        return sentiment_df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        Returns:
        --------
        tuple
            (train_df, test_df) DataFrames
        """
        return train_test_split(df, test_size=test_size, random_state=random_state)
    
    def get_user_item_matrix(self, interactions_df=None):
        """
        Create a user-item matrix from interactions data.
        Returns:
        --------
        tuple
            (user_item_matrix, user_indices, item_indices)
        """
        if interactions_df is None:
            if self.interactions_df is None:
                raise ValueError("Interactions data not loaded. Call load_data() first.")
            interactions_df = self.prepare_collaborative_data()
        
        user_indices = interactions_df['user_idx'].values
        item_indices = interactions_df['product_idx'].values
        ratings = interactions_df['rating'].values
        
        n_users = len(np.unique(user_indices))
        n_items = len(np.unique(item_indices))
        
        user_item_matrix = np.zeros((n_users, n_items))
        for i in range(len(ratings)):
            user_item_matrix[user_indices[i], item_indices[i]] = ratings[i]
        
        return user_item_matrix, user_indices, item_indices

    # --- Sample Data Generation Methods ---
    def generate_sample_product_data(self):
        """Generate sample product data as a DataFrame."""
        df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['Product A', 'Product B', 'Product C'],
            'description': ['A description for product A', 
                            'A description for product B', 
                            'A description for product C']
        })
        self.products_df = df
        return df

    def generate_sample_rating_data(self, products_df):
        """Generate sample interactions (ratings) data based on products_df."""
        df = pd.DataFrame({
            'user_id': [101, 102, 101, 103],
            'product_id': [1, 2, 3, 1],
            'rating': [5, 4, 5, 3]
        })
        self.interactions_df = df
        return df

    def generate_sample_review_data(self, products_df):
        """Generate sample reviews data based on products_df."""
        df = pd.DataFrame({
            'user_id': [101, 102],
            'product_id': [1, 2],
            'review_text': ['Great product!', 'Could be better.'],
            'rating': [5, 3],
            'timestamp': ['2023-01-01', '2023-01-02']
        })
        self.reviews_df = df
        return df

    def generate_sample_user_data(self):
        """Generate sample user data."""
        df = pd.DataFrame({
            'user_id': [101, 102, 103],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        self.users_df = df
        return df
