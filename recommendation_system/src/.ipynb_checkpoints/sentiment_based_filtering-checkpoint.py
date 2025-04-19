"""
Sentiment-Based Filtering Module for E-commerce Recommendation System

This module implements sentiment analysis on product reviews and uses
the sentiment scores to enhance product recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pickle
import os
import re

class SentimentBasedRecommender:
    """
    A class for sentiment-based filtering recommendations.
    
    This class analyzes product reviews to determine sentiment scores
    and uses these scores to enhance product recommendations.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the SentimentBasedRecommender.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.sentiment_model = None
        self.product_sentiment_scores = None
        self.vectorizer = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize NLTK resources
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.vader = SentimentIntensityAnalyzer()
        except:
            print("Warning: NLTK Vader not available. Using basic sentiment analysis.")
            self.vader = None
    
    def preprocess_text(self, text):
        """
        Preprocess text data for sentiment analysis.
        
        Parameters:
        -----------
        text : str
            Text to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fit_vader_sentiment(self, reviews_df, review_text_col='review_text', 
                           product_id_col='product_id', rating_col=None):
        """
        Calculate sentiment scores using VADER lexicon-based approach.
        
        Parameters:
        -----------
        reviews_df : pd.DataFrame
            DataFrame containing product reviews
        review_text_col : str
            Name of the column containing review text
        product_id_col : str
            Name of the column containing product IDs
        rating_col : str
            Name of the column containing ratings (optional)
            
        Returns:
        --------
        self : SentimentBasedRecommender
            Returns self for method chaining
        """
        if self.vader is None:
            raise ValueError("VADER sentiment analyzer not available")
        
        # Verify required columns exist
        required_cols = [review_text_col, product_id_col]
        missing_cols = [col for col in required_cols if col not in reviews_df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")
        
        # Preprocess review text
        reviews_df['processed_text'] = reviews_df[review_text_col].apply(self.preprocess_text)
        
        # Calculate sentiment scores for each review
        reviews_df['sentiment_score'] = reviews_df['processed_text'].apply(
            lambda x: self.vader.polarity_scores(x)['compound']
        )
        
        # If rating column exists, combine with sentiment score
        if rating_col and rating_col in reviews_df.columns:
            # Normalize ratings to [0, 1] range
            max_rating = reviews_df[rating_col].max()
            min_rating = reviews_df[rating_col].min()
            
            if max_rating > min_rating:
                reviews_df['normalized_rating'] = (reviews_df[rating_col] - min_rating) / (max_rating - min_rating)
                
                # Combine normalized rating and sentiment score (equal weights)
                reviews_df['combined_score'] = 0.5 * reviews_df['normalized_rating'] + 0.5 * (reviews_df['sentiment_score'] + 1) / 2
            else:
                reviews_df['combined_score'] = (reviews_df['sentiment_score'] + 1) / 2
        else:
            # Convert sentiment score from [-1, 1] to [0, 1]
            reviews_df['combined_score'] = (reviews_df['sentiment_score'] + 1) / 2
        
        # Calculate average sentiment score for each product
        product_sentiment = reviews_df.groupby(product_id_col).agg({
            'combined_score': 'mean',
            'sentiment_score': 'mean',
            product_id_col: 'count'
        }).rename(columns={product_id_col: 'review_count'})
        
        # Store product sentiment scores
        self.product_sentiment_scores = product_sentiment
        
        return self
    
    def fit_ml_sentiment(self, reviews_df, review_text_col='review_text', 
                        product_id_col='product_id', rating_col=None, 
                        classifier='logistic', test_size=0.2):
        """
        Train a machine learning model for sentiment analysis.
        
        Parameters:
        -----------
        reviews_df : pd.DataFrame
            DataFrame containing product reviews
        review_text_col : str
            Name of the column containing review text
        product_id_col : str
            Name of the column containing product IDs
        rating_col : str
            Name of the column containing ratings (used as labels if provided)
        classifier : str
            Type of classifier to use ('logistic' or 'random_forest')
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        self : SentimentBasedRecommender
            Returns self for method chaining
        """
        # Verify required columns exist
        required_cols = [review_text_col, product_id_col]
        missing_cols = [col for col in required_cols if col not in reviews_df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")
        
        # Preprocess review text
        reviews_df['processed_text'] = reviews_df[review_text_col].apply(self.preprocess_text)
        
        # Create labels for sentiment classification
        if rating_col and rating_col in reviews_df.columns:
            # Use ratings as labels (assuming higher rating = positive sentiment)
            max_rating = reviews_df[rating_col].max()
            min_rating = reviews_df[rating_col].min()
            mid_rating = (max_rating + min_rating) / 2
            
            reviews_df['sentiment_label'] = (reviews_df[rating_col] > mid_rating).astype(int)
        else:
            # Use VADER to create labels
            if self.vader is not None:
                reviews_df['sentiment_score'] = reviews_df['processed_text'].apply(
                    lambda x: self.vader.polarity_scores(x)['compound']
                )
                reviews_df['sentiment_label'] = (reviews_df['sentiment_score'] > 0).astype(int)
            else:
                raise ValueError("Neither rating column nor VADER sentiment analyzer available for creating labels")
        
        # Create feature vectors using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
        X = self.vectorizer.fit_transform(reviews_df['processed_text'])
        y = reviews_df['sentiment_label']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train the classifier
        if classifier == 'logistic':
            self.sentiment_model = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier == 'random_forest':
            self.sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier}. Choose from 'logistic' or 'random_forest'")
        
        self.sentiment_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.sentiment_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Sentiment model accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Calculate sentiment probability for each review
        sentiment_proba = self.sentiment_model.predict_proba(X)[:, 1]
        reviews_df['sentiment_score'] = sentiment_proba
        
        # Calculate average sentiment score for each product
        product_sentiment = reviews_df.groupby(product_id_col).agg({
            'sentiment_score': 'mean',
            product_id_col: 'count'
        }).rename(columns={product_id_col: 'review_count'})
        
        # Store product sentiment scores
        self.product_sentiment_scores = product_sentiment
        
        return self
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        float
            Sentiment score (0-1 range, higher = more positive)
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        if self.sentiment_model is not None and self.vectorizer is not None:
            # Use trained ML model
            X = self.vectorizer.transform([processed_text])
            sentiment_proba = self.sentiment_model.predict_proba(X)[0, 1]
            return sentiment_proba
        elif self.vader is not None:
            # Use VADER
            sentiment_score = self.vader.polarity_scores(processed_text)['compound']
            # Convert from [-1, 1] to [0, 1]
            return (sentiment_score + 1) / 2
        else:
            raise ValueError("No sentiment analysis model available")
    
    def get_product_sentiment(self, product_id):
        """
        Get sentiment score for a specific product.
        
        Parameters:
        -----------
        product_id : str or int
            ID of the product
            
        Returns:
        --------
        float
            Sentiment score (0-1 range, higher = more positive)
        """
        if self.product_sentiment_scores is None:
            raise ValueError("Product sentiment scores not calculated. Call fit_vader_sentiment() or fit_ml_sentiment() first.")
        
        if product_id in self.product_sentiment_scores.index:
            return self.product_sentiment_scores.loc[product_id, 'sentiment_score']
        else:
            # Return neutral sentiment if product not found
            return 0.5
    
    def adjust_recommendations(self, recommendations, sentiment_weight=0.3):
        """
        Adjust recommendation scores based on sentiment.
        
        Parameters:
        -----------
        recommendations : list
            List of tuples (product_id, score) from another recommender
        sentiment_weight : float
            Weight to give to sentiment scores (0-1)
            
        Returns:
        --------
        list
            List of tuples (product_id, adjusted_score) with sentiment-adjusted scores
        """
        if self.product_sentiment_scores is None:
            raise ValueError("Product sentiment scores not calculated. Call fit_vader_sentiment() or fit_ml_sentiment() first.")
        
        adjusted_recommendations = []
        
        for product_id, score in recommendations:
            # Get sentiment score for this product
            if product_id in self.product_sentiment_scores.index:
                sentiment_score = self.product_sentiment_scores.loc[product_id, 'sentiment_score']
                
                # Adjust the recommendation score
                adjusted_score = (1 - sentiment_weight) * score + sentiment_weight * sentiment_score
                
                adjusted_recommendations.append((product_id, adjusted_score))
            else:
                # If no sentiment data, keep original score
                adjusted_recommendations.append((product_id, score))
        
        # Sort by adjusted score
        adjusted_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return adjusted_recommendations
    
    def recommend_products(self, n=10, min_reviews=5):
        """
        Recommend products based solely on sentiment scores.
        
        Parameters:
        -----------
        n : int
            Number of recommendations to return
        min_reviews : int
            Minimum number of reviews required for a product to be recommended
            
        Returns:
        --------
        list
            List of tuples (product_id, sentiment_score) for recommended products
        """
        if self.product_sentiment_scores is None:
            raise ValueError("Product sentiment scores not calculated. Call fit_vader_sentiment() or fit_ml_sentiment() first.")
        
        # Filter products with enough reviews
        qualified_products = self.product_sentiment_scores[self.product_sentiment_scores['review_count'] >= min_reviews]
        
        # Sort by sentiment score
        top_products = qualified_products.sort_values('sentiment_score', ascending=False)
        
        # Get top N products
        recommendations = [(idx, row['sentiment_score']) for idx, row in top_products.iloc[:n].iterrows()]
        
        return recommendations
    
    def save_model(self, filename='sentiment_model.pkl'):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'product_sentiment_scores': self.product_sentiment_scores,
            'sentiment_model': self.sentiment_model,
            'vectorizer': self.vectorizer
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='sentiment_model.pkl'):
        """
        Load the model from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
            
        Returns:
        --------
        self : SentimentBasedRecommender
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.product_sentiment_scores = model_data['product_sentiment_scores']
        self.sentiment_model = model_data['sentiment_model']
        self.vectorizer = model_data['vectorizer']
        
        return self
    
    def evaluate(self, test_reviews, review_text_col='review_text', rating_col='rating'):
        """
        Evaluate the sentiment analysis model on test data.
        
        Parameters:
        -----------
        test_reviews : pd.DataFrame
            DataFrame containing test reviews
        review_text_col : str
            Name of the column containing review text
        rating_col : str
            Name of the column containing ratings
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Verify required columns exist
        required_cols = [review_text_col]
        if rating_col:
            required_cols.append(rating_col)
            
        missing_cols = [col for col in required_cols if col not in test_reviews.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")
        
        # Preprocess review text
        test_reviews['processed_text'] = test_reviews[review_text_col].apply(self.preprocess_text)
        
        # Calculate predicted sentiment scores
        if self.sentiment_model is not None and self.vectorizer is not None:
            # Use trained ML model
            X = self.vectorizer.transform(test_reviews['processed_text'])
            test_reviews['predicted_sentiment'] = self.sentiment_model.predict_proba(X)[:, 1]
        elif self.vader is not None:
            # Use VADER
            test_reviews['predicted_sentiment'] = test_reviews['processed_text'].apply(
                lambda x: (self.vader.polarity_scores(x)['compound'] + 1) / 2
            )
        else:
            raise ValueError("No sentiment analysis model available")
        
        # If rating column exists, calculate correlation with predicted sentiment
        if rating_col and rating_col in test_reviews.columns:
            # Normalize ratings to [0, 1] range
            max_rating = test_reviews[rating_col].max()
            min_rating = test_reviews[rating_col].min()
            
            if max_rating > min_rating:
                test_reviews['normalized_rating'] = (test_reviews[rating_col] - min_rating) / (max_rating - min_rating)
                
                # Calculate correlation
                correlation = test_reviews['predicted_sentiment'].corr(test_reviews['normalized_rating'])
                
                # Calculate binary classification metrics
                mid_rating = (max_rating + min_rating) / 2
                test_reviews['actual_positive'] = (test_reviews[rating_col] > mid_rating).astype(int)
                test_reviews['predicted_positive'] = (test_reviews['predicted_sentiment'] > 0.5).astype(int)
                
                # Calculate accuracy
                accuracy = (test_reviews['actual_positive'] == test_reviews['predicted_positive']).mean()
                
                # Calculate precision and recall
                true_positives = ((test_reviews['actual_positive'] == 1) & (test_reviews['predicted_positive'] == 1)).sum()
                false_positives = ((test_reviews['actual_positive'] == 0) & (test_reviews['predicted_positive'] == 1)).sum()
                false_negatives = ((test_reviews['actual_positive'] == 1) & (test_reviews['predicted_positive'] == 0)).sum()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                # Calculate F1 score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                return {
                    'correlation': correlation,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
            
        # If no rating column or all ratings are the same, return basic metrics
        return {
            'mean_sentiment': test_reviews['predicted_sentiment'].mean(),
            'std_sentiment': test_reviews['predicted_sentiment'].std()
        }
