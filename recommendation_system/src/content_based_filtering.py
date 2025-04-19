"""
Content-Based Filtering Module for E-commerce Recommendation System
This module implements content-based filtering for product recommendations
based on product features and attributes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from typing import List, Tuple, Dict, Optional, Union
# from src.utils.logging_config import logger

class ContentBasedRecommender:
    """
    A class for content-based filtering recommendations.
    This class uses product features to calculate similarity between products
    and recommend similar items to users based on their past interactions.
    """
    
    def __init__(self, model_dir: str = '../model'):
        """
        Initialize the ContentBasedRecommender.
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.product_features = None
        self.similarity_matrix = None
        self.product_id_map = None
        self.reverse_product_id_map = None
        self.feature_columns = None
        self.product_ids = None
        self.vectorizer = None
        self.min_products = 3  # Minimum number of products for recommendations
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def fit(self, product_features_df: pd.DataFrame, feature_columns: Optional[List[str]] = None, 
            id_column: str = 'product_id') -> 'ContentBasedRecommender':
        """
        Fit the content-based recommender with product features.
        
        Parameters:
        -----------
        product_features_df : pd.DataFrame
            DataFrame containing product features
        feature_columns : list, optional
            List of column names to use as features
        id_column : str
            Name of the column containing product IDs
            
        Returns:
        --------
        self : ContentBasedRecommender
            Returns self for method chaining
        """
        try:
            if id_column not in product_features_df.columns:
                raise ValueError(f"ID column '{id_column}' not found in product features DataFrame")

            if len(product_features_df) < self.min_products:
                logger.warning(f"Too few products ({len(product_features_df)}) for content-based filtering. "
                              f"Minimum required: {self.min_products}")
                return self

            if feature_columns is None:
                self.feature_columns = [col for col in product_features_df.columns if col != id_column]
            else:
                missing_cols = [col for col in feature_columns if col not in product_features_df.columns]
                if missing_cols:
                    logger.warning(f"Some feature columns not found: {missing_cols}. Using available features.")
                    self.feature_columns = [col for col in feature_columns if col in product_features_df.columns]
                else:
                    self.feature_columns = feature_columns

            if not self.feature_columns:
                logger.warning("No valid feature columns found. Using ID column as fallback.")
                self.feature_columns = [id_column]

            product_ids = product_features_df[id_column].unique()
            self.product_id_map = {id: i for i, id in enumerate(product_ids)}
            self.reverse_product_id_map = {i: id for id, i in self.product_id_map.items()}
            self.product_ids = product_features_df[id_column].values

            # Check if features are numeric
            numeric_columns = product_features_df[self.feature_columns].select_dtypes(include=['number']).columns
            
            if len(numeric_columns) == len(self.feature_columns):
                # All numeric
                self.product_features = product_features_df[self.feature_columns].fillna(0)
                self.similarity_matrix = cosine_similarity(self.product_features)
            else:
                # Non-numeric (text or mixed), combine into single text column
                try:
                    combined_features = product_features_df[self.feature_columns].astype(str).apply(
                        lambda x: ' '.join(x), axis=1)
                    
                    self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                    self.product_features = self.vectorizer.fit_transform(combined_features)
                    self.similarity_matrix = cosine_similarity(self.product_features)
                except Exception as e:
                    logger.error(f"Error in text feature processing: {str(e)}")
                    # Fallback to simple numeric features if available
                    if numeric_columns.any():
                        logger.info("Falling back to numeric features")
                        self.product_features = product_features_df[numeric_columns].fillna(0)
                        self.similarity_matrix = cosine_similarity(self.product_features)
                    else:
                        # Last resort: use binary features
                        logger.info("Using binary features as last resort")
                        self.product_features = pd.get_dummies(
                            product_features_df[self.feature_columns].astype(str)
                        )
                        self.similarity_matrix = cosine_similarity(self.product_features)

            return self
            
        except Exception as e:
            logger.error(f"Error in fit method: {str(e)}")
            raise

    def fit_from_text(self, product_df: pd.DataFrame, text_column: str, 
                     id_column: str = 'product_id', max_features: int = 5000) -> 'ContentBasedRecommender':
        """
        Fit the content-based recommender using text features.
        Parameters:
        -----------
        product_df : pd.DataFrame
            DataFrame containing product information
        text_column : str
            Name of the column containing text to use for recommendations
        id_column : str
            Name of the column containing product IDs
        max_features : int
            Maximum number of features to extract from text
        Returns:
        --------
        self : ContentBasedRecommender
            Returns self for method chaining
        """
        try:
            if id_column not in product_df.columns:
                raise ValueError(f"ID column '{id_column}' not found in product DataFrame")
            
            if text_column not in product_df.columns:
                raise ValueError(f"Text column '{text_column}' not found in product DataFrame")
            
            if len(product_df) < self.min_products:
                logger.warning(f"Too few products ({len(product_df)}) for content-based filtering. "
                              f"Minimum required: {self.min_products}")
                return self
            
            product_ids = product_df[id_column].unique()
            self.product_id_map = {id: i for i, id in enumerate(product_ids)}
            self.reverse_product_id_map = {i: id for id, i in self.product_id_map.items()}
            
            # Handle missing text values
            text_data = product_df[text_column].fillna('')
            
            try:
                self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
                text_features = self.vectorizer.fit_transform(text_data)
                self.similarity_matrix = cosine_similarity(text_features)
            except Exception as e:
                logger.error(f"Error in text vectorization: {str(e)}")
                # Fallback to simple binary features
                logger.info("Falling back to binary features")
                self.product_features = pd.get_dummies(product_df[[text_column]].astype(str))
                self.similarity_matrix = cosine_similarity(self.product_features)
            
            self.product_ids = product_df[id_column].values
            
            return self
            
        except Exception as e:
            logger.error(f"Error in fit_from_text method: {str(e)}")
            raise
    
    def get_similar_products(self, product_id: Union[str, int], n: int = 10, 
                           exclude_same: bool = True) -> List[Tuple[str, float]]:
        """
        Get similar products based on content similarity.
        Parameters:
        -----------
        product_id : str or int
            ID of the product to find similar items for
        n : int
            Number of similar products to return
        exclude_same : bool
            Whether to exclude the input product from results
        Returns:
        --------
        list
            List of tuples (product_id, similarity_score) for similar products
        """
        try:
            if product_id not in self.product_id_map:
                logger.warning(f"Product ID {product_id} not found in the model")
                return []
            
            product_idx = self.product_id_map[product_id]
            similarity_scores = self.similarity_matrix[product_idx]
            similar_indices = np.argsort(similarity_scores)[::-1]
            
            if exclude_same:
                similar_indices = similar_indices[similar_indices != product_idx]
            
            top_n_indices = similar_indices[:n]
            similar_products = [
                (self.reverse_product_id_map[idx], float(similarity_scores[idx]))
                for idx in top_n_indices
            ]
            
            return similar_products
            
        except Exception as e:
            logger.error(f"Error in get_similar_products: {str(e)}")
            return []
    
    def recommend_for_user(self, user_interactions: Union[pd.DataFrame, List[str]], 
                         n: int = 10, exclude_viewed: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend products for a user based on their past interactions.
        Parameters:
        -----------
        user_interactions : pd.DataFrame or list
            DataFrame or list of product IDs the user has interacted with
        n : int
            Number of recommendations to return
        exclude_viewed : bool
            Whether to exclude products the user has already interacted with
        Returns:
        --------
        list
            List of tuples (product_id, score) for recommended products
        """
        try:
            if isinstance(user_interactions, pd.DataFrame):
                if 'product_id' not in user_interactions.columns:
                    raise ValueError("user_interactions DataFrame must contain 'product_id' column")
                user_product_ids = user_interactions['product_id'].tolist()
            else:
                user_product_ids = user_interactions
            
            valid_product_ids = [pid for pid in user_product_ids if pid in self.product_id_map]
            
            if not valid_product_ids:
                logger.warning("No valid product IDs found in user interactions")
                return []
            
            all_similar_products = {}
            for pid in valid_product_ids:
                similar_products = self.get_similar_products(pid, n=n+len(valid_product_ids))
                for similar_pid, score in similar_products:
                    if similar_pid in all_similar_products:
                        all_similar_products[similar_pid] = max(all_similar_products[similar_pid], score)
                    else:
                        all_similar_products[similar_pid] = score
            
            if exclude_viewed:
                for pid in valid_product_ids:
                    if pid in all_similar_products:
                        del all_similar_products[pid]
            
            recommendations = sorted(all_similar_products.items(), key=lambda x: x[1], reverse=True)[:n]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in recommend_for_user: {str(e)}")
            return []
    
    def save_model(self, filename='content_based_model.pkl'):
        """
        Save the model to a file.
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'product_id_map': self.product_id_map,
            'reverse_product_id_map': self.reverse_product_id_map,
            'product_ids': self.product_ids,
            'feature_columns': self.feature_columns
        }
        if hasattr(self, 'vectorizer'):
            model_data['vectorizer'] = self.vectorizer
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='content_based_model.pkl'):
        """
        Load the model from a file.
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
        Returns:
        --------
        self : ContentBasedRecommender
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.similarity_matrix = model_data['similarity_matrix']
        self.product_id_map = model_data['product_id_map']
        self.reverse_product_id_map = model_data['reverse_product_id_map']
        self.product_ids = model_data['product_ids']
        self.feature_columns = model_data['feature_columns']
        if 'vectorizer' in model_data:
            self.vectorizer = model_data['vectorizer']
        return self
    
    def evaluate(self, test_interactions, k=10):
        """
        Evaluate the recommender on test data.
        Parameters:
        -----------
        test_interactions : pd.DataFrame
            DataFrame containing user-item interactions for testing
        k : int
            Number of recommendations to consider
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if 'user_id' not in test_interactions.columns or 'product_id' not in test_interactions.columns:
            raise ValueError("test_interactions must contain 'user_id' and 'product_id' columns")
        
        user_interactions = test_interactions.groupby('user_id')
        precision_at_k = []
        recall_at_k = []
        
        for user_id, interactions in user_interactions:
            n_interactions = len(interactions)
            n_train = int(0.8 * n_interactions)
            
            if n_train == 0 or n_interactions - n_train == 0:
                continue
            
            train_interactions = interactions.iloc[:n_train]
            test_interactions_user = interactions.iloc[n_train:]
            actual_products = set(test_interactions_user['product_id'])
            
            try:
                recommendations = self.recommend_for_user(train_interactions, n=k)
                recommended_products = set([rec[0] for rec in recommendations])
                
                if recommended_products:
                    precision = len(actual_products.intersection(recommended_products)) / len(recommended_products)
                    precision_at_k.append(precision)
                
                if actual_products:
                    recall = len(actual_products.intersection(recommended_products)) / len(actual_products)
                    recall_at_k.append(recall)
            except ValueError:
                continue
        
        avg_precision = np.mean(precision_at_k) if precision_at_k else 0
        avg_recall = np.mean(recall_at_k) if recall_at_k else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1_score': f1_score,
            'k': k
        }
