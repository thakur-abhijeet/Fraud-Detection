"""
Collaborative Filtering Module for E-commerce Recommendation System

This module implements collaborative filtering for product recommendations
based on user-item interactions.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Tuple, Dict, Optional, Union
#from recommendation_system.src.utils.logging_config import logger

class CollaborativeFilteringRecommender:
    """
    A class for collaborative filtering recommendations.
    
    This class implements both memory-based (user-based, item-based) and
    model-based (matrix factorization) collaborative filtering approaches.
    """
    
    def __init__(self, model_dir: str = '../model'):
        """
        Initialize the CollaborativeFilteringRecommender.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.user_item_matrix = None
        self.user_id_map = None
        self.item_id_map = None
        self.reverse_user_id_map = None
        self.reverse_item_id_map = None
        self.method = None
        self.model = None
        self.min_users = 5  # Minimum number of users for matrix factorization
        self.min_items = 5  # Minimum number of items for matrix factorization
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def fit(self, interactions_df: pd.DataFrame, method: str = 'item_based', 
            n_factors: int = 50, user_id_col: str = 'user_id', 
            item_id_col: str = 'product_id', rating_col: str = 'rating', 
            min_interactions: int = 1) -> 'CollaborativeFilteringRecommender':
        """
        Fit the collaborative filtering recommender with user-item interactions.
        
        Parameters:
        -----------
        interactions_df : pd.DataFrame
            DataFrame containing user-item interactions
        method : str
            Collaborative filtering method to use
        n_factors : int
            Number of latent factors for matrix factorization
        user_id_col : str
            Name of the column containing user IDs
        item_id_col : str
            Name of the column containing item IDs
        rating_col : str
            Name of the column containing ratings
        min_interactions : int
            Minimum number of interactions per user to include
            
        Returns:
        --------
        self : CollaborativeFilteringRecommender
            Returns self for method chaining
        """
        try:
            # Verify required columns exist
            required_cols = [user_id_col, item_id_col]
            missing_cols = [col for col in required_cols if col not in interactions_df.columns]
            if missing_cols:
                raise ValueError(f"Required columns not found: {missing_cols}")

            # Check rating column
            if rating_col and rating_col not in interactions_df.columns:
                logger.warning(f"Rating column '{rating_col}' not found, creating implicit feedback with value 1.0")
                interactions_df = interactions_df.copy()
                interactions_df[rating_col] = 1.0
            
            # Store the method
            self.method = method
            
            # Filter users with too few interactions
            if min_interactions > 1:
                user_counts = interactions_df[user_id_col].value_counts()
                valid_users = user_counts[user_counts >= min_interactions].index
                interactions_df = interactions_df[interactions_df[user_id_col].isin(valid_users)]
                
                if len(interactions_df) == 0:
                    raise ValueError(f"No users with at least {min_interactions} interactions found.")
            
            # Create user and item ID mappings
            user_ids = interactions_df[user_id_col].unique()
            item_ids = interactions_df[item_id_col].unique()
            
            if len(user_ids) < self.min_users or len(item_ids) < self.min_items:
                logger.warning(f"Dataset too small for {method} (users: {len(user_ids)}, items: {len(item_ids)}). "
                              f"Switching to item_based method.")
                method = 'item_based'
            
            self.user_id_map = {id: i for i, id in enumerate(user_ids)}
            self.item_id_map = {id: i for i, id in enumerate(item_ids)}
            self.reverse_user_id_map = {i: id for id, i in self.user_id_map.items()}
            self.reverse_item_id_map = {i: id for id, i in self.item_id_map.items()}
            
            # Create user-item matrix
            interactions_df['user_idx'] = interactions_df[user_id_col].map(self.user_id_map)
            interactions_df['item_idx'] = interactions_df[item_id_col].map(self.item_id_map)
            
            # Create a sparse matrix for efficiency
            n_users = len(self.user_id_map)
            n_items = len(self.item_id_map)
            
            # Create a dense matrix for simplicity (for large datasets, use sparse matrices)
            self.user_item_matrix = np.zeros((n_users, n_items))
            for _, row in interactions_df.iterrows():
                self.user_item_matrix[int(row['user_idx']), int(row['item_idx'])] = row[rating_col]
            
            # Implement the chosen collaborative filtering method
            if method == 'user_based':
                # User-based collaborative filtering
                self.user_similarity = cosine_similarity(self.user_item_matrix)
                self.model = self.user_similarity
                
            elif method == 'item_based':
                # Item-based collaborative filtering
                self.item_similarity = cosine_similarity(self.user_item_matrix.T)
                self.model = self.item_similarity
                
            elif method == 'matrix_factorization':
                # Matrix factorization using SVD
                # Convert to sparse matrix for efficiency
                sparse_user_item = csr_matrix(self.user_item_matrix)
                
                # Perform SVD
                # Ensure n_factors is valid (less than min dimension - 1)
                k = min(n_factors, min(n_users, n_items)-1)
                
                # Handle case where k could be 0 or negative
                if k < 1:
                    k = 1
                    logger.warning(f"n_factors was too large, using k=1 instead")
                    
                try:
                    u, sigma, vt = svds(sparse_user_item, k=k)
                    
                    # Convert sigma to diagonal matrix
                    sigma_diag = np.diag(sigma)
                    
                    # Reconstruct the matrix
                    self.user_factors = u
                    self.item_factors = vt.T
                    self.reconstructed_matrix = u.dot(sigma_diag).dot(vt)
                    self.model = self.reconstructed_matrix
                except Exception as e:
                    logger.error(f"Matrix factorization failed: {str(e)}. Falling back to item-based method.")
                    self.item_similarity = cosine_similarity(self.user_item_matrix.T)
                    self.model = self.item_similarity
                    self.method = 'item_based'
                
            else:
                raise ValueError(f"Unknown method: {method}. Choose from 'user_based', 'item_based', or 'matrix_factorization'")
            
            return self
            
        except Exception as e:
            logger.error(f"Error in fit method: {str(e)}")
            raise

    def recommend_for_user(self, user_id: Union[str, int], n: int = 10, 
                          exclude_viewed: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend products for a user based on collaborative filtering.
        
        Parameters:
        -----------
        user_id : str or int
            ID of the user to recommend products for
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
            # Check if user exists in the model
            if user_id not in self.user_id_map:
                logger.warning(f"User ID {user_id} not found in the model. Using cold start strategy.")
                return self._cold_start_recommendations(n)
            
            user_idx = self.user_id_map[user_id]
            
            if self.method == 'user_based':
                recommendations = self._user_based_recommend(user_idx, n, exclude_viewed)
            elif self.method == 'item_based':
                recommendations = self._item_based_recommend(user_idx, n, exclude_viewed)
            elif self.method == 'matrix_factorization':
                recommendations = self._matrix_factorization_recommend(user_idx, n, exclude_viewed)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in recommend_for_user: {str(e)}")
            return self._cold_start_recommendations(n)
    
    def _cold_start_recommendations(self, n: int) -> List[Tuple[str, float]]:
        """
        Generate recommendations for cold start users.
        
        Parameters:
        -----------
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of tuples (product_id, score) for recommended products
        """
        try:
            # Get most popular items
            item_popularity = np.sum(self.user_item_matrix > 0, axis=0)
            top_items = np.argsort(item_popularity)[::-1][:n]
            
            return [(self.reverse_item_id_map[idx], float(item_popularity[idx])) 
                    for idx in top_items]
        except Exception as e:
            logger.error(f"Error in cold start recommendations: {str(e)}")
            return []

    def _user_based_recommend(self, user_idx: int, n: int, 
                            exclude_viewed: bool) -> List[Tuple[str, float]]:
        """User-based collaborative filtering recommendations."""
        try:
            user_similarities = self.user_similarity[user_idx]
            weighted_ratings = np.zeros(self.user_item_matrix.shape[1])
            similarity_sums = np.zeros(self.user_item_matrix.shape[1])
            
            for other_user_idx in range(self.user_item_matrix.shape[0]):
                if other_user_idx == user_idx:
                    continue
                
                similarity = user_similarities[other_user_idx]
                if similarity <= 0:
                    continue
                
                ratings = self.user_item_matrix[other_user_idx]
                for item_idx in range(len(ratings)):
                    if ratings[item_idx] > 0:
                        weighted_ratings[item_idx] += similarity * ratings[item_idx]
                        similarity_sums[item_idx] += similarity
            
            predicted_ratings = np.zeros(self.user_item_matrix.shape[1])
            for item_idx in range(len(predicted_ratings)):
                if similarity_sums[item_idx] > 0:
                    predicted_ratings[item_idx] = weighted_ratings[item_idx] / similarity_sums[item_idx]
            
            if exclude_viewed:
                user_ratings = self.user_item_matrix[user_idx]
                viewed_items = np.where(user_ratings > 0)[0]
                predicted_ratings[viewed_items] = float('-inf')
            
            top_item_indices = np.argsort(predicted_ratings)[::-1][:n]
            
            return [(self.reverse_item_id_map[idx], float(predicted_ratings[idx]))
                    for idx in top_item_indices if predicted_ratings[idx] > float('-inf')]
        except Exception as e:
            logger.error(f"Error in user-based recommendations: {str(e)}")
            return []

    def _item_based_recommend(self, user_idx: int, n: int, 
                            exclude_viewed: bool) -> List[Tuple[str, float]]:
        """Item-based collaborative filtering recommendations."""
        try:
            user_ratings = self.user_item_matrix[user_idx]
            if np.sum(user_ratings > 0) == 0:
                return self._cold_start_recommendations(n)
            
            predicted_ratings = np.zeros(self.user_item_matrix.shape[1])
            
            for item_idx in range(self.user_item_matrix.shape[1]):
                if user_ratings[item_idx] > 0 and exclude_viewed:
                    continue
                
                item_similarities = self.item_similarity[item_idx]
                weighted_sum = 0
                similarity_sum = 0
                
                for rated_item_idx in range(len(user_ratings)):
                    if user_ratings[rated_item_idx] > 0:
                        similarity = item_similarities[rated_item_idx]
                        if similarity <= 0:
                            continue
                        weighted_sum += similarity * user_ratings[rated_item_idx]
                        similarity_sum += similarity
                
                if similarity_sum > 0:
                    predicted_ratings[item_idx] = weighted_sum / similarity_sum
            
            top_item_indices = np.argsort(predicted_ratings)[::-1][:n]
            
            return [(self.reverse_item_id_map[idx], float(predicted_ratings[idx]))
                    for idx in top_item_indices if predicted_ratings[idx] > 0]
        except Exception as e:
            logger.error(f"Error in item-based recommendations: {str(e)}")
            return []

    def _matrix_factorization_recommend(self, user_idx: int, n: int, 
                                      exclude_viewed: bool) -> List[Tuple[str, float]]:
        """Matrix factorization recommendations."""
        try:
            predicted_ratings = self.reconstructed_matrix[user_idx]
            
            if exclude_viewed:
                user_ratings = self.user_item_matrix[user_idx]
                viewed_items = np.where(user_ratings > 0)[0]
                predicted_ratings[viewed_items] = float('-inf')
            
            top_item_indices = np.argsort(predicted_ratings)[::-1][:n]
            
            return [(self.reverse_item_id_map[idx], float(predicted_ratings[idx]))
                    for idx in top_item_indices if predicted_ratings[idx] > float('-inf')]
        except Exception as e:
            logger.error(f"Error in matrix factorization recommendations: {str(e)}")
            return []

    def recommend_similar_items(self, item_id, n=10):
        """
        Recommend similar items based on collaborative filtering.
        
        Parameters:
        -----------
        item_id : str or int
            ID of the item to find similar items for
        n : int
            Number of similar items to return
            
        Returns:
        --------
        list
            List of tuples (item_id, similarity_score) for similar items
        """
        if self.method != 'item_based':
            raise ValueError("This method is only available for item-based collaborative filtering")
        
        # Check if item exists in the model
        if item_id not in self.item_id_map:
            raise ValueError(f"Item ID {item_id} not found in the model")
        
        item_idx = self.item_id_map[item_id]
        
        # Get item similarity scores
        item_similarities = self.item_similarity[item_idx]
        
        # Get indices of similar items
        similar_indices = np.argsort(item_similarities)[::-1]
        
        # Exclude the same item
        similar_indices = similar_indices[similar_indices != item_idx]
        
        # Get top N similar items
        top_n_indices = similar_indices[:n]
        
        # Convert indices back to item IDs and create result list
        similar_items = [
            (self.reverse_item_id_map[idx], item_similarities[idx])
            for idx in top_n_indices
        ]
        
        return similar_items
    
    def save_model(self, filename='collaborative_filtering_model.pkl'):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_id_map': self.reverse_user_id_map,
            'reverse_item_id_map': self.reverse_item_id_map,
            'method': self.method,
            'model': self.model
        }
        
        # Save additional data for matrix factorization
        if self.method == 'matrix_factorization':
            model_data['user_factors'] = self.user_factors
            model_data['item_factors'] = self.item_factors
            model_data['reconstructed_matrix'] = self.reconstructed_matrix
        elif self.method == 'item_based':
            # Save item similarity matrix
            model_data['item_similarity'] = self.item_similarity
        elif self.method == 'user_based':
            # Save user similarity matrix
            model_data['user_similarity'] = self.user_similarity
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='collaborative_filtering_model.pkl'):
        """
        Load the model from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
            
        Returns:
        --------
        self : CollaborativeFilteringRecommender
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.user_item_matrix = model_data['user_item_matrix']
            self.user_id_map = model_data['user_id_map']
            self.item_id_map = model_data['item_id_map']
            self.reverse_user_id_map = model_data['reverse_user_id_map']
            self.reverse_item_id_map = model_data['reverse_item_id_map']
            self.method = model_data['method']
            self.model = model_data['model']
            
            # Load additional data for matrix factorization
            if self.method == 'matrix_factorization':
                self.user_factors = model_data['user_factors']
                self.item_factors = model_data['item_factors']
                self.reconstructed_matrix = model_data['reconstructed_matrix']
            elif self.method == 'item_based':
                # Load item similarity matrix
                self.item_similarity = model_data['item_similarity']
            elif self.method == 'user_based':
                # Load user similarity matrix
                self.user_similarity = model_data['user_similarity']
                
            print(f"Model successfully loaded from {model_path}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading model: {e}")
        
        return self
    
    def evaluate(self, test_interactions, k=10, user_id_col='user_id', item_id_col='product_id', rating_col='rating'):
        """
        Evaluate the recommender on test data.
        
        Parameters:
        -----------
        test_interactions : pd.DataFrame
            DataFrame containing user-item interactions for testing
        k : int
            Number of recommendations to consider
        user_id_col : str
            Name of the column containing user IDs
        item_id_col : str
            Name of the column containing item IDs
        rating_col : str
            Name of the column containing ratings
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Verify required columns exist
        required_cols = [user_id_col, item_id_col]
        if rating_col:
            required_cols.append(rating_col)
            
        missing_cols = [col for col in required_cols if col not in test_interactions.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")
        
        # Group interactions by user
        user_interactions = test_interactions.groupby(user_id_col)
        
        # Metrics to calculate
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        
        # For each user in the test set
        for user_id, interactions in user_interactions:
            # Skip users not in the training set
            if user_id not in self.user_id_map:
                continue
            
            # Split interactions into train (80%) and test (20%)
            n_interactions = len(interactions)
            n_train = int(0.8 * n_interactions)
            
            if n_train == 0 or n_interactions - n_train == 0:
                continue  # Skip users with too few interactions
            
            train_interactions = interactions.iloc[:n_train]
            test_interactions = interactions.iloc[n_train:]
            
            # Get actual items the user interacted with in the test set
            actual_items = set(test_interactions[item_id_col])
            
            # Get recommendations based on training interactions
            try:
                recommendations = self.recommend_for_user(user_id, n=k)
                recommended_items = [rec[0] for rec in recommendations]
                
                # Calculate precision and recall
                if recommended_items:
                    relevant_and_recommended = actual_items.intersection(recommended_items)
                    precision = len(relevant_and_recommended) / len(recommended_items)
                    precision_at_k.append(precision)
                
                if actual_items:
                    relevant_and_recommended = actual_items.intersection(recommended_items)
                    recall = len(relevant_and_recommended) / len(actual_items)
                    recall_at_k.append(recall)
                
                # Calculate NDCG
                if actual_items:
                    # Create relevance vector (1 if item is relevant, 0 otherwise)
                    relevance = [1 if item in actual_items else 0 for item in recommended_items]
                    
                    # Calculate DCG
                    dcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance)])
                    
                    # Calculate ideal DCG (all relevant items at the top)
                    ideal_relevance = sorted(relevance, reverse=True)
                    idcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
                    
                    # Calculate NDCG
                    if idcg > 0:
                        ndcg = dcg / idcg
                        ndcg_at_k.append(ndcg)
            except ValueError as e:
                # Skip users with no valid recommendations
                print(f"Skipping user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        avg_precision = np.mean(precision_at_k) if precision_at_k else 0
        avg_recall = np.mean(recall_at_k) if recall_at_k else 0
        avg_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0
        
        # Calculate F1 score
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'ndcg@k': avg_ndcg,
            'f1_score': f1_score,
            'k': k
        }
