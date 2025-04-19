"""
Hybrid Recommendation System for E-commerce

This module combines content-based, collaborative, and sentiment-based filtering
approaches to create a comprehensive recommendation system.
"""

import numpy as np
import pandas as pd
import os
import pickle
from src.content_based_filtering import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.sentiment_based_filtering import SentimentBasedRecommender

class HybridRecommender:
    """
    A class for hybrid recommendations combining multiple filtering approaches.
    
    This class integrates content-based, collaborative, and sentiment-based
    filtering to provide comprehensive product recommendations.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the HybridRecommender.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.content_recommender = ContentBasedRecommender(model_dir=model_dir)
        self.collaborative_recommender = CollaborativeFilteringRecommender(model_dir=model_dir)
        self.sentiment_recommender = SentimentBasedRecommender(model_dir=model_dir)
        
        # Default weights optimized for electronics and home appliances
        self.weights = {
            'content': 0.4,  # Increased for electronics where features matter more
            'collaborative': 0.4,
            'sentiment': 0.2
        }
        
        # Category weights for different product types
        self.category_weights = {
            'electronics': {'content': 0.5, 'collaborative': 0.3, 'sentiment': 0.2},
            'home_appliances': {'content': 0.5, 'collaborative': 0.3, 'sentiment': 0.2},
            'commercial': {'content': 0.4, 'collaborative': 0.4, 'sentiment': 0.2},
            'default': {'content': 0.3, 'collaborative': 0.5, 'sentiment': 0.2}
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def initialize_recommenders(self, content_recommender=None, collaborative_recommender=None, sentiment_recommender=None):
        """
        Initialize with pre-trained recommenders.
        
        Parameters:
        -----------
        content_recommender : ContentBasedRecommender, optional
            Pre-trained content-based recommender
        collaborative_recommender : CollaborativeFilteringRecommender, optional
            Pre-trained collaborative filtering recommender
        sentiment_recommender : SentimentBasedRecommender, optional
            Pre-trained sentiment-based recommender
            
        Returns:
        --------
        self : HybridRecommender
            Returns self for method chaining
        """
        if content_recommender is not None:
            self.content_recommender = content_recommender
            
        if collaborative_recommender is not None:
            self.collaborative_recommender = collaborative_recommender
            
        if sentiment_recommender is not None:
            self.sentiment_recommender = sentiment_recommender
            
        return self
    
    def fit(self, products_df, interactions_df, reviews_df=None, 
            content_features=None, collaborative_method='item_based',
            use_sentiment=True, sentiment_method='vader'):
        """
        Fit all recommendation models.
        
        Parameters:
        -----------
        products_df : pd.DataFrame
            DataFrame containing product information
        interactions_df : pd.DataFrame
            DataFrame containing user-item interactions
        reviews_df : pd.DataFrame
            DataFrame containing product reviews
        content_features : list
            List of column names to use as features for content-based filtering
        collaborative_method : str
            Method to use for collaborative filtering
        use_sentiment : bool
            Whether to use sentiment-based filtering
        sentiment_method : str
            Method to use for sentiment analysis ('vader' or 'ml')
            
        Returns:
        --------
        self : HybridRecommender
            Returns self for method chaining
        """
        # Fit content-based recommender
        if content_features:
            self.content_recommender.fit(products_df, feature_columns=content_features)
        else:
            # Try to use product description or name if available
            if 'description' in products_df.columns:
                self.content_recommender.fit_from_text(products_df, 'description')
            elif 'name' in products_df.columns:
                self.content_recommender.fit_from_text(products_df, 'name')
            else:
                # Use all non-ID columns as features
                features = [col for col in products_df.columns if col != 'product_id']
                self.content_recommender.fit(products_df, feature_columns=features)
        
        # Fit collaborative recommender
        self.collaborative_recommender.fit(interactions_df, method=collaborative_method)
        
        # Fit sentiment recommender if reviews are available
        if use_sentiment and reviews_df is not None:
            if sentiment_method == 'vader':
                self.sentiment_recommender.fit_vader_sentiment(reviews_df)
            elif sentiment_method == 'ml':
                self.sentiment_recommender.fit_ml_sentiment(reviews_df)
        
        return self
    
    def set_weights(self, content_weight=0.3, collaborative_weight=0.5, sentiment_weight=0.2):
        """
        Set weights for different recommendation approaches.
        
        Parameters:
        -----------
        content_weight : float
            Weight for content-based recommendations
        collaborative_weight : float
            Weight for collaborative filtering recommendations
        sentiment_weight : float
            Weight for sentiment-based adjustments
            
        Returns:
        --------
        self : HybridRecommender
            Returns self for method chaining
        """
        # Normalize weights to sum to 1
        total_weight = content_weight + collaborative_weight + sentiment_weight
        
        self.weights = {
            'content': content_weight / total_weight,
            'collaborative': collaborative_weight / total_weight,
            'sentiment': sentiment_weight / total_weight
        }
        
        # Update default weights in category weights
        self.category_weights['default'] = self.weights.copy()
        
        return self
    
    def set_category_weights(self, category, content_weight=None, collaborative_weight=None, sentiment_weight=None):
        """
        Set weights for a specific product category.
        
        Parameters:
        -----------
        category : str
            Product category to set weights for (e.g., 'electronics', 'home_appliances', 'commercial')
        content_weight : float, optional
            Weight for content-based recommendations
        collaborative_weight : float, optional
            Weight for collaborative filtering recommendations
        sentiment_weight : float, optional
            Weight for sentiment-based adjustments
            
        Returns:
        --------
        self : HybridRecommender
            Returns self for method chaining
        """
        # Get current weights for the category or use default
        current_weights = self.category_weights.get(category, self.weights.copy())
        
        # Update only specified weights
        if content_weight is not None:
            current_weights['content'] = content_weight
        if collaborative_weight is not None:
            current_weights['collaborative'] = collaborative_weight
        if sentiment_weight is not None:
            current_weights['sentiment'] = sentiment_weight
            
        # Normalize weights to sum to 1
        total_weight = sum(current_weights.values())
        normalized_weights = {k: v / total_weight for k, v in current_weights.items()}
        
        # Store normalized weights
        self.category_weights[category] = normalized_weights
        
        return self
    
    def recommend_for_user(self, user_id, user_interactions=None, n=10, exclude_viewed=True, 
                           products_df=None, category=None):
        """
        Generate hybrid recommendations for a user with category optimizations.
        
        Parameters:
        -----------
        user_id : str or int
            ID of the user to recommend products for
        user_interactions : pd.DataFrame
            DataFrame containing the user's past interactions (optional)
        n : int
            Number of recommendations to return
        exclude_viewed : bool
            Whether to exclude products the user has already interacted with
        products_df : pd.DataFrame
            DataFrame containing product information with categories (optional)
        category : str
            Specific product category to focus on (e.g., 'electronics')
            
        Returns:
        --------
        list
            List of tuples (product_id, score) for recommended products
        """
        recommendations = {}
        
        # Determine which weights to use based on category
        weights = self.weights
        if category and category in self.category_weights:
            weights = self.category_weights[category]
        
        # Get collaborative filtering recommendations
        try:
            cf_recs = self.collaborative_recommender.recommend_for_user(user_id, n=n*2, exclude_viewed=exclude_viewed)
            for product_id, score in cf_recs:
                if product_id not in recommendations:
                    recommendations[product_id] = {
                        'score': 0, 
                        'methods': [],
                        'category': None  # Will be populated if products_df is provided
                    }
                recommendations[product_id]['score'] += score * weights['collaborative']
                recommendations[product_id]['methods'].append('collaborative')
        except Exception as e:
            print(f"Collaborative filtering recommendations failed: {e}")
        
        # Get content-based recommendations if user interactions are provided
        if user_interactions is not None:
            try:
                cb_recs = self.content_recommender.recommend_for_user(user_interactions, n=n*2, exclude_viewed=exclude_viewed)
                for product_id, score in cb_recs:
                    if product_id not in recommendations:
                        recommendations[product_id] = {
                            'score': 0, 
                            'methods': [],
                            'category': None
                        }
                    recommendations[product_id]['score'] += score * weights['content']
                    recommendations[product_id]['methods'].append('content')
            except Exception as e:
                print(f"Content-based recommendations failed: {e}")
        
        # Adjust scores based on sentiment if available
        if hasattr(self.sentiment_recommender, 'product_sentiment_scores') and self.sentiment_recommender.product_sentiment_scores is not None:
            for product_id in recommendations:
                try:
                    sentiment_score = self.sentiment_recommender.get_product_sentiment(product_id)
                    recommendations[product_id]['score'] += sentiment_score * weights['sentiment']
                    recommendations[product_id]['methods'].append('sentiment')
                except Exception as e:
                    pass  # Silently continue if sentiment fails for a product
        
        # Apply category information and filters if products_df is provided
        if products_df is not None and 'product_id' in products_df.columns:
            category_col = None
            # Find the category column
            for col in ['category', 'product_category', 'categories']:
                if col in products_df.columns:
                    category_col = col
                    break
            
            if category_col:
                # Add category info to recommendations
                for product_id in recommendations:
                    product_info = products_df[products_df['product_id'] == product_id]
                    if not product_info.empty:
                        recommendations[product_id]['category'] = product_info[category_col].iloc[0]
                
                # Filter by category if specified
                if category:
                    recommendations = {
                        pid: info for pid, info in recommendations.items()
                        if info['category'] and (
                            category.lower() in str(info['category']).lower() or
                            # Handle subcategories for electronics and home appliances
                            (category == 'electronics' and any(
                                subcat in str(info['category']).lower() 
                                for subcat in ['electronic', 'computer', 'laptop', 'phone', 'tv', 'audio', 'camera']
                            )) or
                            (category == 'home_appliances' and any(
                                subcat in str(info['category']).lower() 
                                for subcat in ['appliance', 'kitchen', 'refrigerator', 'washer', 'dryer', 'vacuum']
                            ))
                        )
                    }
                
                # Apply category-specific weights
                for product_id, info in recommendations.items():
                    prod_category = info['category']
                    if prod_category and prod_category in self.category_weights:
                        cat_weights = self.category_weights[prod_category]
                        # Recalculate score using category-specific weights
                        info['score'] = 0
                        if 'collaborative' in info['methods']:
                            info['score'] += cat_weights['collaborative']
                        if 'content' in info['methods']:
                            info['score'] += cat_weights['content']
                        if 'sentiment' in info['methods']:
                            info['score'] += cat_weights['sentiment']
        
        # Boost scores for electronics and home appliances if requested
        if category in ['electronics', 'home_appliances', 'commercial']:
            for product_id, info in recommendations.items():
                if info['category'] and category.lower() in str(info['category']).lower():
                    # Apply a boost factor for matching category
                    info['score'] *= 1.2
        
        # Sort recommendations by score
        sorted_recommendations = sorted(
            [(product_id, info['score'], info['methods']) for product_id, info in recommendations.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N recommendations
        return [(product_id, score) for product_id, score, _ in sorted_recommendations[:n]]
    
    def recommend_similar_products(self, product_id, n=10):
        """
        Recommend products similar to a given product.
        
        Parameters:
        -----------
        product_id : str or int
            ID of the product to find similar items for
        n : int
            Number of similar products to return
            
        Returns:
        --------
        list
            List of tuples (product_id, score) for similar products
        """
        recommendations = {}
        
        # Get content-based similar products
        try:
            cb_recs = self.content_recommender.get_similar_products(product_id, n=n*2)
            for similar_id, score in cb_recs:
                if similar_id not in recommendations:
                    recommendations[similar_id] = {'score': 0, 'methods': []}
                recommendations[similar_id]['score'] += score * (self.weights['content'] / (self.weights['content'] + self.weights['collaborative']))
                recommendations[similar_id]['methods'].append('content')
        except Exception as e:
            print(f"Content-based similar products failed: {e}")
        
        # Get collaborative filtering similar products
        try:
            cf_recs = self.collaborative_recommender.recommend_similar_items(product_id, n=n*2)
            for similar_id, score in cf_recs:
                if similar_id not in recommendations:
                    recommendations[similar_id] = {'score': 0, 'methods': []}
                recommendations[similar_id]['score'] += score * (self.weights['collaborative'] / (self.weights['content'] + self.weights['collaborative']))
                recommendations[similar_id]['methods'].append('collaborative')
        except Exception as e:
            print(f"Collaborative filtering similar products failed: {e}")
        
        # Adjust scores based on sentiment if available
        if hasattr(self.sentiment_recommender, 'product_sentiment_scores') and self.sentiment_recommender.product_sentiment_scores is not None:
            for similar_id in recommendations:
                try:
                    sentiment_score = self.sentiment_recommender.get_product_sentiment(similar_id)
                    recommendations[similar_id]['score'] += sentiment_score * self.weights['sentiment']
                    recommendations[similar_id]['methods'].append('sentiment')
                except Exception as e:
                    print(f"Sentiment adjustment failed for product {similar_id}: {e}")
        
        # Sort recommendations by score
        sorted_recommendations = sorted(
            [(similar_id, info['score'], info['methods']) for similar_id, info in recommendations.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N recommendations
        return [(similar_id, score) for similar_id, score, _ in sorted_recommendations[:n]]
    
    def recommend_trending_products(self, n=10, min_interactions=10):
        """
        Recommend trending products based on popularity and sentiment.
        
        Parameters:
        -----------
        n : int
            Number of recommendations to return
        min_interactions : int
            Minimum number of interactions required for a product to be considered
            
        Returns:
        --------
        list
            List of tuples (product_id, score) for trending products
        """
        # Get popular products from collaborative data
        popular_products = {}
        
        if hasattr(self.collaborative_recommender, 'user_item_matrix') and self.collaborative_recommender.user_item_matrix is not None:
            # Calculate popularity as sum of interactions
            item_popularity = np.sum(self.collaborative_recommender.user_item_matrix > 0, axis=0)
            
            # Get products with minimum interactions
            popular_indices = np.where(item_popularity >= min_interactions)[0]
            
            # Convert indices to product IDs
            for idx in popular_indices:
                product_id = self.collaborative_recommender.reverse_item_id_map[idx]
                popularity_score = item_popularity[idx] / np.max(item_popularity)
                popular_products[product_id] = {'score': popularity_score, 'methods': ['popularity']}
        
        # Adjust scores based on sentiment if available
        if hasattr(self.sentiment_recommender, 'product_sentiment_scores') and self.sentiment_recommender.product_sentiment_scores is not None:
            for product_id in popular_products:
                try:
                    sentiment_score = self.sentiment_recommender.get_product_sentiment(product_id)
                    # Combine popularity and sentiment
                    popular_products[product_id]['score'] = 0.7 * popular_products[product_id]['score'] + 0.3 * sentiment_score
                    popular_products[product_id]['methods'].append('sentiment')
                except Exception as e:
                    print(f"Sentiment adjustment failed for product {product_id}: {e}")
        
        # Sort by score
        sorted_trending = sorted(
            [(product_id, info['score'], info['methods']) for product_id, info in popular_products.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N trending products
        return [(product_id, score) for product_id, score, _ in sorted_trending[:n]]
    
    def save_model(self, filename='hybrid_recommender.pkl'):
        """
        Save the hybrid model to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        # Save individual models first
        self.content_recommender.save_model('content_based_model.pkl')
        self.collaborative_recommender.save_model('collaborative_filtering_model.pkl')
        
        if hasattr(self.sentiment_recommender, 'product_sentiment_scores') and self.sentiment_recommender.product_sentiment_scores is not None:
            self.sentiment_recommender.save_model('sentiment_model.pkl')
        
        # Save hybrid model configuration
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'weights': self.weights
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='hybrid_recommender.pkl'):
        """
        Load the hybrid model from files.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the hybrid configuration from
            
        Returns:
        --------
        self : HybridRecommender
            Returns self for method chaining
        """
        # Load individual models
        try:
            self.content_recommender.load_model('content_based_model.pkl')
        except Exception as e:
            print(f"Could not load content-based model: {e}")
        
        try:
            self.collaborative_recommender.load_model('collaborative_filtering_model.pkl')
        except Exception as e:
            print(f"Could not load collaborative filtering model: {e}")
        
        try:
            self.sentiment_recommender.load_model('sentiment_model.pkl')
        except Exception as e:
            print(f"Could not load sentiment model: {e}")
        
        # Load hybrid configuration
        model_path = os.path.join(self.model_dir, filename)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data['weights']
        except Exception as e:
            print(f"Could not load hybrid configuration: {e}")
        
        return self
    
    def evaluate(self, test_interactions, k=10):
        """
        Evaluate the hybrid recommender on test data.
        
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
        
        # Group interactions by user
        user_interactions = test_interactions.groupby('user_id')
        
        # Metrics to calculate
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        
        # For each user in the test set
        for user_id, interactions in user_interactions:
            # Skip users not in the training set
            if user_id not in self.collaborative_recommender.user_id_map:
                continue
            
            # Split interactions into train (80%) and test (20%)
            n_interactions = len(interactions)
            n_train = int(0.8 * n_interactions)
            
            if n_train == 0 or n_interactions - n_train == 0:
                continue  # Skip users with too few interactions
            
            train_interactions = interactions.iloc[:n_train]
            test_interactions = interactions.iloc[n_train:]
            
            # Get actual items the user interacted with in the test set
            actual_items = set(test_interactions['product_id'])
            
            # Get recommendations based on training interactions
            try:
                recommendations = self.recommend_for_user(user_id, train_interactions, n=k)
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
            except Exception as e:
                print(f"Evaluation failed for user {user_id}: {e}")
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
