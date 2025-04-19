"""
Flask API for E-commerce Recommendation System
This module implements a RESTful API for the recommendation system,
providing endpoints for content-based, collaborative, and sentiment-based filtering.
"""

from flask import Flask, request, jsonify
import sys
import os
import pandas as pd
import json
import traceback
from typing import Dict, Any, Optional
from werkzeug.exceptions import HTTPException

# Add parent directory to path to import recommendation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import recommendation modules
from src.data_preprocessing import DataPreprocessor
from src.content_based_filtering import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.sentiment_based_filtering import SentimentBasedRecommender
from src.hybrid_recommender import HybridRecommender
from src.utils.logging_config import logger
from src.utils.cache import cache

app = Flask(__name__)

# Initialize recommendation components
data_preprocessor = DataPreprocessor()
content_recommender = ContentBasedRecommender()
collaborative_recommender = CollaborativeFilteringRecommender()
sentiment_recommender = SentimentBasedRecommender()
hybrid_recommender = HybridRecommender()

# Global variables to store loaded data
products_df = None
ratings_df = None
reviews_df = None
users_df = None

@app.errorhandler(Exception)
def handle_error(e: Exception) -> tuple:
    """
    Global error handler for the API.
    
    Parameters:
    -----------
    e : Exception
        The exception that occurred
        
    Returns:
    --------
    tuple
        (error_response, status_code)
    """
    # Log the error
    logger.error(f"Error occurred: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Handle HTTP exceptions
    if isinstance(e, HTTPException):
        return jsonify({
            'status': 'error',
            'message': str(e),
            'code': e.code
        }), e.code
    
    # Handle custom exceptions
    if isinstance(e, ValueError):
        return jsonify({
            'status': 'error',
            'message': str(e),
            'code': 400
        }), 400
    
    # Handle other exceptions
    return jsonify({
        'status': 'error',
        'message': 'An unexpected error occurred',
        'code': 500
    }), 500

@app.route('/api/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify API is running."""
    return {
        'status': 'healthy',
        'message': 'Recommendation System API is running',
        'version': '1.0.0'
    }

@app.route('/api/load_data', methods=['POST'])
# @cache.cached(key_prefix='load_data:')
def load_data() -> Dict[str, Any]:
    """
    Load and preprocess data for the recommendation system.
    
    Expected JSON payload:
    {
        "data_dir": "path/to/data",
        "products_file": "products.csv",
        "ratings_file": "ratings.csv",
        "reviews_file": "reviews.csv",
        "users_file": "users.csv"
    }
    """
    global products_df, ratings_df, reviews_df, users_df

    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data provided in request")
            
        data_dir = data.get('data_dir', '../data/')
        
        # Update data directory in preprocessor
        data_preprocessor.data_dir = data_dir

        # Load products data
        if 'products_file' in data:
            file_path = os.path.join(data_dir, data['products_file'])
            if os.path.exists(file_path):
                data_preprocessor.load_data(products_file=data['products_file'])
                data_preprocessor.clean_data()
                products_df = data_preprocessor.products_df
            else:
                logger.warning(f"Products file not found: {file_path}")
                products_df = data_preprocessor.generate_sample_product_data()
        else:
            logger.info("No products file specified, generating sample data")
            products_df = data_preprocessor.generate_sample_product_data()

        # Load ratings data (interactions)
        if 'ratings_file' in data:
            file_path = os.path.join(data_dir, data['ratings_file'])
            if os.path.exists(file_path):
                data_preprocessor.load_data(interactions_file=data['ratings_file'])
                data_preprocessor.clean_data()
                ratings_df = data_preprocessor.interactions_df
            else:
                logger.warning(f"Ratings file not found: {file_path}")
                ratings_df = data_preprocessor.generate_sample_rating_data(products_df)
        else:
            logger.info("No ratings file specified, generating sample data")
            ratings_df = data_preprocessor.generate_sample_rating_data(products_df)

        # Load reviews data
        if 'reviews_file' in data:
            file_path = os.path.join(data_dir, data['reviews_file'])
            if os.path.exists(file_path):
                data_preprocessor.load_data(reviews_file=data['reviews_file'])
                data_preprocessor.clean_data()
                reviews_df = data_preprocessor.reviews_df
            else:
                logger.warning(f"Reviews file not found: {file_path}")
                reviews_df = data_preprocessor.generate_sample_review_data(products_df)
        else:
            logger.info("No reviews file specified, generating sample data")
            reviews_df = data_preprocessor.generate_sample_review_data(products_df)

        # Load users data
        if 'users_file' in data:
            file_path = os.path.join(data_dir, data['users_file'])
            if os.path.exists(file_path):
                data_preprocessor.load_data(users_file=data['users_file'])
                data_preprocessor.clean_data()
                users_df = data_preprocessor.users_df
            else:
                logger.warning(f"Users file not found: {file_path}")
                users_df = data_preprocessor.generate_sample_user_data()
        else:
            logger.info("No users file specified, generating sample data")
            users_df = data_preprocessor.generate_sample_user_data()

        # Process category columns to improve electronics and home appliances recommendations
        if products_df is not None:
            # Identify category column
            category_col = None
            for col in ['category', 'product_category', 'categories']:
                if col in products_df.columns:
                    category_col = col
                    break
            
            # Enhance category information for better recommendations
            if category_col:
                # Define category keywords for better matching
                electronics_keywords = ['electronics', 'electronic', 'computer', 'laptop', 'phone', 'smartphone', 
                                        'tv', 'television', 'camera', 'audio', 'headphone', 'speaker']
                home_appliances_keywords = ['appliance', 'home appliance', 'kitchen', 'refrigerator', 'fridge', 
                                           'washer', 'dryer', 'vacuum', 'microwave', 'blender', 'mixer']
                commercial_keywords = ['commercial', 'business', 'office', 'industrial', 'professional', 
                                     'enterprise', 'corporate']
                
                # Standardize categories
                def standardize_category(cat):
                    cat_lower = str(cat).lower()
                    if any(kw in cat_lower for kw in electronics_keywords):
                        return 'electronics'
                    elif any(kw in cat_lower for kw in home_appliances_keywords):
                        return 'home_appliances'
                    elif any(kw in cat_lower for kw in commercial_keywords):
                        return 'commercial'
                    return cat
                
                # Add standardized category column
                products_df['category_standard'] = products_df[category_col].apply(standardize_category)

        # Now fit recommenders with correct dataframes
        try:
            # Try to extract useful features for content-based filtering
            text_features = []
            for col in ['description', 'name', 'title', 'features']:
                if col in products_df.columns:
                    text_features.append(col)
            
            categorical_features = []
            for col in ['category', 'brand', 'category_standard']:
                if col in products_df.columns:
                    categorical_features.append(col)
                    
            numerical_features = []
            for col in ['price', 'rating_avg']:
                if col in products_df.columns:
                    numerical_features.append(col)
                    
            if text_features or categorical_features or numerical_features:
                content_features = data_preprocessor.extract_content_features(
                    text_columns=text_features,
                    categorical_columns=categorical_features,
                    numerical_columns=numerical_features
                )
                content_recommender.fit(content_features)
            else:
                content_recommender.fit(products_df)
                
            collaborative_recommender.fit(ratings_df)
            sentiment_recommender.fit_vader_sentiment(reviews_df)
            
            # Initialize the hybrid recommender
            hybrid_recommender.initialize_recommenders(
                content_recommender=content_recommender,
                collaborative_recommender=collaborative_recommender,
                sentiment_recommender=sentiment_recommender
            )
            
        except Exception as e:
            logger.error(f"Error fitting recommenders: {str(e)}")
            # Fallback to simple initialization
            try:
                content_recommender.fit(products_df)
                collaborative_recommender.fit(ratings_df)
                sentiment_recommender.fit_vader_sentiment(reviews_df)
                
                hybrid_recommender.initialize_recommenders(
                    content_recommender=content_recommender,
                    collaborative_recommender=collaborative_recommender,
                    sentiment_recommender=sentiment_recommender
                )
            except Exception as e2:
                logger.error(f"Fallback initialization also failed: {str(e2)}")
                raise ValueError(f"Could not initialize recommenders: {str(e2)}")

        return {
            'status': 'success',
            'message': 'Data loaded successfully',
            'data_summary': {
                'products': len(products_df) if products_df is not None else 0,
                'ratings': len(ratings_df) if ratings_df is not None else 0,
                'reviews': len(reviews_df) if reviews_df is not None else 0,
                'users': len(users_df) if users_df is not None else 0
            }
        }

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@app.route('/api/content_recommendations', methods=['POST'])
def get_content_recommendations():
    """
    Get content-based recommendations for a product.
    
    Expected JSON payload:
    {
        "product_id": "123",
        "num_recommendations": 5
    }
    """
    try:
        # Check if data is loaded
        if products_df is None:
            return jsonify({
                'status': 'error',
                'message': 'Data not loaded. Call /api/load_data first.'
            }), 400
        
        data = request.get_json()

        # Validate required parameters
        if 'product_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'product_id is required'
            }), 400

        product_id = data['product_id']
        num_recommendations = data.get('num_recommendations', 5)

        # Get recommendations
        recommendations = content_recommender.get_similar_products(product_id, n=num_recommendations)

        # Convert to list of dictionaries for JSON response if needed
        if isinstance(recommendations, pd.DataFrame):
            recommendations_list = recommendations.to_dict('records')
        else:
            # Convert tuple recommendations to dict entries
            recommendations_list = [{'product_id': pid, 'score': score} for pid, score in recommendations]

        return jsonify({
            'status': 'success',
            'product_id': product_id,
            'recommendations': recommendations_list
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

# The rest of the endpoints remain unchanged.
# (Please ensure that other modules which are imported (e.g. CollaborativeFilteringRecommender,
# SentimentBasedRecommender, HybridRecommender) work with these dataframes.)

@app.route('/api/collaborative_recommendations', methods=['POST'])
def get_collaborative_recommendations():
    """
    Get collaborative filtering recommendations for a user.
    
    Expected JSON payload:
    {
        "user_id": "456",
        "num_recommendations": 5,
        "method": "user_based"  # Options: "user_based", "item_based", "matrix_factorization"
    }
    """
    try:
        if ratings_df is None:
            return jsonify({
                'status': 'error',
                'message': 'Data not loaded. Call /api/load_data first.'
            }), 400

        data = request.get_json()

        if 'user_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400

        user_id = data['user_id']
        num_recommendations = data.get('num_recommendations', 5)
        method = data.get('method', 'user_based')

        if method == 'user_based':
            recommendations = collaborative_recommender.user_based_recommend(user_id, n=num_recommendations)
        elif method == 'item_based':
            recommendations = collaborative_recommender.item_based_recommend(user_id, n=num_recommendations)
        elif method == 'matrix_factorization':
            recommendations = collaborative_recommender.matrix_factorization_recommend(user_id, n=num_recommendations)
        else:
            return jsonify({
                'status': 'error',
                'message': f'Invalid method: {method}. Valid options are: user_based, item_based, matrix_factorization'
            }), 400

        if isinstance(recommendations, pd.DataFrame):
            recommendations_list = recommendations.to_dict('records')
        else:
            recommendations_list = recommendations

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'method': method,
            'recommendations': recommendations_list
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/sentiment_recommendations', methods=['POST'])
def get_sentiment_recommendations():
    """
    Get sentiment-based recommendations.
    
    Expected JSON payload:
    {
        "num_recommendations": 5,
        "min_reviews": 3
    }
    """
    try:
        if reviews_df is None:
            return jsonify({
                'status': 'error',
                'message': 'Data not loaded. Call /api/load_data first.'
            }), 400

        data = request.get_json()

        num_recommendations = data.get('num_recommendations', 5)
        min_reviews = data.get('min_reviews', 3)

        recommendations = sentiment_recommender.recommend(n=num_recommendations, min_reviews=min_reviews)

        if isinstance(recommendations, pd.DataFrame):
            recommendations_list = recommendations.to_dict('records')
        else:
            recommendations_list = recommendations

        return jsonify({
            'status': 'success',
            'recommendations': recommendations_list
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/hybrid_recommendations', methods=['POST'])
def get_hybrid_recommendations():
    """
    Get hybrid recommendations combining multiple filtering approaches.
    
    Expected JSON payload:
    {
        "user_id": "456",
        "num_recommendations": 5,
        "category": "electronics",
        "content_weight": 0.4,
        "collaborative_weight": 0.4,
        "sentiment_weight": 0.2
    }
    """
    try:
        if products_df is None or ratings_df is None:
            return jsonify({
                'status': 'error',
                'message': 'Data not loaded. Call /api/load_data first.'
            }), 400

        data = request.get_json()

        if 'user_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400

        user_id = data['user_id']
        num_recommendations = data.get('num_recommendations', 5)
        category = data.get('category')
        
        # Set weights if provided
        if any(k in data for k in ['content_weight', 'collaborative_weight', 'sentiment_weight']):
            content_weight = data.get('content_weight')
            collaborative_weight = data.get('collaborative_weight')
            sentiment_weight = data.get('sentiment_weight')
            
            if category:
                # Set weights for the specific category
                hybrid_recommender.set_category_weights(
                    category=category,
                    content_weight=content_weight,
                    collaborative_weight=collaborative_weight,
                    sentiment_weight=sentiment_weight
                )
            else:
                # Set global weights
                hybrid_recommender.set_weights(
                    content_weight=content_weight or 0.3,
                    collaborative_weight=collaborative_weight or 0.5,
                    sentiment_weight=sentiment_weight or 0.2
                )
        elif category:
            # Use the default weights for this category
            # The weights are already defined in the hybrid recommender
            pass
        
        # Get user's past interactions for content-based filtering
        user_interactions = None
        if ratings_df is not None and 'user_id' in ratings_df.columns:
            user_interactions = ratings_df[ratings_df['user_id'] == user_id]
        
        # Get hybrid recommendations with category filtering
        recommendations = hybrid_recommender.recommend_for_user(
            user_id=user_id,
            user_interactions=user_interactions,
            n=num_recommendations,
            products_df=products_df,
            category=category
        )
        
        # If no recommendations were found and category was specified, try without category filtering
        if not recommendations and category:
            recommendations = hybrid_recommender.recommend_for_user(
                user_id=user_id,
                user_interactions=user_interactions,
                n=num_recommendations,
                products_df=products_df
            )
            
        # Convert to list of dictionaries for JSON response
        recommendations_list = [{'product_id': pid, 'score': score} for pid, score in recommendations]
        
        # Add product details if available
        if products_df is not None and 'product_id' in products_df.columns:
            for rec in recommendations_list:
                product_info = products_df[products_df['product_id'] == rec['product_id']]
                if not product_info.empty:
                    for col in product_info.columns:
                        if col != 'product_id':
                            try:
                                # Only include serializable values
                                value = product_info[col].iloc[0]
                                if pd.notna(value):
                                    if isinstance(value, (str, int, float, bool)):
                                        rec[col] = value
                                    else:
                                        rec[col] = str(value)
                            except:
                                pass

        # Add category-specific message
        category_message = ""
        if category:
            category_message = f"optimized for {category}"
            
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'category': category,
            'category_message': category_message,
            'recommendations': recommendations_list
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/personalized_recommendations', methods=['POST'])
def get_personalized_recommendations():
    """
    Get personalized recommendations for a user based on their browsing history.
    
    Expected JSON payload:
    {
        "user_id": "456",
        "browsing_history": ["123", "456", "789"],
        "num_recommendations": 5
    }
    """
    try:
        if products_df is None or ratings_df is None:
            return jsonify({
                'status': 'error',
                'message': 'Data not loaded. Call /api/load_data first.'
            }), 400

        data = request.get_json()

        if 'user_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400

        if 'browsing_history' not in data or not data['browsing_history']:
            return jsonify({
                'status': 'error',
                'message': 'browsing_history is required and cannot be empty'
            }), 400

        user_id = data['user_id']
        browsing_history = data['browsing_history']
        num_recommendations = data.get('num_recommendations', 5)

        content_recs = []
        for prod_id in browsing_history:
            recs = content_recommender.get_similar_products(prod_id, n=3)
            if isinstance(recs, list):
                content_recs.extend([rec[0] for rec in recs])
            else:
                content_recs.extend(recs)

        collab_recs = collaborative_recommender.user_based_recommend(user_id, n=5)
        if isinstance(collab_recs, pd.DataFrame):
            collab_recs = collab_recs['product_id'].tolist()
        elif isinstance(collab_recs, list) and collab_recs and isinstance(collab_recs[0], dict):
            collab_recs = [r['product_id'] for r in collab_recs]

        all_recs = content_recs + collab_recs
        unique_recs = []
        for rec in all_recs:
            if rec not in unique_recs and rec not in browsing_history:
                unique_recs.append(rec)
                if len(unique_recs) >= num_recommendations:
                    break

        recommended_products = []
        for rec_id in unique_recs:
            product = products_df[products_df['product_id'] == rec_id]
            if not product.empty:
                recommended_products.append(product.iloc[0].to_dict())

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'browsing_history': browsing_history,
            'recommendations': recommended_products
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/trending_products', methods=['GET'])
def get_trending_products():
    """
    Get trending products based on recent ratings and reviews.
    """
    try:
        if ratings_df is None or reviews_df is None:
            return jsonify({
                'status': 'error',
                'message': 'Data not loaded. Call /api/load_data first.'
            }), 400

        num_products = request.args.get('num_products', default=5, type=int)

        if 'timestamp' in ratings_df.columns:
            recent_ratings = ratings_df.sort_values('timestamp', ascending=False)
            recent_product_ids = recent_ratings['product_id'].value_counts().head(num_products).index.tolist()
        else:
            recent_product_ids = ratings_df.groupby('product_id')['rating'].mean().sort_values(ascending=False).head(num_products).index.tolist()

        trending_products = []
        for prod_id in recent_product_ids:
            product = products_df[products_df['product_id'] == prod_id]
            if not product.empty:
                trending_products.append(product.iloc[0].to_dict())

        return jsonify({
            'status': 'success',
            'trending_products': trending_products
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
