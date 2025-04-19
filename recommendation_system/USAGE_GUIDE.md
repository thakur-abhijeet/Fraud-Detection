# Recommendation System Usage Guide

This guide provides step-by-step instructions for using the hybrid recommendation system that combines content-based, collaborative, and sentiment-based filtering approaches.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [System Initialization](#system-initialization)
5. [Training the Models](#training-the-models)
6. [Generating Recommendations](#generating-recommendations)
7. [Saving and Loading Models](#saving-and-loading-models)
8. [Evaluating Performance](#evaluating-performance)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.7+
- Required Python packages:
  ```bash
  numpy
  pandas
  scikit-learn
  nltk
  flask (for API)
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd recommendation_system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data model
   ```

## Data Preparation

The system requires three main types of data:

1. **Product Data** (`products.csv` or `products.json`):
   - Required columns: `product_id`
   - Optional columns: `name`, `description`, `category`, and other product attributes

2. **User-Item Interactions** (`interactions.csv` or `interactions.json`):
   - Required columns: `user_id`, `product_id`
   - Optional columns: `rating`, `timestamp`

3. **Product Reviews** (`reviews.csv` or `reviews.json`):
   - Required columns: `product_id`, `review_text`
   - Optional columns: `rating`, `user_id`, `timestamp`

Example data structure:
```python
# products.csv
product_id,name,description,category
1,Product A,Description A,Electronics
2,Product B,Description B,Home Appliances

# interactions.csv
user_id,product_id,rating,timestamp
101,1,5,2023-01-01
101,2,4,2023-01-02

# reviews.csv
product_id,review_text,rating,user_id
1,"Great product!",5,101
2,"Good but could be better",4,102
```

## System Initialization

1. Import the necessary modules:
   ```python
   from src.hybrid_recommender import HybridRecommender
   from src.data_preprocessing import DataPreprocessor
   ```

2. Initialize the data preprocessor:
   ```python
   preprocessor = DataPreprocessor(data_dir='data/')
   ```

3. Load and preprocess the data:
   ```python
   preprocessor.load_data(
       products_file='products.csv',
       interactions_file='interactions.csv',
       reviews_file='reviews.csv'
   )
   preprocessor.clean_data()
   ```

4. Initialize the hybrid recommender:
   ```python
   recommender = HybridRecommender(model_dir='model/')
   ```

## Training the Models

1. Train the hybrid recommender:
   ```python
   recommender.fit(
       products_df=preprocessor.products_df,
       interactions_df=preprocessor.interactions_df,
       reviews_df=preprocessor.reviews_df,
       content_features=['name', 'description', 'category'],
       collaborative_method='item_based',
       use_sentiment=True,
       sentiment_method='vader'
   )
   ```

2. (Optional) Set custom weights for different recommendation approaches:
   ```python
   recommender.set_weights(
       content_weight=0.4,
       collaborative_weight=0.4,
       sentiment_weight=0.2
   )
   ```

3. (Optional) Set category-specific weights:
   ```python
   recommender.set_category_weights(
       category='electronics',
       content_weight=0.5,
       collaborative_weight=0.3,
       sentiment_weight=0.2
   )
   ```

## Generating Recommendations

1. Get personalized recommendations for a user:
   ```python
   user_id = 101
   recommendations = recommender.recommend_for_user(
       user_id=user_id,
       n=10,
       exclude_viewed=True
   )
   ```

2. Get recommendations for a specific category:
   ```python
   electronics_recommendations = recommender.recommend_for_user(
       user_id=user_id,
       n=10,
       category='electronics'
   )
   ```

3. Get similar products:
   ```python
   similar_products = recommender.recommend_similar_products(
       product_id=1,
       n=5
   )
   ```

4. Get trending products:
   ```python
   trending_products = recommender.recommend_trending_products(
       n=10,
       min_interactions=5
   )
   ```

## Saving and Loading Models

1. Save the trained model:
   ```python
   recommender.save_model('hybrid_recommender.pkl')
   ```

2. Load a saved model:
   ```python
   recommender = HybridRecommender(model_dir='model/')
   recommender.load_model('hybrid_recommender.pkl')
   ```

## Evaluating Performance

1. Evaluate the recommender on test data:
   ```python
   metrics = recommender.evaluate(
       test_interactions=test_df,
       k=10
   )
   print(metrics)
   ```

2. Expected output format:
   ```python
   {
       'precision@k': 0.45,
       'recall@k': 0.38,
       'ndcg@k': 0.52,
       'f1_score': 0.41,
       'k': 10
   }
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Data Loading Errors**
   - Ensure all required columns are present in your data files
   - Check file formats (CSV/JSON) and encoding
   - Verify file paths are correct

2. **Memory Issues**
   - Reduce the number of features in content-based filtering
   - Use sparse matrices for large datasets
   - Increase `min_interactions` threshold

3. **Poor Recommendations**
   - Check data quality and completeness
   - Adjust weights for different recommendation approaches
   - Ensure sufficient training data
   - Try different collaborative filtering methods

4. **Model Loading/Saving Issues**
   - Verify file permissions
   - Check disk space
   - Ensure consistent Python versions between save and load

### Getting Help

For additional support:
1. Check the system logs in `logs/` directory
2. Review the API documentation
3. Consult the source code documentation
4. Open an issue on the project repository

## Best Practices

1. **Data Quality**
   - Ensure clean and consistent data
   - Handle missing values appropriately
   - Remove duplicates
   - Normalize text data

2. **Performance Optimization**
   - Use appropriate data structures
   - Implement caching where possible
   - Monitor memory usage
   - Regular model updates

3. **Recommendation Quality**
   - Regularly evaluate model performance
   - Adjust weights based on business needs
   - Consider user feedback
   - Implement A/B testing

4. **Maintenance**
   - Regular model retraining
   - Monitor system performance
   - Keep dependencies updated
   - Maintain documentation

## Next Steps

1. Implement user feedback collection
2. Set up automated model retraining
3. Add more recommendation strategies
4. Implement A/B testing framework
5. Add monitoring and alerting 