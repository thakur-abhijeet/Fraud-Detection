# Recommendation System Guide

This guide provides detailed information about the recommendation system's components, algorithms, and usage.

## Table of Contents

1. [System Overview](#system-overview)
2. [Recommendation Algorithms](#recommendation-algorithms)
3. [Data Processing](#data-processing)
4. [API Endpoints](#api-endpoints)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## System Overview

The recommendation system uses a hybrid approach combining three main recommendation techniques:

1. **Content-Based Filtering**
   - Uses product features and attributes
   - Good for cold-start problems
   - Works well with detailed product information

2. **Collaborative Filtering**
   - Uses user-item interactions
   - Three methods available:
     - User-based
     - Item-based
     - Matrix factorization
   - Requires sufficient user interaction data

3. **Sentiment-Based Filtering**
   - Uses product reviews
   - Two approaches:
     - VADER lexicon-based
     - Machine learning-based
   - Helps identify popular and well-received products

## Recommendation Algorithms

### Content-Based Filtering

The content-based recommender uses cosine similarity to find similar products based on their features:

```python
similarity = cosine_similarity(feature_vectors)
```

Features can be:
- Text (description, name, features)
- Categorical (category, brand)
- Numerical (price, rating)

### Collaborative Filtering

1. **User-Based CF**
   - Finds similar users based on their ratings
   - Uses cosine similarity between user vectors
   - Predicts ratings using weighted average of similar users' ratings

2. **Item-Based CF**
   - Finds similar items based on user ratings
   - Uses cosine similarity between item vectors
   - Predicts ratings using weighted average of similar items' ratings

3. **Matrix Factorization**
   - Decomposes user-item matrix into latent factors
   - Uses SVD (Singular Value Decomposition)
   - Captures underlying patterns in user-item interactions

### Sentiment Analysis

1. **VADER Approach**
   - Uses pre-trained sentiment lexicon
   - Analyzes text for positive/negative sentiment
   - Combines with ratings for final sentiment score

2. **Machine Learning Approach**
   - Uses TF-IDF for feature extraction
   - Trains classifier on review text
   - Predicts sentiment probability

## Data Processing

### Text Preprocessing

1. Convert to lowercase
2. Remove special characters
3. Remove numbers
4. Tokenize
5. Remove stopwords
6. Lemmatize

### Feature Extraction

1. **Text Features**
   - TF-IDF vectorization
   - Maximum 5000 features
   - English stopwords removed

2. **Categorical Features**
   - One-hot encoding
   - Top 10 categories kept
   - Others grouped as "Other"

3. **Numerical Features**
   - Min-max scaling
   - Missing values filled with mean

## API Endpoints

### Data Management

1. **Load Data**
   ```
   POST /api/load_data
   ```
   - Loads and preprocesses data
   - Initializes recommenders
   - Returns data summary

### Recommendations

1. **Content-Based**
   ```
   POST /api/content_recommendations
   ```
   - Requires product_id
   - Returns similar products

2. **Collaborative Filtering**
   ```
   POST /api/collaborative_recommendations
   ```
   - Requires user_id
   - Optional: method (user_based/item_based/matrix_factorization)

3. **Sentiment-Based**
   ```
   POST /api/sentiment_recommendations
   ```
   - Optional: min_reviews threshold

4. **Hybrid**
   ```
   POST /api/hybrid_recommendations
   ```
   - Combines all approaches
   - Category-specific weights

## Configuration

### Weights Configuration

Default weights in `weights.json`:
```json
{
  "electronics": {
    "content": 0.5,
    "collaborative": 0.3,
    "sentiment": 0.2
  },
  "home_appliances": {
    "content": 0.5,
    "collaborative": 0.3,
    "sentiment": 0.2
  },
  "default": {
    "content": 0.3,
    "collaborative": 0.5,
    "sentiment": 0.2
  }
}
```

### Environment Variables

Required in `.env`:
```
DATA_DIR=/path/to/data
MODEL_DIR=/path/to/models
DEBUG=True/False
```

## Best Practices

1. **Data Quality**
   - Ensure complete product information
   - Clean and preprocess data properly
   - Handle missing values appropriately

2. **Performance**
   - Use appropriate method based on data size
   - Cache frequently accessed data
   - Monitor memory usage

3. **Recommendation Quality**
   - Regularly evaluate metrics
   - Adjust weights based on category
   - Consider business rules

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Check file paths
   - Verify file formats
   - Ensure required columns

2. **Memory Issues**
   - Reduce feature dimensions
   - Use sparse matrices
   - Implement batch processing

3. **Poor Recommendations**
   - Check data quality
   - Adjust weights
   - Consider more features

### Debugging Tips

1. Enable debug mode in `.env`
2. Check logs for errors
3. Validate input data
4. Monitor performance metrics 