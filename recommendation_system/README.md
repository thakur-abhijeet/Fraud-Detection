# E-commerce Recommendation System

A comprehensive recommendation system that combines content-based, collaborative, and sentiment-based filtering approaches to provide personalized product recommendations.

## Features

- **Content-Based Filtering**: Recommends products based on their features and attributes
- **Collaborative Filtering**: Uses user-item interactions to find similar users/items
- **Sentiment-Based Filtering**: Incorporates product reviews and sentiment analysis
- **Hybrid Recommendations**: Combines multiple approaches for better recommendations
- **RESTful API**: Easy integration with web applications
- **Customizable Weights**: Adjust recommendation weights based on product categories

## System Architecture

The system consists of the following components:

1. **API Layer** (`api/app.py`):
   - RESTful endpoints for recommendations
   - Data loading and preprocessing
   - Error handling and response formatting

2. **Recommendation Engines**:
   - Content-Based Recommender (`src/content_based_filtering.py`)
   - Collaborative Filtering Recommender (`src/collaborative_filtering.py`)
   - Sentiment-Based Recommender (`src/sentiment_based_filtering.py`)
   - Hybrid Recommender (`src/hybrid_recommender.py`)

3. **Data Processing** (`src/data_preprocessing.py`):
   - Data loading and cleaning
   - Feature extraction
   - Text preprocessing
   - Data splitting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recommendation_system.git
cd recommendation_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Usage

1. Start the API server:
```bash
python api/app.py
```

2. Load data:
```bash
curl -X POST http://localhost:5000/api/load_data \
  -H "Content-Type: application/json" \
  -d '{
    "products_file": "products.csv",
    "ratings_file": "ratings.csv",
    "reviews_file": "reviews.csv"
  }'
```

3. Get recommendations:
```bash
# Content-based recommendations
curl -X POST http://localhost:5000/api/content_recommendations \
  -H "Content-Type: application/json" \
  -d '{"product_id": "123", "num_recommendations": 5}'

# Collaborative filtering recommendations
curl -X POST http://localhost:5000/api/collaborative_recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "456", "num_recommendations": 5}'

# Hybrid recommendations
curl -X POST http://localhost:5000/api/hybrid_recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "456",
    "category": "electronics",
    "num_recommendations": 5
  }'
```

## Configuration

The system can be configured through the following files:

- `config.py`: Global configuration settings
- `weights.json`: Recommendation weights for different categories
- `.env`: Environment variables

## Data Format

### Products Data
```csv
product_id,name,description,category,price,features
1,Product A,Description A,Electronics,99.99,Feature A|Feature B
```

### Ratings Data
```csv
user_id,product_id,rating,timestamp
101,1,5,2023-01-01
```

### Reviews Data
```csv
user_id,product_id,review_text,rating,timestamp
101,1,Great product!,5,2023-01-01
```

## Evaluation Metrics

The system uses the following metrics to evaluate recommendations:
- Precision@K
- Recall@K
- NDCG@K
- F1 Score

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for text processing
- scikit-learn for machine learning algorithms
- Flask for the API framework 