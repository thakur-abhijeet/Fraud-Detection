# 🎯 Product Recommendation System

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](documentation/SYSTEM_GUIDE.md)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive recommendation system that combines multiple approaches to provide personalized product recommendations using advanced machine learning techniques.

## 🌟 Features

- **Collaborative Filtering**: User-item interaction analysis for personalized recommendations
- **Content-Based Filtering**: Product feature analysis for similar item recommendations
- **Hybrid Recommender**: Combines multiple approaches for optimal results
- **Sentiment Analysis**: Incorporates user reviews and feedback
- **Real-time Updates**: Dynamic recommendation updates based on user behavior
- **Cold-start Handling**: Special handling for new users and items
- **Scalable Architecture**: Designed for high-performance and scalability
- **API Integration**: RESTful API for easy integration
- **Customizable Weights**: Adjustable weights for different recommendation methods
- **Performance Monitoring**: Built-in performance tracking and analysis

## 📊 System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Input     │────▶│  Processing     │────▶│  Recommendation │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Storage   │◀───▶│  ML Models      │◀───▶│  API Layer      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
recommendation_system/
├── api/                    # API implementation
│   ├── app.py             # Main API application
│   ├── routes/            # API route handlers
│   └── test_api.py        # API tests
├── data/                  # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── model/                # Trained models
├── notebook/            # Jupyter notebooks
├── src/                 # Source code
│   ├── config.py        # Configuration management
│   ├── database/        # Database models and services
│   │   ├── models.py    # SQLAlchemy models
│   │   └── service.py   # Database service
│   └── models/          # Recommendation models
│       ├── collaborative_filtering.py  # Collaborative filtering model
│       ├── content_based_filtering.py  # Content-based filtering model
│       ├── hybrid_recommender.py       # Hybrid recommender model
│       └── sentiment_based_filtering.py # Sentiment-based filtering model
├── tests/               # Test suite
│   ├── conftest.py     # Test configuration
│   └── test_*.py       # Test modules
├── documentation/       # System documentation
│   └── SYSTEM_GUIDE.md # Comprehensive system guide
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
├── .gitignore         # Git ignore file
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PostgreSQL 12+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recommendation_system.git
cd recommendation_system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python -c "from src.database.service import DatabaseService; DatabaseService().init_db()"
```

## 💻 Usage

### Running the API

```bash
python api/app.py
```

The API will be available at http://localhost:5001.

### Example API Usage

```python
import requests
import json

# Base URL for the API
base_url = "http://localhost:5001"

# Get recommendations for a user
response = requests.post(
    f"{base_url}/api/recommendations",
    json={
        "user_id": "U12345",
        "n_recommendations": 5,
        "method": "hybrid"  # Options: collaborative, content, hybrid, sentiment
    }
)

print(json.dumps(response.json(), indent=2))
```

### Running Tests

```bash
pytest tests/
```

## ⚙️ Configuration

The system is configured through environment variables. See `.env.example` for available options:

- Database configuration (host, port, credentials)
- API settings (host, port, debug mode)
- Model parameters (training samples, update interval)
- Recommendation thresholds
- Method weights for hybrid approach

## 📚 Documentation

For detailed documentation, please refer to:
- [System Guide](documentation/SYSTEM_GUIDE.md)
- [API Documentation](api/README.md)
- [Model Documentation](src/models/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, please:
- Open an issue
- Contact: support@recommendationsystem.com
- Visit: docs.recommendationsystem.com

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by various open-source recommendation systems
- Built with ❤️ by the recommendation system team 