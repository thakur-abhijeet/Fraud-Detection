# Fraud Detection System Setup Guide

## System Overview

This fraud detection system is designed to identify and prevent various types of fraud in e-commerce platforms using machine learning. The system includes multiple detection modules:

1. Credit Card Fraud Detection
2. Account Takeover Prevention
3. Friendly Fraud Detection
4. Additional Fraud Types (Promotion abuse, Refund fraud, etc.)
5. Unified Risk Scoring

## Prerequisites

Before setting up the system, ensure you have:

1. Python 3.7+ installed
2. PostgreSQL 12+ installed and running
3. Redis server installed (for caching and task queue)
4. Git installed
5. Virtual environment tool (venv or conda)

## Step-by-Step Setup Guide

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud_detection_system.git
cd fraud_detection_system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraud_detection
DB_USER=your_username
DB_PASSWORD=your_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=5001
DEBUG=True

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Configuration
MODEL_PATH=./model
DATA_PATH=./data
```

### 3. Database Setup

1. Create PostgreSQL database:
```bash
createdb fraud_detection
```

2. Initialize the database:
```bash
python -c "from src.database.service import DatabaseService; DatabaseService().init_db()"
```

### 4. Data Preparation

1. Place your training data in the `data/raw` directory
2. Run data preprocessing:
```bash
python src/data_preprocessing.py
```

### 5. Model Training

Train individual models:
```bash
# Credit Card Fraud Model
python src/credit_card_fraud_detection.py --train

# Account Takeover Model
python src/account_takeover_prevention.py --train

# Friendly Fraud Model
python src/friendly_fraud_detection.py --train

# Additional Fraud Models
python src/additional_fraud_detection.py --train
```

### 6. Running the System

1. Start Redis server:
```bash
redis-server
```

2. Start Celery worker:
```bash
celery -A src.celery_app worker --loglevel=info
```

3. Start the API server:
```bash
python api/app.py
```

The API will be available at http://localhost:5001

## Using the System

### API Endpoints

1. Transaction Analysis:
```bash
POST /api/analyze_transaction
Content-Type: application/json

{
    "transaction_data": {
        "transaction_id": "T123456",
        "user_id": "U12345",
        "amount": 299.99,
        "payment_method": "credit_card",
        "timestamp": "2024-03-20T14:30:00"
    }
}
```

2. Batch Analysis:
```bash
POST /api/analyze_batch
Content-Type: application/json

{
    "transactions": [
        {
            "transaction_id": "T123456",
            "user_id": "U12345",
            "amount": 299.99
        }
    ]
}
```

### Monitoring

1. Access Flower dashboard (Celery monitoring):
   - http://localhost:5555

2. Access API documentation:
   - http://localhost:5001/docs

## Development Guidelines

1. Code Style:
   - Follow PEP 8 guidelines
   - Use Black for code formatting
   - Run pylint for code quality checks

2. Testing:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=src tests/
```

3. Documentation:
   - Update docstrings for new functions
   - Keep README.md updated
   - Document API changes

## Troubleshooting

1. Database Connection Issues:
   - Check PostgreSQL service is running
   - Verify database credentials in .env
   - Ensure database exists

2. Model Training Issues:
   - Check data format in data/raw
   - Verify model parameters in config.py
   - Check GPU availability for deep learning models

3. API Issues:
   - Check API logs
   - Verify Redis connection
   - Check Celery worker status

## Support

For issues and support:
1. Check the documentation
2. Open an issue on GitHub
3. Contact the development team

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details. 