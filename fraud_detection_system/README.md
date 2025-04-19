# E-commerce Fraud Detection System

A comprehensive fraud detection system for e-commerce platforms that identifies and prevents various types of fraud including credit card fraud, account takeovers, friendly fraud (chargeback fraud), promotion abuse, refund fraud, bot attacks, and synthetic identity fraud.

## Features

- **Multi-layered Fraud Detection**: Combines multiple approaches to provide comprehensive protection
- **Credit Card Fraud Detection**: Identifies fraudulent credit card transactions using machine learning
- **Account Takeover Prevention**: Detects unauthorized access attempts through behavioral analysis
- **Friendly Fraud Detection**: Predicts potential chargeback fraud using historical patterns
- **Promotion Abuse Detection**: Identifies misuse of promotional offers
- **Refund Fraud Detection**: Detects suspicious refund patterns
- **Bot Attack Detection**: Identifies automated activities and attacks
- **Synthetic Identity Detection**: Detects fake or synthetic identities
- **Unified Risk Scoring**: Combines results from all modules for comprehensive risk assessment
- **Real-time API**: RESTful API for real-time fraud detection
- **Configurable Risk Thresholds**: Adjustable risk thresholds based on business needs
- **Detailed Risk Analysis**: Provides detailed risk factors and explanations
- **Historical Analysis**: Tracks and analyzes patterns over time

## Project Structure

```
fraud_detection_system/
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
│   └── models/          # Fraud detection models
│       ├── base_model.py             # Base model class
│       ├── credit_card_fraud.py      # Credit card fraud model
│       ├── account_takeover.py       # Account takeover model
│       └── friendly_fraud.py         # Friendly fraud model
├── tests/               # Test suite
│   ├── conftest.py     # Test configuration
│   └── test_*.py       # Test modules
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
├── .gitignore         # Git ignore file
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Requirements

- Python 3.7+
- PostgreSQL 12+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud_detection_system.git
cd fraud_detection_system
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

## Usage

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

# Analyze a transaction
response = requests.post(
    f"{base_url}/api/analyze_transaction",
    json={
        "transaction_data": {
            "transaction_id": "T123456",
            "user_id": "U12345",
            "timestamp": "2025-04-01T14:30:00",
            "amount": 299.99,
            "payment_method": "credit_card",
            "product_category": "electronics",
            "ip_address": "192.168.1.1",
            "device_id": "D12345",
            "billing_country": "US",
            "shipping_country": "US"
        },
        "user_data": {
            "user_id": "U12345",
            "account_age_days": 120,
            "previous_purchases": 5
        }
    }
)

print(json.dumps(response.json(), indent=2))
```

### Running Tests

```bash
pytest tests/
```

## Configuration

The system is configured through environment variables. See `.env.example` for available options:

- Database configuration (host, port, credentials)
- API settings (host, port, debug mode)
- Model parameters (training samples, retraining interval)
- Fraud detection thresholds
- Risk weights for different fraud types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
