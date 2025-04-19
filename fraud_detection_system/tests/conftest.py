"""
Pytest configuration and fixtures.
"""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from src.config import Config
from src.database.models import Base
from src.database.service import DatabaseService
from src.models.credit_card_fraud import CreditCardFraudModel

@pytest.fixture(scope="session")
def config():
    """Get test configuration."""
    # Override environment variables for testing
    os.environ['DB_NAME'] = 'fraud_detection_test'
    os.environ['API_DEBUG'] = 'True'
    return Config.get_instance()

@pytest.fixture(scope="session")
def db_engine(config):
    """Create test database engine."""
    db_url = f"postgresql://{config.db.username}:{config.db.password}@{config.db.host}:{config.db.port}/{config.db.database}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create test database session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture(scope="function")
def db_service(db_engine):
    """Create test database service."""
    service = DatabaseService()
    service.engine = db_engine
    return service

@pytest.fixture
def sample_transaction():
    """Create sample transaction data."""
    return {
        'transaction_id': 'T12345',
        'user_id': 'U12345',
        'amount': 299.99,
        'currency': 'USD',
        'timestamp': datetime.utcnow(),
        'payment_method': 'credit_card',
        'card_present': False,
        'merchant_id': 'M12345',
        'merchant_category': 'electronics',
        'billing_country': 'US',
        'shipping_country': 'US',
        'ip_address': '192.168.1.1',
        'device_id': 'D12345'
    }

@pytest.fixture
def sample_user():
    """Create sample user data."""
    return {
        'user_id': 'U12345',
        'email': 'user@example.com',
        'registration_date': datetime.utcnow() - timedelta(days=30),
        'country': 'US',
        'risk_score': 0.1,
        'is_blocked': False
    }

@pytest.fixture
def sample_device():
    """Create sample device data."""
    return {
        'device_id': 'D12345',
        'user_id': 'U12345',
        'device_type': 'mobile',
        'os': 'iOS',
        'browser': 'Safari',
        'ip_address': '192.168.1.1',
        'first_seen': datetime.utcnow() - timedelta(days=30),
        'last_seen': datetime.utcnow(),
        'is_trusted': True
    }

@pytest.fixture
def credit_card_model():
    """Create credit card fraud model."""
    return CreditCardFraudModel()

@pytest.fixture
def sample_transaction_history():
    """Create sample transaction history."""
    now = datetime.utcnow()
    return [
        {
            'transaction_id': f'T{i}',
            'user_id': 'U12345',
            'amount': 100 + i * 10,
            'currency': 'USD',
            'timestamp': now - timedelta(hours=i),
            'payment_method': 'credit_card',
            'billing_country': 'US',
            'shipping_country': 'US'
        }
        for i in range(10)
    ]

@pytest.fixture
def sample_login_history():
    """Create sample login history."""
    now = datetime.utcnow()
    return [
        {
            'user_id': 'U12345',
            'device_id': 'D12345',
            'ip_address': '192.168.1.1',
            'timestamp': now - timedelta(hours=i),
            'success': i % 5 != 0  # Every 5th login fails
        }
        for i in range(20)
    ]

@pytest.fixture
def sample_risk_rule():
    """Create sample risk rule."""
    return {
        'name': 'high_amount_new_account',
        'description': 'High transaction amount for new account',
        'rule_type': 'transaction',
        'conditions': {
            'amount_threshold': 500,
            'account_age_days': 30
        },
        'risk_score': 0.8,
        'is_active': True
    } 