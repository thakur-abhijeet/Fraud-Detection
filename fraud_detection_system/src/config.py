"""
Configuration management for the fraud detection system.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    username: str
    password: str
    database: str

@dataclass
class APIConfig:
    """API configuration."""
    host: str
    port: int
    debug: bool
    secret_key: str

@dataclass
class ModelConfig:
    """Model configuration."""
    model_dir: str
    data_dir: str
    min_training_samples: int
    retraining_interval_days: int

@dataclass
class FraudConfig:
    """Fraud detection configuration."""
    velocity_thresholds: Dict[str, float]
    risk_thresholds: Dict[str, float]
    risk_weights: Dict[str, float]

class Config:
    """Main configuration class."""
    
    def __init__(self):
        """Initialize configuration."""
        load_dotenv()  # Load environment variables from .env file
        
        self.db = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            username=os.getenv('DB_USERNAME', ''),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'fraud_detection')
        )
        
        self.api = APIConfig(
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('API_PORT', '5001')),
            debug=os.getenv('API_DEBUG', 'False').lower() == 'true',
            secret_key=os.getenv('API_SECRET_KEY', 'your-secret-key')
        )
        
        self.model = ModelConfig(
            model_dir=os.getenv('MODEL_DIR', '../model'),
            data_dir=os.getenv('DATA_DIR', '../data'),
            min_training_samples=int(os.getenv('MIN_TRAINING_SAMPLES', '1000')),
            retraining_interval_days=int(os.getenv('RETRAINING_INTERVAL_DAYS', '7'))
        )
        
        self.fraud = FraudConfig(
            velocity_thresholds={
                'amount_1h': float(os.getenv('VELOCITY_AMOUNT_1H', '1000')),
                'count_1h': int(os.getenv('VELOCITY_COUNT_1H', '3')),
                'countries_1h': int(os.getenv('VELOCITY_COUNTRIES_1H', '2')),
                'amount_24h': float(os.getenv('VELOCITY_AMOUNT_24H', '2000')),
                'count_24h': int(os.getenv('VELOCITY_COUNT_24H', '10')),
                'countries_24h': int(os.getenv('VELOCITY_COUNTRIES_24H', '3'))
            },
            risk_thresholds={
                'high_risk': float(os.getenv('RISK_THRESHOLD_HIGH', '0.7')),
                'medium_risk': float(os.getenv('RISK_THRESHOLD_MEDIUM', '0.4')),
                'low_risk': float(os.getenv('RISK_THRESHOLD_LOW', '0.2'))
            },
            risk_weights={
                'credit_card_fraud': float(os.getenv('WEIGHT_CREDIT_CARD', '0.3')),
                'account_takeover': float(os.getenv('WEIGHT_ACCOUNT_TAKEOVER', '0.25')),
                'friendly_fraud': float(os.getenv('WEIGHT_FRIENDLY_FRAUD', '0.2')),
                'promotion_abuse': float(os.getenv('WEIGHT_PROMOTION_ABUSE', '0.1')),
                'refund_fraud': float(os.getenv('WEIGHT_REFUND_FRAUD', '0.1')),
                'bot_activity': float(os.getenv('WEIGHT_BOT_ACTIVITY', '0.05'))
            }
        )

    @classmethod
    def get_instance(cls) -> 'Config':
        """Get singleton instance of Config."""
        if not hasattr(cls, '_instance'):
            cls._instance = Config()
        return cls._instance 