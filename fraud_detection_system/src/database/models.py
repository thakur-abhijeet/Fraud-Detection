"""
Database models for the fraud detection system.
"""

from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Transaction(Base):
    """Transaction model."""
    
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(50), unique=True, nullable=False)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    payment_method = Column(String(50), nullable=False)
    card_present = Column(Boolean, default=False)
    merchant_id = Column(String(50))
    merchant_category = Column(String(50))
    billing_country = Column(String(2))
    shipping_country = Column(String(2))
    ip_address = Column(String(45))
    device_id = Column(String(50))
    risk_score = Column(Float)
    risk_level = Column(String(20))
    is_fraud = Column(Boolean, default=False)
    metadata = Column(JSON)

    user = relationship("User", back_populates="transactions")
    fraud_analysis = relationship("FraudAnalysis", back_populates="transaction", uselist=False)

class User(Base):
    """User model."""
    
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True)
    registration_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    country = Column(String(2))
    risk_score = Column(Float, default=0.0)
    is_blocked = Column(Boolean, default=False)
    metadata = Column(JSON)

    transactions = relationship("Transaction", back_populates="user")
    devices = relationship("Device", back_populates="user")
    login_attempts = relationship("LoginAttempt", back_populates="user")

class Device(Base):
    """Device model."""
    
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), unique=True, nullable=False)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False)
    device_type = Column(String(50))
    os = Column(String(50))
    browser = Column(String(50))
    ip_address = Column(String(45))
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_trusted = Column(Boolean, default=False)
    metadata = Column(JSON)

    user = relationship("User", back_populates="devices")

class LoginAttempt(Base):
    """Login attempt model."""
    
    __tablename__ = 'login_attempts'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False)
    device_id = Column(String(50))
    ip_address = Column(String(45))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    risk_score = Column(Float)
    metadata = Column(JSON)

    user = relationship("User", back_populates="login_attempts")

class FraudAnalysis(Base):
    """Fraud analysis model."""
    
    __tablename__ = 'fraud_analyses'

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(50), ForeignKey('transactions.transaction_id'), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    risk_factors = Column(JSON)
    recommended_action = Column(String(50))
    final_action = Column(String(50))
    reviewed_by = Column(String(50))
    review_notes = Column(String(1000))
    metadata = Column(JSON)

    transaction = relationship("Transaction", back_populates="fraud_analysis")

class RiskRule(Base):
    """Risk rule model."""
    
    __tablename__ = 'risk_rules'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(500))
    rule_type = Column(String(50), nullable=False)
    conditions = Column(JSON, nullable=False)
    risk_score = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata = Column(JSON) 