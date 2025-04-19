"""
Database service for the fraud detection system.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.sql import func
from sqlalchemy.pool import QueuePool

from .models import Base, Transaction, User, Device, LoginAttempt, FraudAnalysis, RiskRule
from ..config import Config

class DatabaseService:
    """Service for handling database operations."""

    def __init__(self):
        """Initialize database service."""
        config = Config.get_instance()
        db_url = f"postgresql://{config.db.username}:{config.db.password}@{config.db.host}:{config.db.port}/{config.db.database}"
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

    def init_db(self):
        """Initialize database schema."""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()

    def save_transaction(self, transaction_data: Dict[str, Any]) -> Transaction:
        """
        Save a transaction to the database.

        Args:
            transaction_data: Transaction data to save

        Returns:
            Saved transaction object
        """
        with self.get_session() as session:
            transaction = Transaction(**transaction_data)
            session.add(transaction)
            session.commit()
            return transaction

    def save_fraud_analysis(self, analysis_data: Dict[str, Any]) -> FraudAnalysis:
        """
        Save fraud analysis results.

        Args:
            analysis_data: Fraud analysis data to save

        Returns:
            Saved fraud analysis object
        """
        with self.get_session() as session:
            analysis = FraudAnalysis(**analysis_data)
            session.add(analysis)
            session.commit()
            return analysis

    def get_user_transactions(
        self, 
        user_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Transaction]:
        """
        Get transactions for a user within a date range.

        Args:
            user_id: User identifier
            start_date: Start date for transaction history
            end_date: End date for transaction history

        Returns:
            List of transactions
        """
        with self.get_session() as session:
            query = session.query(Transaction).filter(Transaction.user_id == user_id)

            if start_date:
                query = query.filter(Transaction.timestamp >= start_date)
            if end_date:
                query = query.filter(Transaction.timestamp <= end_date)

            return query.order_by(Transaction.timestamp.desc()).all()

    def get_user_devices(self, user_id: str) -> List[Device]:
        """
        Get devices associated with a user.

        Args:
            user_id: User identifier

        Returns:
            List of devices
        """
        with self.get_session() as session:
            return session.query(Device).filter(Device.user_id == user_id).all()

    def get_login_history(
        self, 
        user_id: str,
        hours: Optional[int] = None
    ) -> List[LoginAttempt]:
        """
        Get login history for a user.

        Args:
            user_id: User identifier
            hours: Number of hours of history to retrieve

        Returns:
            List of login attempts
        """
        with self.get_session() as session:
            query = session.query(LoginAttempt).filter(LoginAttempt.user_id == user_id)

            if hours:
                start_time = datetime.utcnow() - timedelta(hours=hours)
                query = query.filter(LoginAttempt.timestamp >= start_time)

            return query.order_by(LoginAttempt.timestamp.desc()).all()

    def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get risk profile for a user.

        Args:
            user_id: User identifier

        Returns:
            User risk profile
        """
        with self.get_session() as session:
            # Get user data
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                return {}

            # Get transaction statistics
            tx_stats = session.query(
                func.count(Transaction.id).label('total_transactions'),
                func.avg(Transaction.amount).label('avg_amount'),
                func.sum(case((Transaction.is_fraud == True, 1), else_=0)).label('fraud_count')
            ).filter(Transaction.user_id == user_id).first()

            # Get device count
            device_count = session.query(func.count(Device.id))\
                .filter(Device.user_id == user_id).scalar()

            # Get login attempt statistics
            login_stats = session.query(
                func.count(LoginAttempt.id).label('total_attempts'),
                func.sum(case((LoginAttempt.success == False, 1), else_=0)).label('failed_attempts')
            ).filter(LoginAttempt.user_id == user_id).first()

            return {
                'user_id': user_id,
                'risk_score': user.risk_score,
                'is_blocked': user.is_blocked,
                'registration_date': user.registration_date.isoformat(),
                'total_transactions': tx_stats.total_transactions or 0,
                'avg_transaction_amount': float(tx_stats.avg_amount or 0),
                'fraud_transactions': tx_stats.fraud_count or 0,
                'device_count': device_count,
                'total_login_attempts': login_stats.total_attempts or 0,
                'failed_login_attempts': login_stats.failed_attempts or 0
            }

    def update_user_risk_score(self, user_id: str, risk_score: float) -> None:
        """
        Update risk score for a user.

        Args:
            user_id: User identifier
            risk_score: New risk score
        """
        with self.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if user:
                user.risk_score = risk_score
                user.updated_at = datetime.utcnow()
                session.commit()

    def get_active_risk_rules(self, rule_type: Optional[str] = None) -> List[RiskRule]:
        """
        Get active risk rules.

        Args:
            rule_type: Optional rule type filter

        Returns:
            List of active risk rules
        """
        with self.get_session() as session:
            query = session.query(RiskRule).filter(RiskRule.is_active == True)
            
            if rule_type:
                query = query.filter(RiskRule.rule_type == rule_type)
            
            return query.all()

    def save_risk_rule(self, rule_data: Dict[str, Any]) -> RiskRule:
        """
        Save a risk rule.

        Args:
            rule_data: Risk rule data

        Returns:
            Saved risk rule object
        """
        with self.get_session() as session:
            rule = RiskRule(**rule_data)
            session.add(rule)
            session.commit()
            return rule

    def get_transaction_velocity(
        self,
        user_id: str,
        hours: int = 24,
        include_current: bool = True
    ) -> Dict[str, Any]:
        """
        Get transaction velocity metrics for a user.

        Args:
            user_id: User identifier
            hours: Number of hours to look back
            include_current: Whether to include current hour in calculations

        Returns:
            Dictionary with velocity metrics
        """
        with self.get_session() as session:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Base query for transactions in the time window
            query = session.query(Transaction)\
                .filter(Transaction.user_id == user_id)\
                .filter(Transaction.timestamp >= start_time)

            if not include_current:
                current_hour = end_time.replace(minute=0, second=0, microsecond=0)
                query = query.filter(Transaction.timestamp < current_hour)

            transactions = query.all()

            # Calculate metrics
            total_amount = sum(tx.amount for tx in transactions)
            unique_countries = len(set(tx.billing_country for tx in transactions))
            
            return {
                'transaction_count': len(transactions),
                'total_amount': total_amount,
                'unique_countries': unique_countries,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }

    def mark_transaction_as_fraud(
        self,
        transaction_id: str,
        reviewed_by: str,
        notes: Optional[str] = None
    ) -> None:
        """
        Mark a transaction as fraudulent.

        Args:
            transaction_id: Transaction identifier
            reviewed_by: Identifier of the reviewer
            notes: Optional review notes
        """
        with self.get_session() as session:
            # Update transaction
            transaction = session.query(Transaction)\
                .filter(Transaction.transaction_id == transaction_id)\
                .first()
            
            if transaction:
                transaction.is_fraud = True
                transaction.risk_level = 'high'
                
                # Update fraud analysis
                analysis = transaction.fraud_analysis
                if analysis:
                    analysis.final_action = 'block'
                    analysis.reviewed_by = reviewed_by
                    analysis.review_notes = notes
                    analysis.updated_at = datetime.utcnow()
                
                # Update user risk score
                user = transaction.user
                if user:
                    user.risk_score = min(1.0, user.risk_score + 0.2)
                
                session.commit() 