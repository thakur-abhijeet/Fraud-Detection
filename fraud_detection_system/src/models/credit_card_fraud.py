"""
Credit card fraud detection model.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

from .base_model import BaseFraudModel

class CreditCardFraudModel(BaseFraudModel):
    """Model for detecting credit card fraud."""

    def __init__(self):
        """Initialize credit card fraud model."""
        super().__init__("credit_card_fraud")
        self.scaler = StandardScaler()
        self.required_features = [
            'amount', 'time_since_last_tx', 'tx_per_day', 
            'avg_amount_30d', 'std_amount_30d', 'max_amount_30d',
            'different_countries_30d', 'risky_merchant_count_30d',
            'declined_tx_30d', 'card_present', 'international',
            'weekend', 'night_tx', 'high_risk_merchant',
            'high_risk_country', 'address_mismatch'
        ]

    def preprocess_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess transaction data for the model.

        Args:
            data: Raw transaction data

        Returns:
            Preprocessed features
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        features = pd.DataFrame()

        # Basic transaction features
        features['amount'] = data['amount']
        features['card_present'] = data.get('card_present', 0)
        features['international'] = (data['billing_country'] != data['shipping_country']).astype(int)
        
        # Temporal features
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            features['weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
            features['night_tx'] = ((timestamps.dt.hour >= 23) | (timestamps.dt.hour <= 4)).astype(int)

        # Location features
        high_risk_countries = self.config.fraud.high_risk_countries
        features['high_risk_country'] = data['billing_country'].isin(high_risk_countries).astype(int)
        features['address_mismatch'] = (data['billing_country'] != data['shipping_country']).astype(int)

        # Historical features (if available)
        if 'user_id' in data.columns:
            historical_features = self._calculate_historical_features(data)
            features = pd.concat([features, historical_features], axis=1)

        # Fill missing values
        features = features.fillna(0)

        # Scale features
        if self.model is not None:  # If model exists, we're in prediction mode
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns,
                index=features.index
            )

        return features

    def _calculate_historical_features(self, current_tx: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical features for transactions.

        Args:
            current_tx: Current transaction data

        Returns:
            Historical features
        """
        features = pd.DataFrame(index=current_tx.index)
        
        # These would typically come from a database in production
        # For now, we'll return placeholder values
        features['time_since_last_tx'] = 24.0  # hours
        features['tx_per_day'] = 2.5
        features['avg_amount_30d'] = 100.0
        features['std_amount_30d'] = 50.0
        features['max_amount_30d'] = 200.0
        features['different_countries_30d'] = 1
        features['risky_merchant_count_30d'] = 0
        features['declined_tx_30d'] = 0

        return features

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the credit card fraud detection model.

        Args:
            X: Training features
            y: Training labels (0 for legitimate, 1 for fraudulent)
            **kwargs: Additional training parameters
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            class_weight=kwargs.get('class_weight', 'balanced'),
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether transactions are fraudulent.

        Args:
            X: Transaction features

        Returns:
            Array of predictions (0 for legitimate, 1 for fraudulent)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get fraud probability scores.

        Args:
            X: Transaction features

        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive fraud analysis on a transaction.

        Args:
            transaction: Transaction data

        Returns:
            Analysis results including risk scores and explanations
        """
        # Preprocess transaction
        features = self.preprocess_data(transaction)
        
        # Get model prediction and probability
        fraud_probability = float(self.predict_proba(features)[0])
        
        # Get risk level based on thresholds
        if fraud_probability >= self.config.fraud.risk_thresholds['high_risk']:
            risk_level = 'high'
            recommended_action = 'block'
        elif fraud_probability >= self.config.fraud.risk_thresholds['medium_risk']:
            risk_level = 'medium'
            recommended_action = 'review'
        else:
            risk_level = 'low'
            recommended_action = 'allow'

        # Generate risk factors
        risk_factors = self._identify_risk_factors(features.iloc[0], fraud_probability)

        return {
            'transaction_id': transaction.get('transaction_id'),
            'fraud_probability': fraud_probability,
            'risk_level': risk_level,
            'recommended_action': recommended_action,
            'risk_factors': risk_factors,
            'timestamp': datetime.now().isoformat()
        }

    def _identify_risk_factors(self, features: pd.Series, fraud_probability: float) -> List[Dict[str, Any]]:
        """
        Identify specific risk factors contributing to fraud probability.

        Args:
            features: Transaction features
            fraud_probability: Overall fraud probability

        Returns:
            List of risk factors with descriptions and impact scores
        """
        risk_factors = []
        feature_impacts = self._calculate_feature_impacts(features)

        for feature, impact in feature_impacts.items():
            if impact > 0.1:  # Only include significant factors
                risk_factors.append({
                    'factor': feature,
                    'impact': impact,
                    'description': self._get_risk_factor_description(feature, features[feature])
                })

        return sorted(risk_factors, key=lambda x: x['impact'], reverse=True)

    def _calculate_feature_impacts(self, features: pd.Series) -> Dict[str, float]:
        """
        Calculate the impact of each feature on the fraud prediction.

        Args:
            features: Transaction features

        Returns:
            Dictionary of feature impacts
        """
        impacts = {}
        baseline_features = features.copy()
        baseline_prob = float(self.predict_proba(pd.DataFrame([baseline_features]))[0])

        for feature in features.index:
            # Calculate impact by zeroing out the feature
            modified_features = baseline_features.copy()
            modified_features[feature] = 0
            modified_prob = float(self.predict_proba(pd.DataFrame([modified_features]))[0])
            impacts[feature] = abs(baseline_prob - modified_prob)

        return impacts

    def _get_risk_factor_description(self, feature: str, value: float) -> str:
        """
        Get human-readable description of a risk factor.

        Args:
            feature: Feature name
            value: Feature value

        Returns:
            Description of the risk factor
        """
        descriptions = {
            'amount': 'Transaction amount is unusually high',
            'time_since_last_tx': 'Unusual transaction timing',
            'tx_per_day': 'High transaction frequency',
            'different_countries_30d': 'Multiple countries in recent transactions',
            'international': 'International transaction',
            'night_tx': 'Transaction during night hours',
            'high_risk_merchant': 'High-risk merchant category',
            'high_risk_country': 'Transaction from high-risk country',
            'address_mismatch': 'Billing and shipping address mismatch'
        }

        return descriptions.get(feature, f"Unusual pattern in {feature}") 