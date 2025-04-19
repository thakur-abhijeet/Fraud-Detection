"""
Friendly Fraud Detection Module for E-commerce

This module implements detection mechanisms for friendly fraud (chargeback fraud)
in e-commerce systems, including customer purchase history analysis, delivery
confirmation tracking, and chargeback pattern recognition.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import datetime
import json
from collections import defaultdict

class FriendlyFraudDetector:
    """
    A class for detecting friendly fraud (chargeback fraud) in e-commerce transactions.
    
    This class implements various methods for identifying potentially fraudulent
    chargebacks and customer-initiated fraud.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the FriendlyFraudDetector.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.chargeback_model = None
        self.customer_profiles = {}
        self.feature_importance = None
        
        # Thresholds for detection
        self.thresholds = {
            'chargeback_ratio_threshold': 0.05,  # Ratio of chargebacks to total orders
            'high_risk_chargeback_ratio': 0.1,   # High risk threshold
            'time_to_chargeback_days': 30,       # Typical time window for chargebacks
            'quick_chargeback_days': 7,          # Suspiciously quick chargeback
            'repeat_purchase_window_days': 90,   # Window to check for repeat purchases
            'digital_goods_risk': 0.7,           # Risk score for digital goods
            'high_value_threshold': 500,         # Threshold for high-value orders
            'new_customer_days': 30              # Days to consider a customer as new
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def set_thresholds(self, thresholds):
        """
        Set thresholds for detection mechanisms.
        
        Parameters:
        -----------
        thresholds : dict
            Dictionary of threshold values
            
        Returns:
        --------
        self : FriendlyFraudDetector
            Returns self for method chaining
        """
        self.thresholds.update(thresholds)
        return self
    
    def train_chargeback_model(self, X_train, y_train, model_type='random_forest', 
                              optimize_hyperparams=False, class_weight='balanced'):
        """
        Train a machine learning model for chargeback fraud detection.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (fraudulent=1, legitimate=0)
        model_type : str
            Type of model to train ('random_forest' or 'logistic_regression')
        optimize_hyperparams : bool
            Whether to optimize hyperparameters using grid search
        class_weight : str or dict
            Class weights to handle imbalanced data
            
        Returns:
        --------
        self : FriendlyFraudDetector
            Returns self for method chaining
        """
        if model_type == 'random_forest':
            if optimize_hyperparams:
                # Define parameter grid
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Create base model
                base_model = RandomForestClassifier(random_state=42, class_weight=class_weight)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                self.chargeback_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                # Create and train model with default parameters
                self.chargeback_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight=class_weight
                )
                self.chargeback_model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.chargeback_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif model_type == 'logistic_regression':
            if optimize_hyperparams:
                # Define parameter grid
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
                
                # Create base model
                base_model = LogisticRegression(random_state=42, class_weight=class_weight, max_iter=1000)
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                self.chargeback_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                # Create and train model with default parameters
                self.chargeback_model = LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    random_state=42,
                    class_weight=class_weight,
                    max_iter=1000
                )
                self.chargeback_model.fit(X_train, y_train)
            
            # Calculate feature importance (coefficients)
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.chargeback_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from 'random_forest' or 'logistic_regression'")
        
        return self
    
    def build_customer_profile(self, customer_id, transactions, chargebacks=None):
        """
        Build a profile for a customer based on their transaction and chargeback history.
        
        Parameters:
        -----------
        customer_id : str
            Customer identifier
        transactions : pd.DataFrame
            DataFrame with customer's transaction history
        chargebacks : pd.DataFrame
            DataFrame with customer's chargeback history
            
        Returns:
        --------
        dict
            Customer profile with behavioral patterns
        """
        if transactions.empty:
            return {
                'customer_id': customer_id,
                'transaction_count': 0,
                'chargeback_count': 0,
                'chargeback_ratio': 0.0,
                'avg_order_value': 0.0,
                'first_order_date': None,
                'last_order_date': None,
                'customer_age_days': 0,
                'product_categories': [],
                'payment_methods': [],
                'high_risk_score': 0.0
            }
        
        # Calculate basic transaction metrics
        transaction_count = len(transactions)
        
        # Calculate average order value
        if 'amount' in transactions.columns:
            avg_order_value = transactions['amount'].mean()
        else:
            avg_order_value = 0.0
        
        # Calculate customer age
        if 'timestamp' in transactions.columns:
            first_order_date = transactions['timestamp'].min()
            last_order_date = transactions['timestamp'].max()
            
            # Calculate customer age in days
            if isinstance(first_order_date, str):
                first_order_date = pd.to_datetime(first_order_date)
                
            if isinstance(last_order_date, str):
                last_order_date = pd.to_datetime(last_order_date)
                
            now = pd.Timestamp.now()
            customer_age_days = (now - first_order_date).days
        else:
            first_order_date = None
            last_order_date = None
            customer_age_days = 0
        
        # Extract product categories and payment methods
        product_categories = []
        if 'product_category' in transactions.columns:
            product_categories = transactions['product_category'].unique().tolist()
        
        payment_methods = []
        if 'payment_method' in transactions.columns:
            payment_methods = transactions['payment_method'].unique().tolist()
        
        # Calculate chargeback metrics
        chargeback_count = 0
        chargeback_ratio = 0.0
        
        if chargebacks is not None and not chargebacks.empty:
            chargeback_count = len(chargebacks)
            chargeback_ratio = chargeback_count / transaction_count
        
        # Calculate high risk score
        high_risk_score = 0.0
        
        # New customer risk
        if customer_age_days < self.thresholds['new_customer_days']:
            high_risk_score += 0.3
        
        # Chargeback ratio risk
        if chargeback_ratio > self.thresholds['high_risk_chargeback_ratio']:
            high_risk_score += 0.5
        elif chargeback_ratio > self.thresholds['chargeback_ratio_threshold']:
            high_risk_score += 0.3
        
        # Payment method risk
        high_risk_payment_methods = ['gift_card', 'prepaid_card', 'cryptocurrency']
        if any(method in high_risk_payment_methods for method in payment_methods):
            high_risk_score += 0.2
        
        # Cap the risk score at 1.0
        high_risk_score = min(1.0, high_risk_score)
        
        # Create customer profile
        profile = {
            'customer_id': customer_id,
            'transaction_count': transaction_count,
            'chargeback_count': chargeback_count,
            'chargeback_ratio': chargeback_ratio,
            'avg_order_value': avg_order_value,
            'first_order_date': first_order_date,
            'last_order_date': last_order_date,
            'customer_age_days': customer_age_days,
            'product_categories': product_categories,
            'payment_methods': payment_methods,
            'high_risk_score': high_risk_score
        }
        
        # Store the profile
        self.customer_profiles[customer_id] = profile
        
        return profile
    
    def analyze_chargeback(self, chargeback_data, transaction_data=None, customer_profile=None):
        """
        Analyze a chargeback for potential friendly fraud.
        
        Parameters:
        -----------
        chargeback_data : dict or pd.Series
            Data about the chargeback
        transaction_data : dict or pd.Series
            Data about the original transaction
        customer_profile : dict
            Customer's behavioral profile
            
        Returns:
        --------
        dict
            Analysis results with risk scores
        """
        # Convert data to Series if they're dicts
        if isinstance(chargeback_data, dict):
            chargeback_data = pd.Series(chargeback_data)
        
        if transaction_data is not None and isinstance(transaction_data, dict):
            transaction_data = pd.Series(transaction_data)
        
        # Initialize results
        results = {
            'chargeback_id': chargeback_data.get('chargeback_id', None),
            'transaction_id': chargeback_data.get('transaction_id', None),
            'customer_id': chargeback_data.get('user_id', None),
            'timestamp': chargeback_data.get('timestamp', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # If no customer profile is provided, try to get it from stored profiles
        if customer_profile is None and 'user_id' in chargeback_data:
            customer_profile = self.customer_profiles.get(chargeback_data['user_id'], None)
        
        # Check 1: Customer history analysis
        if customer_profile is not None:
            customer_risk = self.analyze_customer_history(customer_profile)
            results['checks']['customer_history'] = customer_risk
            results['risk_scores']['customer_history'] = customer_risk['risk_score']
        
        # Check 2: Time to chargeback analysis
        if transaction_data is not None and 'timestamp' in transaction_data and 'timestamp' in chargeback_data:
            time_risk = self.analyze_time_to_chargeback(chargeback_data, transaction_data)
            results['checks']['time_to_chargeback'] = time_risk
            results['risk_scores']['time_to_chargeback'] = time_risk['risk_score']
        
        # Check 3: Delivery confirmation analysis
        if transaction_data is not None and 'delivery_status' in transaction_data:
            delivery_risk = self.analyze_delivery_confirmation(transaction_data)
            results['checks']['delivery_confirmation'] = delivery_risk
            results['risk_scores']['delivery_confirmation'] = delivery_risk['risk_score']
        
        # Check 4: Product type analysis
        if transaction_data is not None and 'product_type' in transaction_data:
            product_risk = self.analyze_product_type(transaction_data)
            results['checks']['product_type'] = product_risk
            results['risk_scores']['product_type'] = product_risk['risk_score']
        
        # Check 5: Chargeback reason analysis
        if 'reason' in chargeback_data:
            reason_risk = self.analyze_chargeback_reason(chargeback_data)
            results['checks']['chargeback_reason'] = reason_risk
            results['risk_scores']['chargeback_reason'] = reason_risk['risk_score']
        
        # Check 6: Order value analysis
        if transaction_data is not None and 'amount' in transaction_data:
            value_risk = self.analyze_order_value(transaction_data, customer_profile)
            results['checks']['order_value'] = value_risk
            results['risk_scores']['order_value'] = value_risk['risk_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'customer_history': 0.25,
            'time_to_chargeback': 0.15,
            'delivery_confirmation': 0.2,
            'product_type': 0.1,
            'chargeback_reason': 0.2,
            'order_value': 0.1
        }
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for key, weight in weights.items():
            if key in results['risk_scores']:
                weighted_sum += results['risk_scores'][key] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            results['overall_risk_score'] = weighted_sum / weight_sum
        else:
            results['overall_risk_score'] = 0.0
        
        # Determine risk level
        if results['overall_risk_score'] >= 0.7:
            results['risk_level'] = 'high'
            results['recommended_action'] = 'investigate_and_challenge'
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
            results['recommended_action'] = 'collect_evidence'
        else:
            results['risk_level'] = 'low'
            results['recommended_action'] = 'accept'
        
        return results
    
    def analyze_customer_history(self, customer_profile):
        """
        Analyze customer history for chargeback patterns.
        
        Parameters:
        -----------
        customer_profile : dict
            Customer's behavioral profile
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Default risk for new customers or missing data
        if customer_profile is None:
            return {
                'chargeback_ratio': 0.0,
                'previous_chargebacks': 0,
                'is_new_customer': True,
                'risk_score': 0.5  # Moderate risk for unknown customers
            }
        
        # Extract relevant data from profile
        chargeback_ratio = customer_profile.get('chargeback_ratio', 0.0)
        chargeback_count = customer_profile.get('chargeback_count', 0)
        customer_age_days = customer_profile.get('customer_age_days', 0)
        
        # Determine if customer is new
        is_new_customer = customer_age_days < self.thresholds['new_customer_days']
        
        # Calculate risk score
        risk_score = 0.0
        
        # Risk based on chargeback ratio
        if chargeback_ratio >= self.thresholds['high_risk_chargeback_ratio']:
            risk_score += 0.6  # High risk
        elif chargeback_ratio >= self.thresholds['chargeback_ratio_threshold']:
            risk_score += 0.4  # Medium risk
        
        # Risk based on previous chargebacks
        if chargeback_count > 2:
            risk_score += 0.3  # Multiple previous chargebacks
        elif chargeback_count > 0:
            risk_score += 0.2  # At least one previous chargeback
        
        # Risk based on customer age
        if is_new_customer:
            risk_score += 0.2  # New customer risk
        
        # Cap the risk score at 1.0
        risk_score = min(1.0, risk_score)
        
        return {
            'chargeback_ratio': chargeback_ratio,
            'previous_chargebacks': chargeback_count,
            'is_new_customer': is_new_customer,
            'customer_age_days': customer_age_days,
            'risk_score': risk_score
        }
    
    def analyze_time_to_chargeback(self, chargeback_data, transaction_data):
        """
        Analyze the time between transaction and chargeback.
        
        Parameters:
        -----------
        chargeback_data : pd.Series
            Data about the chargeback
        transaction_data : pd.Series
            Data about the original transaction
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Convert timestamps to datetime if they're strings
        if isinstance(chargeback_data['timestamp'], str):
            chargeback_time = pd.to_datetime(chargeback_data['timestamp'])
        else:
            chargeback_time = chargeback_data['timestamp']
        
        if isinstance(transaction_data['timestamp'], str):
            transaction_time = pd.to_datetime(transaction_data['timestamp'])
        else:
            transaction_time = transaction_data['timestamp']
        
        # Calculate time difference in days
        time_to_chargeback = (chargeback_time - transaction_time).total_seconds() / (24 * 3600)
        
        # Determine if time to chargeback is suspicious
        quick_chargeback = time_to_chargeback < self.thresholds['quick_chargeback_days']
        late_chargeback = time_to_chargeback > self.thresholds['time_to_chargeback_days']
        
        # Calculate risk score
        if quick_chargeback:
            # Very quick chargebacks can be suspicious
            risk_score = 0.7
        elif late_chargeback:
            # Late chargebacks can also be suspicious
            risk_score = 0.6
        else:
            # Normal time frame
            risk_score = 0.3
        
        return {
            'time_to_chargeback_days': time_to_chargeback,
            'quick_chargeback': quick_chargeback,
            'late_chargeback': late_chargeback,
            'risk_score': risk_score
        }
    
    def analyze_delivery_confirmation(self, transaction_data):
        """
        Analyze delivery confirmation status.
        
        Parameters:
        -----------
        transaction_data : pd.Series
            Data about the original transaction
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract delivery status
        delivery_status = transaction_data.get('delivery_status', 'unknown')
        
        # Extract delivery confirmation data if available
        delivery_confirmation = transaction_data.get('delivery_confirmation', None)
        delivery_signature = transaction_data.get('delivery_signature', None)
        delivery_date = transaction_data.get('delivery_date', None)
        
        # Determine if delivery is confirmed
        delivery_confirmed = delivery_status in ['delivered', 'received']
        has_confirmation = delivery_confirmation is not None
        has_signature = delivery_signature is not None
        
        # Calculate risk score
        if delivery_confirmed and has_signature:
            risk_score = 0.1  # Very low risk - confirmed delivery with signature
        elif delivery_confirmed and has_confirmation:
            risk_score = 0.2  # Low risk - confirmed delivery with some evidence
        elif delivery_confirmed:
            risk_score = 0.3  # Low-medium risk - delivery marked as confirmed but no evidence
        elif delivery_status == 'in_transit':
            risk_score = 0.5  # Medium risk - item still in transit
        elif delivery_status == 'unknown':
            risk_score = 0.6  # Medium-high risk - unknown delivery status
        else:
            risk_score = 0.4  # Medium risk - other status
        
        return {
            'delivery_status': delivery_status,
            'delivery_confirmed': delivery_confirmed,
            'has_confirmation': has_confirmation,
            'has_signature': has_signature,
            'delivery_date': delivery_date,
            'risk_score': risk_score
        }
    
    def analyze_product_type(self, transaction_data):
        """
        Analyze product type for chargeback risk.
        
        Parameters:
        -----------
        transaction_data : pd.Series
            Data about the original transaction
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract product type
        product_type = transaction_data.get('product_type', 'unknown')
        product_category = transaction_data.get('product_category', 'unknown')
        
        # Determine if product is digital
        is_digital = product_type in ['digital', 'download', 'subscription', 'virtual']
        
        # Determine if product is high-risk category
        high_risk_categories = ['electronics', 'jewelry', 'luxury', 'gift_card']
        is_high_risk_category = product_category in high_risk_categories
        
        # Calculate risk score
        if is_digital:
            risk_score = self.thresholds['digital_goods_risk']  # Digital goods have higher risk
        elif is_high_risk_category:
            risk_score = 0.6  # High-risk physical product categories
        else:
            risk_score = 0.3  # Standard physical products
        
        return {
            'product_type': product_type,
            'product_category': product_category,
            'is_digital': is_digital,
            'is_high_risk_category': is_high_risk_category,
            'risk_score': risk_score
        }
    
    def analyze_chargeback_reason(self, chargeback_data):
        """
        Analyze chargeback reason for fraud indicators.
        
        Parameters:
        -----------
        chargeback_data : pd.Series
            Data about the chargeback
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract chargeback reason
        reason = chargeback_data.get('reason', 'unknown')
        
        # Define risk levels for different reasons
        reason_risk = {
            'unauthorized': 0.8,         # Claims they didn't make the purchase
            'product_not_received': 0.6, # Claims they didn't receive the item
            'not_as_described': 0.5,     # Claims item wasn't as described
            'duplicate': 0.4,            # Claims they were charged twice
            'credit_not_processed': 0.3, # Claims refund wasn't processed
            'general': 0.5,              # General dissatisfaction
            'unknown': 0.5               # Unknown reason
        }
        
        # Get risk score for this reason
        risk_score = reason_risk.get(reason, 0.5)
        
        # Determine if reason is high risk
        high_risk_reason = risk_score >= 0.6
        
        return {
            'reason': reason,
            'high_risk_reason': high_risk_reason,
            'risk_score': risk_score
        }
    
    def analyze_order_value(self, transaction_data, customer_profile=None):
        """
        Analyze order value for anomalies.
        
        Parameters:
        -----------
        transaction_data : pd.Series
            Data about the original transaction
        customer_profile : dict
            Customer's behavioral profile
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract order amount
        amount = transaction_data.get('amount', 0)
        
        # Determine if order is high value
        high_value = amount > self.thresholds['high_value_threshold']
        
        # Calculate risk score
        risk_score = 0.0
        
        # Base risk on order value
        if high_value:
            risk_score += 0.4  # Higher risk for high-value orders
        
        # Compare to customer's average order value if profile available
        if customer_profile is not None and 'avg_order_value' in customer_profile:
            avg_order_value = customer_profile['avg_order_value']
            
            if avg_order_value > 0:
                # Calculate ratio of current order to average
                value_ratio = amount / avg_order_value
                
                if value_ratio > 3:
                    risk_score += 0.4  # Much higher than usual
                elif value_ratio > 2:
                    risk_score += 0.2  # Moderately higher than usual
            
            # Include average order value in results
            avg_order_value_result = avg_order_value
        else:
            avg_order_value_result = None
            value_ratio = None
        
        # Cap the risk score at 1.0
        risk_score = min(1.0, risk_score)
        
        return {
            'amount': amount,
            'high_value': high_value,
            'avg_order_value': avg_order_value_result,
            'value_ratio': value_ratio,
            'risk_score': risk_score
        }
    
    def check_repeat_purchase_patterns(self, customer_id, transactions, chargebacks):
        """
        Check for repeat purchase patterns that may indicate friendly fraud.
        
        Parameters:
        -----------
        customer_id : str
            Customer identifier
        transactions : pd.DataFrame
            DataFrame with customer's transaction history
        chargebacks : pd.DataFrame
            DataFrame with customer's chargeback history
            
        Returns:
        --------
        dict
            Analysis results with risk indicators
        """
        if transactions.empty or chargebacks.empty:
            return {
                'has_repeat_pattern': False,
                'pattern_count': 0,
                'risk_score': 0.1
            }
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(transactions['timestamp'].iloc[0]):
            transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        
        if not pd.api.types.is_datetime64_any_dtype(chargebacks['timestamp'].iloc[0]):
            chargebacks['timestamp'] = pd.to_datetime(chargebacks['timestamp'])
        
        # Get transaction IDs with chargebacks
        chargeback_tx_ids = chargebacks['transaction_id'].unique()
        
        # Filter transactions with chargebacks
        chargeback_txs = transactions[transactions['transaction_id'].isin(chargeback_tx_ids)]
        
        # Initialize pattern counter
        pattern_count = 0
        
        # Check for repeat purchases of same product
        if 'product_id' in transactions.columns:
            # Group transactions by product
            product_txs = transactions.groupby('product_id')
            
            for product_id, product_group in product_txs:
                if len(product_group) > 1:
                    # Check if any transaction for this product had a chargeback
                    product_chargebacks = product_group[product_group['transaction_id'].isin(chargeback_tx_ids)]
                    
                    if not product_chargebacks.empty:
                        # Get the chargeback transaction
                        chargeback_tx = product_chargebacks.iloc[0]
                        
                        # Check for purchases of the same product after chargeback
                        for _, tx in product_group.iterrows():
                            # Skip the chargeback transaction itself
                            if tx['transaction_id'] == chargeback_tx['transaction_id']:
                                continue
                            
                            # Check if purchase was after chargeback
                            time_diff = (tx['timestamp'] - chargeback_tx['timestamp']).total_seconds() / (24 * 3600)
                            
                            if 0 < time_diff < self.thresholds['repeat_purchase_window_days']:
                                pattern_count += 1
        
        # Check for repeat purchases in same category
        if 'product_category' in transactions.columns:
            # Group transactions by category
            category_txs = transactions.groupby('product_category')
            
            for category, category_group in category_txs:
                if len(category_group) > 1:
                    # Check if any transaction in this category had a chargeback
                    category_chargebacks = category_group[category_group['transaction_id'].isin(chargeback_tx_ids)]
                    
                    if not category_chargebacks.empty:
                        # Get the chargeback transaction
                        chargeback_tx = category_chargebacks.iloc[0]
                        
                        # Check for purchases in the same category after chargeback
                        for _, tx in category_group.iterrows():
                            # Skip the chargeback transaction itself
                            if tx['transaction_id'] == chargeback_tx['transaction_id']:
                                continue
                            
                            # Check if purchase was after chargeback
                            time_diff = (tx['timestamp'] - chargeback_tx['timestamp']).total_seconds() / (24 * 3600)
                            
                            if 0 < time_diff < self.thresholds['repeat_purchase_window_days']:
                                pattern_count += 1
        
        # Calculate risk score based on pattern count
        if pattern_count > 2:
            risk_score = 0.9  # Very high risk
        elif pattern_count > 0:
            risk_score = 0.7  # High risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'has_repeat_pattern': pattern_count > 0,
            'pattern_count': pattern_count,
            'risk_score': risk_score
        }
    
    def collect_digital_evidence(self, transaction_data):
        """
        Collect digital evidence that can be used to dispute a chargeback.
        
        Parameters:
        -----------
        transaction_data : pd.Series
            Data about the original transaction
            
        Returns:
        --------
        dict
            Dictionary with collected evidence
        """
        evidence = {
            'transaction_id': transaction_data.get('transaction_id', None),
            'evidence_items': []
        }
        
        # IP address evidence
        if 'ip_address' in transaction_data:
            evidence['evidence_items'].append({
                'type': 'ip_address',
                'value': transaction_data['ip_address'],
                'description': 'IP address used for the transaction'
            })
        
        # Device evidence
        if 'device_id' in transaction_data:
            evidence['evidence_items'].append({
                'type': 'device_id',
                'value': transaction_data['device_id'],
                'description': 'Device used for the transaction'
            })
        
        # Delivery confirmation
        if 'delivery_confirmation' in transaction_data:
            evidence['evidence_items'].append({
                'type': 'delivery_confirmation',
                'value': transaction_data['delivery_confirmation'],
                'description': 'Delivery confirmation details'
            })
        
        if 'delivery_signature' in transaction_data:
            evidence['evidence_items'].append({
                'type': 'delivery_signature',
                'value': transaction_data['delivery_signature'],
                'description': 'Delivery signature'
            })
        
        # Digital product access logs
        if 'product_type' in transaction_data and transaction_data['product_type'] in ['digital', 'download', 'subscription']:
            if 'access_logs' in transaction_data:
                evidence['evidence_items'].append({
                    'type': 'access_logs',
                    'value': transaction_data['access_logs'],
                    'description': 'Logs showing access to the digital product'
                })
        
        # User activity logs
        if 'user_activity' in transaction_data:
            evidence['evidence_items'].append({
                'type': 'user_activity',
                'value': transaction_data['user_activity'],
                'description': 'User activity related to the transaction'
            })
        
        # Previous successful transactions
        if 'previous_transactions' in transaction_data:
            evidence['evidence_items'].append({
                'type': 'previous_transactions',
                'value': transaction_data['previous_transactions'],
                'description': 'History of previous successful transactions'
            })
        
        return evidence
    
    def predict_chargeback_risk(self, transaction_features):
        """
        Predict the risk of chargeback fraud for a transaction using the ML model.
        
        Parameters:
        -----------
        transaction_features : pd.DataFrame
            Features of the transaction to check
            
        Returns:
        --------
        float
            Risk score (0-1)
        """
        if self.chargeback_model is None:
            raise ValueError("Model not trained. Call train_chargeback_model() first.")
        
        # Ensure transaction_features is a DataFrame
        if isinstance(transaction_features, pd.Series):
            transaction_features = pd.DataFrame([transaction_features])
        
        # Predict probability
        risk_score = self.chargeback_model.predict_proba(transaction_features)[0, 1]
        
        return risk_score
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the chargeback model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels (fraudulent=1, legitimate=0)
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.chargeback_model is None:
            raise ValueError("Model not trained. Call train_chargeback_model() first.")
        
        # Make predictions
        y_pred = self.chargeback_model.predict(X_test)
        y_pred_proba = self.chargeback_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Detection rate (recall)
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 score
        f1_score = 2 * (precision * detection_rate) / (precision + detection_rate) if (precision + detection_rate) > 0 else 0
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'auc_score': auc_score,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'f1_score': f1_score
        }
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        save_path : str
            Path to save the plot (if None, plot is displayed)
            
        Returns:
        --------
        None
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train a model first.")
        
        # Get top N features
        top_features = self.feature_importance.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Features for Friendly Fraud Detection')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_model(self, filename='friendly_fraud_model.pkl'):
        """
        Save the trained model and configuration to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'chargeback_model': self.chargeback_model,
            'feature_importance': self.feature_importance,
            'thresholds': self.thresholds,
            'customer_profiles': self.customer_profiles
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='friendly_fraud_model.pkl'):
        """
        Load trained model and configuration from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
            
        Returns:
        --------
        self : FriendlyFraudDetector
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.chargeback_model = model_data['chargeback_model']
        self.feature_importance = model_data['feature_importance']
        self.thresholds = model_data['thresholds']
        self.customer_profiles = model_data['customer_profiles']
        
        return self
    
    def extract_chargeback_features(self, transaction_data, customer_profile=None, chargeback_data=None):
        """
        Extract features for chargeback risk prediction.
        
        Parameters:
        -----------
        transaction_data : pd.Series
            Data about the transaction
        customer_profile : dict
            Customer's behavioral profile
        chargeback_data : pd.Series
            Data about the chargeback (if available)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted features
        """
        # Convert transaction_data to Series if it's a dict
        if isinstance(transaction_data, dict):
            transaction_data = pd.Series(transaction_data)
        
        # Initialize features dictionary
        features = {}
        
        # Extract transaction features
        if 'amount' in transaction_data:
            features['amount'] = transaction_data['amount']
        
        if 'product_type' in transaction_data:
            features['is_digital'] = 1 if transaction_data['product_type'] in ['digital', 'download', 'subscription', 'virtual'] else 0
        
        if 'product_category' in transaction_data:
            high_risk_categories = ['electronics', 'jewelry', 'luxury', 'gift_card']
            features['high_risk_category'] = 1 if transaction_data['product_category'] in high_risk_categories else 0
        
        if 'payment_method' in transaction_data:
            high_risk_methods = ['gift_card', 'prepaid_card', 'cryptocurrency']
            features['high_risk_payment'] = 1 if transaction_data['payment_method'] in high_risk_methods else 0
        
        if 'delivery_status' in transaction_data:
            features['delivery_confirmed'] = 1 if transaction_data['delivery_status'] in ['delivered', 'received'] else 0
        
        # Extract customer features if profile is available
        if customer_profile is not None:
            features['customer_age_days'] = customer_profile.get('customer_age_days', 0)
            features['transaction_count'] = customer_profile.get('transaction_count', 0)
            features['chargeback_count'] = customer_profile.get('chargeback_count', 0)
            features['chargeback_ratio'] = customer_profile.get('chargeback_ratio', 0.0)
            features['avg_order_value'] = customer_profile.get('avg_order_value', 0.0)
            
            # Calculate value ratio
            if features['avg_order_value'] > 0 and 'amount' in features:
                features['value_ratio'] = features['amount'] / features['avg_order_value']
            else:
                features['value_ratio'] = 1.0
            
            # New customer flag
            features['is_new_customer'] = 1 if features['customer_age_days'] < self.thresholds['new_customer_days'] else 0
        
        # Extract chargeback features if available
        if chargeback_data is not None:
            if isinstance(chargeback_data, dict):
                chargeback_data = pd.Series(chargeback_data)
            
            if 'reason' in chargeback_data:
                # One-hot encode reason
                reason = chargeback_data['reason']
                features['reason_unauthorized'] = 1 if reason == 'unauthorized' else 0
                features['reason_not_received'] = 1 if reason == 'product_not_received' else 0
                features['reason_not_as_described'] = 1 if reason == 'not_as_described' else 0
                features['reason_duplicate'] = 1 if reason == 'duplicate' else 0
                features['reason_credit_not_processed'] = 1 if reason == 'credit_not_processed' else 0
            
            # Calculate time to chargeback if timestamps available
            if 'timestamp' in chargeback_data and 'timestamp' in transaction_data:
                # Convert timestamps to datetime if they're strings
                if isinstance(chargeback_data['timestamp'], str):
                    chargeback_time = pd.to_datetime(chargeback_data['timestamp'])
                else:
                    chargeback_time = chargeback_data['timestamp']
                
                if isinstance(transaction_data['timestamp'], str):
                    transaction_time = pd.to_datetime(transaction_data['timestamp'])
                else:
                    transaction_time = transaction_data['timestamp']
                
                # Calculate time difference in days
                time_to_chargeback = (chargeback_time - transaction_time).total_seconds() / (24 * 3600)
                features['time_to_chargeback_days'] = time_to_chargeback
                
                # Quick chargeback flag
                features['quick_chargeback'] = 1 if time_to_chargeback < self.thresholds['quick_chargeback_days'] else 0
                
                # Late chargeback flag
                features['late_chargeback'] = 1 if time_to_chargeback > self.thresholds['time_to_chargeback_days'] else 0
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
    
    def create_sample_data(self, n_customers=50, n_transactions=1000, n_chargebacks=100, 
                          fraudulent_ratio=0.3, save_to_csv=True):
        """
        Create sample data for demonstration purposes.
        
        Parameters:
        -----------
        n_customers : int
            Number of customers to generate
        n_transactions : int
            Number of transactions to generate
        n_chargebacks : int
            Number of chargebacks to generate
        fraudulent_ratio : float
            Proportion of fraudulent chargebacks
        save_to_csv : bool
            Whether to save the generated data to CSV files
            
        Returns:
        --------
        tuple
            (transactions_df, customers_df, chargebacks_df) DataFrames
        """
        np.random.seed(42)
        
        # Generate customer data
        customer_ids = [f"U{i:05d}" for i in range(1, n_customers + 1)]
        
        customers_data = {
            'user_id': customer_ids,
            'name': [f"Customer {i}" for i in range(1, n_customers + 1)],
            'email': [f"customer{i}@example.com" for i in range(1, n_customers + 1)],
            'registration_date': pd.date_range(start='2020-01-01', periods=n_customers, freq='D'),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France'], n_customers),
            'phone': [f"+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_customers)]
        }
        
        customers_df = pd.DataFrame(customers_data)
        
        # Generate transaction data
        transaction_ids = [f"T{i:06d}" for i in range(1, n_transactions + 1)]
        
        # Define product categories and types
        product_categories = ['electronics', 'clothing', 'home', 'books', 'jewelry', 'digital', 'gift_card']
        product_types = ['physical', 'digital']
        delivery_statuses = ['delivered', 'in_transit', 'processing', 'unknown']
        payment_methods = ['credit_card', 'debit_card', 'digital_wallet', 'gift_card', 'bank_transfer']
        
        transactions_data = {
            'transaction_id': transaction_ids,
            'user_id': np.random.choice(customer_ids, n_transactions),
            'timestamp': pd.date_range(start='2023-01-01', periods=n_transactions, freq='30min'),
            'amount': np.random.uniform(10, 1000, n_transactions).round(2),
            'product_category': np.random.choice(product_categories, n_transactions),
            'product_type': np.random.choice(product_types, n_transactions, p=[0.8, 0.2]),
            'payment_method': np.random.choice(payment_methods, n_transactions),
            'delivery_status': np.random.choice(delivery_statuses, n_transactions, p=[0.7, 0.2, 0.05, 0.05])
        }
        
        # Add delivery confirmation for delivered items
        delivery_confirmation = []
        delivery_signature = []
        
        for status in transactions_data['delivery_status']:
            if status == 'delivered':
                delivery_confirmation.append(f"CONF-{np.random.randint(10000, 99999)}")
                # 70% chance of having a signature
                if np.random.random() < 0.7:
                    delivery_signature.append(f"SIG-{np.random.randint(10000, 99999)}")
                else:
                    delivery_signature.append(None)
            else:
                delivery_confirmation.append(None)
                delivery_signature.append(None)
        
        transactions_data['delivery_confirmation'] = delivery_confirmation
        transactions_data['delivery_signature'] = delivery_signature
        
        # Create DataFrame
        transactions_df = pd.DataFrame(transactions_data)
        
        # Generate chargeback data
        # Select random transactions for chargebacks
        chargeback_tx_indices = np.random.choice(range(n_transactions), n_chargebacks, replace=False)
        chargeback_transactions = transactions_df.iloc[chargeback_tx_indices]
        
        # Determine which chargebacks are fraudulent
        n_fraudulent = int(n_chargebacks * fraudulent_ratio)
        is_fraudulent = np.zeros(n_chargebacks, dtype=bool)
        is_fraudulent[:n_fraudulent] = True
        np.random.shuffle(is_fraudulent)
        
        # Define chargeback reasons
        legitimate_reasons = ['product_not_received', 'not_as_described', 'duplicate', 'credit_not_processed']
        fraudulent_reasons = ['unauthorized', 'product_not_received', 'not_as_described']
        
        chargeback_ids = [f"C{i:05d}" for i in range(1, n_chargebacks + 1)]
        chargeback_reasons = []
        chargeback_timestamps = []
        
        for i, (_, tx) in enumerate(chargeback_transactions.iterrows()):
            if is_fraudulent[i]:
                reason = np.random.choice(fraudulent_reasons)
                
                # Fraudulent chargebacks often happen quickly or very late
                if np.random.random() < 0.5:
                    # Quick chargeback
                    days_to_chargeback = np.random.randint(1, 7)
                else:
                    # Late chargeback
                    days_to_chargeback = np.random.randint(30, 90)
            else:
                reason = np.random.choice(legitimate_reasons)
                # Normal timeframe for legitimate chargebacks
                days_to_chargeback = np.random.randint(7, 30)
            
            chargeback_reasons.append(reason)
            
            # Calculate chargeback timestamp
            tx_time = tx['timestamp']
            chargeback_time = tx_time + pd.Timedelta(days=days_to_chargeback)
            chargeback_timestamps.append(chargeback_time)
        
        chargebacks_data = {
            'chargeback_id': chargeback_ids,
            'transaction_id': chargeback_transactions['transaction_id'].values,
            'user_id': chargeback_transactions['user_id'].values,
            'timestamp': chargeback_timestamps,
            'reason': chargeback_reasons,
            'amount': chargeback_transactions['amount'].values,
            'status': np.random.choice(['pending', 'approved', 'rejected'], n_chargebacks),
            'is_fraudulent': is_fraudulent
        }
        
        chargebacks_df = pd.DataFrame(chargebacks_data)
        
        # Save to CSV if requested
        if save_to_csv:
            data_dir = os.path.dirname(self.model_dir)
            data_dir = os.path.join(data_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            transactions_df.to_csv(os.path.join(data_dir, 'sample_transactions.csv'), index=False)
            customers_df.to_csv(os.path.join(data_dir, 'sample_customers.csv'), index=False)
            chargebacks_df.to_csv(os.path.join(data_dir, 'sample_chargebacks.csv'), index=False)
        
        return transactions_df, customers_df, chargebacks_df
