"""
Credit Card Fraud Detection Module for E-commerce

This module implements detection mechanisms for credit card fraud in e-commerce
transactions, including pattern analysis, velocity checks, and anomaly detection.
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

class CreditCardFraudDetector:
    """
    A class for detecting credit card fraud in e-commerce transactions.
    
    This class implements various methods for identifying potentially fraudulent
    credit card transactions, including machine learning models and rule-based approaches.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the CreditCardFraudDetector.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.ml_model = None
        self.anomaly_detector = None
        self.feature_importance = None
        self.rules = []
        self.velocity_thresholds = {
            'amount_1h': 1000,  # Max amount in 1 hour
            'count_1h': 3,      # Max transactions in 1 hour
            'countries_1h': 2,  # Max different countries in 1 hour
            'amount_24h': 2000, # Max amount in 24 hours
            'count_24h': 10,    # Max transactions in 24 hours
            'countries_24h': 3  # Max different countries in 24 hours
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def add_rule(self, rule_function, rule_name, rule_description):
        """
        Add a rule-based detection function.
        
        Parameters:
        -----------
        rule_function : function
            Function that takes a transaction and returns a risk score (0-1)
        rule_name : str
            Name of the rule
        rule_description : str
            Description of what the rule checks for
            
        Returns:
        --------
        self : CreditCardFraudDetector
            Returns self for method chaining
        """
        self.rules.append({
            'function': rule_function,
            'name': rule_name,
            'description': rule_description
        })
        
        return self
    
    def set_velocity_thresholds(self, thresholds):
        """
        Set thresholds for velocity checks.
        
        Parameters:
        -----------
        thresholds : dict
            Dictionary of threshold values for velocity checks
            
        Returns:
        --------
        self : CreditCardFraudDetector
            Returns self for method chaining
        """
        self.velocity_thresholds.update(thresholds)
        return self
    
    def train_ml_model(self, X_train, y_train, model_type='random_forest', 
                      optimize_hyperparams=False, class_weight='balanced'):
        """
        Train a machine learning model for credit card fraud detection.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (fraud=1, legitimate=0)
        model_type : str
            Type of model to train ('random_forest' or 'logistic_regression')
        optimize_hyperparams : bool
            Whether to optimize hyperparameters using grid search
        class_weight : str or dict
            Class weights to handle imbalanced data
            
        Returns:
        --------
        self : CreditCardFraudDetector
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
                self.ml_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                # Create and train model with default parameters
                self.ml_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight=class_weight
                )
                self.ml_model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.ml_model.feature_importances_
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
                self.ml_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                # Create and train model with default parameters
                self.ml_model = LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    random_state=42,
                    class_weight=class_weight,
                    max_iter=1000
                )
                self.ml_model.fit(X_train, y_train)
            
            # Calculate feature importance (coefficients)
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.ml_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from 'random_forest' or 'logistic_regression'")
        
        return self
    
    def train_anomaly_detector(self, X_train, contamination=0.05):
        """
        Train an anomaly detection model for identifying unusual transactions.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        contamination : float
            Expected proportion of outliers in the data
            
        Returns:
        --------
        self : CreditCardFraudDetector
            Returns self for method chaining
        """
        # Create and train isolation forest model
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        
        self.anomaly_detector.fit(X_train)
        
        return self
    
    def evaluate_ml_model(self, X_test, y_test):
        """
        Evaluate the machine learning model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels (fraud=1, legitimate=0)
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.ml_model is None:
            raise ValueError("Model not trained. Call train_ml_model() first.")
        
        # Make predictions
        y_pred = self.ml_model.predict(X_test)
        y_pred_proba = self.ml_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate additional fraud-specific metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Fraud detection rate (recall)
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 score
        f1_score = 2 * (precision * fraud_detection_rate) / (precision + fraud_detection_rate) if (precision + fraud_detection_rate) > 0 else 0
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'auc_score': auc_score,
            'fraud_detection_rate': fraud_detection_rate,
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
        plt.title(f'Top {top_n} Features for Credit Card Fraud Detection')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def check_transaction_velocity(self, transaction, user_transactions):
        """
        Check transaction velocity for a user to detect unusual patterns.
        
        Parameters:
        -----------
        transaction : dict or pd.Series
            Current transaction to check
        user_transactions : pd.DataFrame
            Previous transactions for the same user
            
        Returns:
        --------
        dict
            Dictionary with velocity check results and risk scores
        """
        # Convert transaction to Series if it's a dict
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(transaction['timestamp']):
            transaction_time = pd.to_datetime(transaction['timestamp'])
        else:
            transaction_time = transaction['timestamp']
        
        # Ensure user_transactions timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(user_transactions['timestamp'].iloc[0]):
            user_transactions['timestamp'] = pd.to_datetime(user_transactions['timestamp'])
        
        # Calculate time windows
        one_hour_ago = transaction_time - pd.Timedelta(hours=1)
        one_day_ago = transaction_time - pd.Timedelta(days=1)
        
        # Get transactions in the last hour and day
        transactions_1h = user_transactions[user_transactions['timestamp'] >= one_hour_ago]
        transactions_24h = user_transactions[user_transactions['timestamp'] >= one_day_ago]
        
        # Calculate velocity metrics
        amount_1h = transactions_1h['amount'].sum() + transaction['amount']
        count_1h = len(transactions_1h) + 1
        countries_1h = len(set(transactions_1h['billing_country'].tolist() + [transaction['billing_country']]))
        
        amount_24h = transactions_24h['amount'].sum() + transaction['amount']
        count_24h = len(transactions_24h) + 1
        countries_24h = len(set(transactions_24h['billing_country'].tolist() + [transaction['billing_country']]))
        
        # Check against thresholds
        velocity_results = {
            'amount_1h': {
                'value': amount_1h,
                'threshold': self.velocity_thresholds['amount_1h'],
                'exceeded': amount_1h > self.velocity_thresholds['amount_1h']
            },
            'count_1h': {
                'value': count_1h,
                'threshold': self.velocity_thresholds['count_1h'],
                'exceeded': count_1h > self.velocity_thresholds['count_1h']
            },
            'countries_1h': {
                'value': countries_1h,
                'threshold': self.velocity_thresholds['countries_1h'],
                'exceeded': countries_1h > self.velocity_thresholds['countries_1h']
            },
            'amount_24h': {
                'value': amount_24h,
                'threshold': self.velocity_thresholds['amount_24h'],
                'exceeded': amount_24h > self.velocity_thresholds['amount_24h']
            },
            'count_24h': {
                'value': count_24h,
                'threshold': self.velocity_thresholds['count_24h'],
                'exceeded': count_24h > self.velocity_thresholds['count_24h']
            },
            'countries_24h': {
                'value': countries_24h,
                'threshold': self.velocity_thresholds['countries_24h'],
                'exceeded': countries_24h > self.velocity_thresholds['countries_24h']
            }
        }
        
        # Calculate overall velocity risk score (0-1)
        exceeded_count = sum(1 for check in velocity_results.values() if check['exceeded'])
        velocity_risk_score = exceeded_count / len(velocity_results)
        
        return {
            'velocity_checks': velocity_results,
            'velocity_risk_score': velocity_risk_score
        }
    
    def check_bin_country(self, card_bin, country, bin_country_db=None):
        """
        Check if the card BIN (first 6 digits) matches the expected country.
        
        Parameters:
        -----------
        card_bin : str
            First 6 digits of the card number
        country : str
            Country code of the billing address
        bin_country_db : dict
            Dictionary mapping BIN ranges to countries (if None, uses a simple example)
            
        Returns:
        --------
        dict
            Dictionary with BIN check results and risk score
        """
        # Simple example BIN database (in a real system, this would be comprehensive)
        if bin_country_db is None:
            bin_country_db = {
                '4': 'USA',       # Visa
                '51': 'USA',      # Mastercard
                '52': 'USA',      # Mastercard
                '53': 'USA',      # Mastercard
                '54': 'USA',      # Mastercard
                '55': 'USA',      # Mastercard
                '34': 'USA',      # Amex
                '37': 'USA',      # Amex
                '6011': 'USA',    # Discover
                '65': 'USA',      # Discover
                '5019': 'UK',     # Dankort
                '4571': 'UK',     # Visa UK
                '4576': 'Spain',  # Visa Spain
                '4' : 'USA'       # Default Visa
            }
        
        # Find the matching BIN prefix
        matching_bin = None
        for bin_prefix in sorted(bin_country_db.keys(), key=len, reverse=True):
            if card_bin.startswith(bin_prefix):
                matching_bin = bin_prefix
                break
        
        if matching_bin is None:
            return {
                'bin_check': 'unknown',
                'expected_country': None,
                'actual_country': country,
                'bin_risk_score': 0.5  # Medium risk for unknown BIN
            }
        
        expected_country = bin_country_db[matching_bin]
        match = expected_country == country
        
        return {
            'bin_check': 'match' if match else 'mismatch',
            'expected_country': expected_country,
            'actual_country': country,
            'bin_risk_score': 0.0 if match else 0.8  # High risk for mismatch
        }
    
    def check_address_verification(self, transaction):
        """
        Simulate an Address Verification System (AVS) check.
        
        Parameters:
        -----------
        transaction : dict or pd.Series
            Transaction data containing address information
            
        Returns:
        --------
        dict
            Dictionary with AVS check results and risk score
        """
        # Convert transaction to Series if it's a dict
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        
        # In a real system, this would call an actual AVS service
        # For this example, we'll simulate results based on available data
        
        # Check for address mismatches
        address_mismatch = False
        country_mismatch = False
        zip_mismatch = False
        
        if 'billing_country' in transaction and 'shipping_country' in transaction:
            country_mismatch = transaction['billing_country'] != transaction['shipping_country']
        
        if 'billing_zip' in transaction and 'shipping_zip' in transaction:
            zip_mismatch = transaction['billing_zip'] != transaction['shipping_zip']
        
        if 'billing_address' in transaction and 'shipping_address' in transaction:
            address_mismatch = transaction['billing_address'] != transaction['shipping_address']
        
        # Determine AVS code (simplified version)
        if country_mismatch:
            avs_code = 'N'  # No match
            avs_description = 'Country mismatch'
            risk_score = 0.8
        elif zip_mismatch:
            avs_code = 'Z'  # ZIP match only
            avs_description = 'ZIP mismatch'
            risk_score = 0.6
        elif address_mismatch:
            avs_code = 'A'  # Address match only
            avs_description = 'Address mismatch'
            risk_score = 0.4
        else:
            avs_code = 'Y'  # Match
            avs_description = 'Full match'
            risk_score = 0.0
        
        return {
            'avs_code': avs_code,
            'avs_description': avs_description,
            'country_mismatch': country_mismatch,
            'zip_mismatch': zip_mismatch,
            'address_mismatch': address_mismatch,
            'avs_risk_score': risk_score
        }
    
    def detect_anomalous_amount(self, transaction, user_transactions):
        """
        Detect if a transaction amount is anomalous for a specific user.
        
        Parameters:
        -----------
        transaction : dict or pd.Series
            Current transaction to check
        user_transactions : pd.DataFrame
            Previous transactions for the same user
            
        Returns:
        --------
        dict
            Dictionary with amount anomaly check results and risk score
        """
        # Convert transaction to Series if it's a dict
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        
        # If user has no previous transactions, use a default approach
        if len(user_transactions) == 0:
            # For new users, consider large amounts as higher risk
            amount = transaction['amount']
            if amount > 500:
                return {
                    'amount_anomaly': 'high',
                    'z_score': None,
                    'user_avg_amount': None,
                    'user_std_amount': None,
                    'amount_risk_score': 0.7
                }
            elif amount > 200:
                return {
                    'amount_anomaly': 'medium',
                    'z_score': None,
                    'user_avg_amount': None,
                    'user_std_amount': None,
                    'amount_risk_score': 0.4
                }
            else:
                return {
                    'amount_anomaly': 'low',
                    'z_score': None,
                    'user_avg_amount': None,
                    'user_std_amount': None,
                    'amount_risk_score': 0.1
                }
        
        # Calculate user's average and standard deviation of transaction amounts
        user_avg_amount = user_transactions['amount'].mean()
        user_std_amount = user_transactions['amount'].std()
        
        # If standard deviation is zero, set a small value to avoid division by zero
        if user_std_amount == 0:
            user_std_amount = 1.0
        
        # Calculate z-score for current transaction amount
        amount = transaction['amount']
        z_score = (amount - user_avg_amount) / user_std_amount
        
        # Determine anomaly level based on z-score
        if abs(z_score) > 3:
            anomaly = 'high'
            risk_score = 0.8
        elif abs(z_score) > 2:
            anomaly = 'medium'
            risk_score = 0.5
        else:
            anomaly = 'low'
            risk_score = 0.1
        
        return {
            'amount_anomaly': anomaly,
            'z_score': z_score,
            'user_avg_amount': user_avg_amount,
            'user_std_amount': user_std_amount,
            'amount_risk_score': risk_score
        }
    
    def apply_rules(self, transaction):
        """
        Apply all rule-based checks to a transaction.
        
        Parameters:
        -----------
        transaction : dict or pd.Series
            Transaction to check
            
        Returns:
        --------
        dict
            Dictionary with rule check results and risk scores
        """
        # Convert transaction to Series if it's a dict
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        
        rule_results = {}
        
        for rule in self.rules:
            try:
                result = rule['function'](transaction)
                rule_results[rule['name']] = result
            except Exception as e:
                print(f"Error applying rule {rule['name']}: {e}")
                rule_results[rule['name']] = 0.0
        
        return rule_results
    
    def predict_fraud_probability(self, transaction_features):
        """
        Predict the probability of fraud for a transaction using the ML model.
        
        Parameters:
        -----------
        transaction_features : pd.DataFrame
            Features of the transaction to check
            
        Returns:
        --------
        float
            Probability of fraud (0-1)
        """
        if self.ml_model is None:
            raise ValueError("Model not trained. Call train_ml_model() first.")
        
        # Ensure transaction_features is a DataFrame
        if isinstance(transaction_features, pd.Series):
            transaction_features = pd.DataFrame([transaction_features])
        
        # Predict probability
        fraud_probability = self.ml_model.predict_proba(transaction_features)[0, 1]
        
        return fraud_probability
    
    def detect_anomaly(self, transaction_features):
        """
        Detect if a transaction is anomalous using the anomaly detection model.
        
        Parameters:
        -----------
        transaction_features : pd.DataFrame
            Features of the transaction to check
            
        Returns:
        --------
        dict
            Dictionary with anomaly detection results
        """
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector not trained. Call train_anomaly_detector() first.")
        
        # Ensure transaction_features is a DataFrame
        if isinstance(transaction_features, pd.Series):
            transaction_features = pd.DataFrame([transaction_features])
        
        # Predict anomaly
        anomaly_score = self.anomaly_detector.decision_function(transaction_features)[0]
        is_anomaly = self.anomaly_detector.predict(transaction_features)[0] == -1
        
        # Convert anomaly score to a 0-1 scale (higher means more anomalous)
        # Note: decision_function returns negative values for anomalies
        normalized_score = 1 / (1 + np.exp(anomaly_score))
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': normalized_score
        }
    
    def analyze_transaction(self, transaction, user_transactions=None, full_features=None):
        """
        Perform a comprehensive fraud analysis on a transaction.
        
        Parameters:
        -----------
        transaction : dict or pd.Series
            Transaction to analyze
        user_transactions : pd.DataFrame
            Previous transactions for the same user
        full_features : pd.DataFrame
            Full feature set for ML model prediction
            
        Returns:
        --------
        dict
            Dictionary with comprehensive fraud analysis results
        """
        # Convert transaction to Series if it's a dict
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        
        results = {
            'transaction_id': transaction.get('transaction_id', None),
            'timestamp': transaction.get('timestamp', None),
            'amount': transaction.get('amount', None),
            'user_id': transaction.get('user_id', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # Initialize user_transactions if not provided
        if user_transactions is None:
            user_transactions = pd.DataFrame(columns=transaction.index)
        
        # Perform velocity checks
        if 'timestamp' in transaction and 'amount' in transaction and 'billing_country' in transaction:
            velocity_results = self.check_transaction_velocity(transaction, user_transactions)
            results['checks']['velocity'] = velocity_results['velocity_checks']
            results['risk_scores']['velocity'] = velocity_results['velocity_risk_score']
        
        # Perform BIN country check
        if 'card_bin' in transaction and 'billing_country' in transaction:
            bin_results = self.check_bin_country(transaction['card_bin'], transaction['billing_country'])
            results['checks']['bin'] = bin_results
            results['risk_scores']['bin'] = bin_results['bin_risk_score']
        
        # Perform address verification
        avs_results = self.check_address_verification(transaction)
        results['checks']['avs'] = avs_results
        results['risk_scores']['avs'] = avs_results['avs_risk_score']
        
        # Detect anomalous amount
        if 'amount' in transaction:
            amount_results = self.detect_anomalous_amount(transaction, user_transactions)
            results['checks']['amount'] = amount_results
            results['risk_scores']['amount'] = amount_results['amount_risk_score']
        
        # Apply rule-based checks
        if self.rules:
            rule_results = self.apply_rules(transaction)
            results['checks']['rules'] = rule_results
            results['risk_scores']['rules'] = sum(rule_results.values()) / len(rule_results) if rule_results else 0.0
        
        # ML model prediction
        if self.ml_model is not None and full_features is not None:
            ml_probability = self.predict_fraud_probability(full_features)
            results['checks']['ml_model'] = {'probability': ml_probability}
            results['risk_scores']['ml_model'] = ml_probability
        
        # Anomaly detection
        if self.anomaly_detector is not None and full_features is not None:
            anomaly_results = self.detect_anomaly(full_features)
            results['checks']['anomaly'] = anomaly_results
            results['risk_scores']['anomaly'] = anomaly_results['anomaly_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'velocity': 0.2,
            'bin': 0.15,
            'avs': 0.15,
            'amount': 0.1,
            'rules': 0.1,
            'ml_model': 0.2,
            'anomaly': 0.1
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
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
        else:
            results['risk_level'] = 'low'
        
        return results
    
    def save_model(self, filename='credit_card_fraud_model.pkl'):
        """
        Save the trained models and configuration to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'ml_model': self.ml_model,
            'anomaly_detector': self.anomaly_detector,
            'feature_importance': self.feature_importance,
            'rules': [(rule['name'], rule['description']) for rule in self.rules],  # Can't pickle functions
            'velocity_thresholds': self.velocity_thresholds
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='credit_card_fraud_model.pkl'):
        """
        Load trained models and configuration from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
            
        Returns:
        --------
        self : CreditCardFraudDetector
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ml_model = model_data['ml_model']
        self.anomaly_detector = model_data['anomaly_detector']
        self.feature_importance = model_data['feature_importance']
        self.velocity_thresholds = model_data['velocity_thresholds']
        
        # Note: rule functions can't be pickled, so only names and descriptions are saved
        # Rules need to be re-added with their functions after loading
        
        return self
    
    def create_default_rules(self):
        """
        Create a set of default rules for credit card fraud detection.
        
        Returns:
        --------
        self : CreditCardFraudDetector
            Returns self for method chaining
        """
        # Rule 1: High amount for new account
        def high_amount_new_account(transaction):
            if 'amount' in transaction and 'account_age_days' in transaction:
                if transaction['amount'] > 500 and transaction['account_age_days'] < 30:
                    return 0.8
            return 0.0
        
        self.add_rule(
            high_amount_new_account,
            'high_amount_new_account',
            'Detects high transaction amounts for new accounts'
        )
        
        # Rule 2: Mismatched shipping and billing countries
        def country_mismatch(transaction):
            if 'billing_country' in transaction and 'shipping_country' in transaction:
                if transaction['billing_country'] != transaction['shipping_country']:
                    return 0.7
            return 0.0
        
        self.add_rule(
            country_mismatch,
            'country_mismatch',
            'Detects mismatches between billing and shipping countries'
        )
        
        # Rule 3: High-risk payment method
        def high_risk_payment(transaction):
            if 'payment_method' in transaction:
                high_risk_methods = ['gift_card', 'prepaid_card', 'cryptocurrency']
                if transaction['payment_method'] in high_risk_methods:
                    return 0.6
            return 0.0
        
        self.add_rule(
            high_risk_payment,
            'high_risk_payment',
            'Detects high-risk payment methods'
        )
        
        # Rule 4: Transaction time during unusual hours
        def unusual_hours(transaction):
            if 'timestamp' in transaction:
                if isinstance(transaction['timestamp'], str):
                    hour = pd.to_datetime(transaction['timestamp']).hour
                else:
                    hour = transaction['timestamp'].hour
                
                if hour >= 1 and hour <= 5:  # Between 1 AM and 5 AM
                    return 0.5
            return 0.0
        
        self.add_rule(
            unusual_hours,
            'unusual_hours',
            'Detects transactions during unusual hours (1 AM - 5 AM)'
        )
        
        # Rule 5: High-risk country
        def high_risk_country(transaction):
            if 'billing_country' in transaction:
                high_risk_countries = ['CountryA', 'CountryB', 'CountryC']
                if transaction['billing_country'] in high_risk_countries:
                    return 0.7
            return 0.0
        
        self.add_rule(
            high_risk_country,
            'high_risk_country',
            'Detects transactions from high-risk countries'
        )
        
        return self
