"""
Unified Risk Scoring System for E-commerce Fraud Detection

This module implements a unified risk scoring system that integrates all fraud detection
modules to provide a comprehensive fraud risk assessment for e-commerce transactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

# Import fraud detection modules
from credit_card_fraud_detection import CreditCardFraudDetector
from account_takeover_prevention import AccountTakeoverDetector
from friendly_fraud_detection import FriendlyFraudDetector
from additional_fraud_detection import AdditionalFraudDetector

class UnifiedRiskScoringSystem:
    """
    A class for integrating all fraud detection modules and providing a unified
    risk assessment for e-commerce transactions.
    
    This class combines the results from various fraud detection modules and
    calculates an overall risk score based on configurable weights and thresholds.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the UnifiedRiskScoringSystem.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        
        # Initialize fraud detection modules
        self.credit_card_detector = CreditCardFraudDetector(model_dir=model_dir)
        self.account_takeover_detector = AccountTakeoverDetector(model_dir=model_dir)
        self.friendly_fraud_detector = FriendlyFraudDetector(model_dir=model_dir)
        self.additional_fraud_detector = AdditionalFraudDetector(model_dir=model_dir)
        
        # Risk score weights for different fraud types
        self.risk_weights = {
            'credit_card_fraud': 0.3,
            'account_takeover': 0.25,
            'friendly_fraud': 0.2,
            'promotion_abuse': 0.1,
            'refund_fraud': 0.1,
            'bot_activity': 0.05
        }
        
        # Risk thresholds for decision making
        self.risk_thresholds = {
            'high_risk': 0.7,    # Transactions with risk score >= 0.7 are high risk
            'medium_risk': 0.4,  # Transactions with risk score >= 0.4 are medium risk
            'low_risk': 0.2      # Transactions with risk score >= 0.2 are low risk
        }
        
        # Transaction history for risk analysis
        self.transaction_history = []
        self.user_risk_profiles = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def set_risk_weights(self, weights):
        """
        Set weights for different fraud types in the unified risk score.
        
        Parameters:
        -----------
        weights : dict
            Dictionary of weights for different fraud types
            
        Returns:
        --------
        self : UnifiedRiskScoringSystem
            Returns self for method chaining
        """
        self.risk_weights.update(weights)
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(self.risk_weights.values())
        if total_weight != 1.0:
            for key in self.risk_weights:
                self.risk_weights[key] /= total_weight
        
        return self
    
    def set_risk_thresholds(self, thresholds):
        """
        Set thresholds for risk level classification.
        
        Parameters:
        -----------
        thresholds : dict
            Dictionary of threshold values
            
        Returns:
        --------
        self : UnifiedRiskScoringSystem
            Returns self for method chaining
        """
        self.risk_thresholds.update(thresholds)
        return self
    
    def load_models(self):
        """
        Load all fraud detection models.
        
        Returns:
        --------
        self : UnifiedRiskScoringSystem
            Returns self for method chaining
        """
        try:
            self.credit_card_detector.load_model('credit_card_fraud_model.pkl')
        except:
            print("Credit card fraud model not found. Using default model.")
        
        try:
            self.account_takeover_detector.load_model('account_takeover_model.pkl')
        except:
            print("Account takeover model not found. Using default model.")
        
        try:
            self.friendly_fraud_detector.load_model('friendly_fraud_model.pkl')
        except:
            print("Friendly fraud model not found. Using default model.")
        
        try:
            self.additional_fraud_detector.load_model('additional_fraud_models.pkl')
        except:
            print("Additional fraud models not found. Using default models.")
        
        return self
    
    def save_models(self):
        """
        Save all fraud detection models.
        
        Returns:
        --------
        self : UnifiedRiskScoringSystem
            Returns self for method chaining
        """
        self.credit_card_detector.save_model('credit_card_fraud_model.pkl')
        self.account_takeover_detector.save_model('account_takeover_model.pkl')
        self.friendly_fraud_detector.save_model('friendly_fraud_model.pkl')
        self.additional_fraud_detector.save_model('additional_fraud_models.pkl')
        
        return self
    
    def analyze_transaction(self, transaction_data, user_data=None, session_data=None):
        """
        Perform a comprehensive fraud analysis on a transaction.
        
        Parameters:
        -----------
        transaction_data : dict or pd.Series
            Data about the transaction
        user_data : dict or pd.Series
            Data about the user
        session_data : dict or pd.Series
            Data about the user session
            
        Returns:
        --------
        dict
            Dictionary with comprehensive fraud analysis results
        """
        # Convert data to Series if they're dicts
        if isinstance(transaction_data, dict):
            transaction_data = pd.Series(transaction_data)
        
        if user_data is not None and isinstance(user_data, dict):
            user_data = pd.Series(user_data)
        
        if session_data is not None and isinstance(session_data, dict):
            session_data = pd.Series(session_data)
        
        # Initialize results
        results = {
            'transaction_id': transaction_data.get('transaction_id', None),
            'user_id': transaction_data.get('user_id', None),
            'timestamp': transaction_data.get('timestamp', None),
            'amount': transaction_data.get('amount', None),
            'fraud_checks': {},
            'risk_scores': {},
            'overall_risk_score': None,
            'risk_level': None,
            'recommended_action': None
        }
        
        # Get user's transaction history
        user_id = transaction_data.get('user_id', None)
        user_transactions = self.get_user_transactions(user_id)
        
        # 1. Credit Card Fraud Detection
        if 'payment_method' in transaction_data and transaction_data['payment_method'] in ['credit_card', 'debit_card']:
            cc_results = self.credit_card_detector.analyze_transaction(transaction_data, user_transactions)
            results['fraud_checks']['credit_card_fraud'] = cc_results
            results['risk_scores']['credit_card_fraud'] = cc_results['overall_risk_score']
        
        # 2. Account Takeover Detection (if login data is available)
        if session_data is not None and 'login_data' in session_data:
            login_data = session_data['login_data']
            if isinstance(login_data, dict):
                login_data = pd.Series(login_data)
            
            # Get user's login history
            login_history = self.get_user_login_history(user_id)
            
            # Build user profile if not already available
            user_profile = self.account_takeover_detector.user_profiles.get(user_id, None)
            if user_profile is None and not login_history.empty:
                user_profile = self.account_takeover_detector.build_user_profile(user_id, login_history)
            
            at_results = self.account_takeover_detector.analyze_login_attempt(login_data, user_profile, login_history)
            results['fraud_checks']['account_takeover'] = at_results
            results['risk_scores']['account_takeover'] = at_results['overall_risk_score']
        
        # 3. Friendly Fraud Detection (for potential chargeback prediction)
        if user_id is not None:
            # Build customer profile if not already available
            customer_profile = self.friendly_fraud_detector.customer_profiles.get(user_id, None)
            if customer_profile is None and not user_transactions.empty:
                # Get user's chargeback history
                user_chargebacks = self.get_user_chargebacks(user_id)
                customer_profile = self.friendly_fraud_detector.build_customer_profile(user_id, user_transactions, user_chargebacks)
            
            # Extract features for chargeback prediction
            if customer_profile is not None:
                chargeback_features = self.friendly_fraud_detector.extract_chargeback_features(transaction_data, customer_profile)
                
                # Predict chargeback risk if model is available
                if self.friendly_fraud_detector.chargeback_model is not None:
                    chargeback_risk = self.friendly_fraud_detector.predict_chargeback_risk(chargeback_features)
                    
                    ff_results = {
                        'chargeback_risk': chargeback_risk,
                        'customer_profile': customer_profile
                    }
                    
                    results['fraud_checks']['friendly_fraud'] = ff_results
                    results['risk_scores']['friendly_fraud'] = chargeback_risk
        
        # 4. Additional Fraud Detection
        
        # 4.1 Promotion Abuse Detection (if promotion data is available)
        if 'promotion_code' in transaction_data:
            promotion_data = {
                'promotion_id': transaction_data.get('promotion_id', None),
                'promotion_code': transaction_data.get('promotion_code', None),
                'user_id': user_id,
                'order_id': transaction_data.get('transaction_id', None),
                'timestamp': transaction_data.get('timestamp', None),
                'amount': transaction_data.get('amount', None)
            }
            
            pa_results = self.additional_fraud_detector.detect_promotion_abuse(promotion_data, user_data, transaction_data)
            results['fraud_checks']['promotion_abuse'] = pa_results
            results['risk_scores']['promotion_abuse'] = pa_results['overall_risk_score']
        
        # 4.2 Bot Activity Detection (if session data is available)
        if session_data is not None:
            request_data = {
                'request_id': session_data.get('request_id', None),
                'session_id': session_data.get('session_id', None),
                'ip_address': session_data.get('ip_address', None),
                'timestamp': session_data.get('timestamp', None),
                'user_agent': session_data.get('user_agent', None)
            }
            
            bot_results = self.additional_fraud_detector.detect_bot_activity(request_data, session_data)
            results['fraud_checks']['bot_activity'] = bot_results
            results['risk_scores']['bot_activity'] = bot_results['overall_risk_score']
        
        # Calculate unified risk score
        self.calculate_unified_risk_score(results)
        
        # Add transaction to history
        self.add_transaction_to_history(transaction_data, results)
        
        # Update user risk profile
        self.update_user_risk_profile(user_id, results)
        
        return results
    
    def calculate_unified_risk_score(self, results):
        """
        Calculate a unified risk score based on all fraud detection results.
        
        Parameters:
        -----------
        results : dict
            Dictionary with fraud detection results
            
        Returns:
        --------
        None (updates the results dictionary in place)
        """
        weighted_sum = 0.0
        weight_sum = 0.0
        
        # Calculate weighted sum of risk scores
        for fraud_type, weight in self.risk_weights.items():
            if fraud_type in results['risk_scores']:
                weighted_sum += results['risk_scores'][fraud_type] * weight
                weight_sum += weight
        
        # Calculate overall risk score
        if weight_sum > 0:
            overall_risk_score = weighted_sum / weight_sum
        else:
            overall_risk_score = 0.0
        
        # Determine risk level
        if overall_risk_score >= self.risk_thresholds['high_risk']:
            risk_level = 'high'
            recommended_action = 'block_transaction'
        elif overall_risk_score >= self.risk_thresholds['medium_risk']:
            risk_level = 'medium'
            recommended_action = 'additional_verification'
        elif overall_risk_score >= self.risk_thresholds['low_risk']:
            risk_level = 'low'
            recommended_action = 'monitor'
        else:
            risk_level = 'very_low'
            recommended_action = 'allow'
        
        # Update results
        results['overall_risk_score'] = overall_risk_score
        results['risk_level'] = risk_level
        results['recommended_action'] = recommended_action
    
    def get_user_transactions(self, user_id):
        """
        Get a user's transaction history.
        
        Parameters:
        -----------
        user_id : str
            User identifier
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with user's transaction history
        """
        if user_id is None:
            return pd.DataFrame()
        
        # Filter transaction history for this user
        user_transactions = [
            tx for tx in self.transaction_history
            if tx.get('user_id') == user_id
        ]
        
        if not user_transactions:
            return pd.DataFrame()
        
        return pd.DataFrame(user_transactions)
    
    def get_user_login_history(self, user_id):
        """
        Get a user's login history.
        
        Parameters:
        -----------
        user_id : str
            User identifier
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with user's login history
        """
        # This would typically come from a database
        # For this implementation, we'll return an empty DataFrame
        return pd.DataFrame()
    
    def get_user_chargebacks(self, user_id):
        """
        Get a user's chargeback history.
        
        Parameters:
        -----------
        user_id : str
            User identifier
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with user's chargeback history
        """
        # This would typically come from a database
        # For this implementation, we'll return an empty DataFrame
        return pd.DataFrame()
    
    def add_transaction_to_history(self, transaction_data, results):
        """
        Add a transaction to the history.
        
        Parameters:
        -----------
        transaction_data : pd.Series
            Data about the transaction
        results : dict
            Fraud analysis results
            
        Returns:
        --------
        None
        """
        # Convert transaction_data to dict if it's a Series
        if isinstance(transaction_data, pd.Series):
            transaction_data = transaction_data.to_dict()
        
        # Add risk assessment results
        transaction_data['risk_score'] = results['overall_risk_score']
        transaction_data['risk_level'] = results['risk_level']
        
        # Add to history
        self.transaction_history.append(transaction_data)
        
        # Limit history size to prevent memory issues
        if len(self.transaction_history) > 10000:
            self.transaction_history = self.transaction_history[-10000:]
    
    def update_user_risk_profile(self, user_id, results):
        """
        Update a user's risk profile based on transaction results.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        results : dict
            Fraud analysis results
            
        Returns:
        --------
        None
        """
        if user_id is None:
            return
        
        # Get or create user risk profile
        if user_id not in self.user_risk_profiles:
            self.user_risk_profiles[user_id] = {
                'user_id': user_id,
                'transaction_count': 0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'average_risk_score': 0.0,
                'last_transaction_timestamp': None,
                'risk_factors': []
            }
        
        profile = self.user_risk_profiles[user_id]
        
        # Update profile
        profile['transaction_count'] += 1
        
        if results['risk_level'] == 'high':
            profile['high_risk_count'] += 1
        elif results['risk_level'] == 'medium':
            profile['medium_risk_count'] += 1
        
        # Update average risk score
        profile['average_risk_score'] = (
            (profile['average_risk_score'] * (profile['transaction_count'] - 1) + results['overall_risk_score']) / 
            profile['transaction_count']
        )
        
        # Update last transaction timestamp
        profile['last_transaction_timestamp'] = results['timestamp']
        
        # Update risk factors
        risk_factors = []
        
        for fraud_type, risk_score in results['risk_scores'].items():
            if risk_score >= self.risk_thresholds['medium_risk']:
                risk_factors.append(fraud_type)
        
        profile['risk_factors'] = list(set(profile['risk_factors'] + risk_factors))
    
    def get_user_risk_profile(self, user_id):
        """
        Get a user's risk profile.
        
        Parameters:
        -----------
        user_id : str
            User identifier
            
        Returns:
        --------
        dict
            User's risk profile
        """
        return self.user_risk_profiles.get(user_id, None)
    
    def analyze_refund_request(self, refund_data, order_data=None, user_data=None):
        """
        Analyze a refund request for potential fraud.
        
        Parameters:
        -----------
        refund_data : dict or pd.Series
            Data about the refund request
        order_data : dict or pd.Series
            Data about the original order
        user_data : dict or pd.Series
            Data about the user
            
        Returns:
        --------
        dict
            Dictionary with refund fraud analysis results
        """
        # Convert data to Series if they're dicts
        if isinstance(refund_data, dict):
            refund_data = pd.Series(refund_data)
        
        if order_data is not None and isinstance(order_data, dict):
            order_data = pd.Series(order_data)
        
        if user_data is not None and isinstance(user_data, dict):
            user_data = pd.Series(user_data)
        
        # Analyze refund request
        refund_results = self.additional_fraud_detector.detect_refund_fraud(refund_data, order_data, user_data)
        
        # Get user's risk profile
        user_id = refund_data.get('user_id', None)
        user_risk_profile = self.get_user_risk_profile(user_id)
        
        # Adjust risk based on user's risk profile
        if user_risk_profile is not None:
            # If user has high average risk score, increase refund risk
            if user_risk_profile['average_risk_score'] >= self.risk_thresholds['high_risk']:
                refund_results['overall_risk_score'] = min(1.0, refund_results['overall_risk_score'] + 0.2)
            
            # If user has multiple high-risk transactions, increase refund risk
            if user_risk_profile['high_risk_count'] >= 2:
                refund_results['overall_risk_score'] = min(1.0, refund_results['overall_risk_score'] + 0.1)
            
            # Update risk level and recommended action based on adjusted risk score
            if refund_results['overall_risk_score'] >= self.risk_thresholds['high_risk']:
                refund_results['risk_level'] = 'high'
                refund_results['recommended_action'] = 'deny_refund'
            elif refund_results['overall_risk_score'] >= self.risk_thresholds['medium_risk']:
                refund_results['risk_level'] = 'medium'
                refund_results['recommended_action'] = 'manual_review'
        
        return refund_results
    
    def analyze_login_attempt(self, login_data, user_data=None, session_data=None):
        """
        Analyze a login attempt for potential account takeover.
        
        Parameters:
        -----------
        login_data : dict or pd.Series
            Data about the login attempt
        user_data : dict or pd.Series
            Data about the user
        session_data : dict or pd.Series
            Data about the user session
            
        Returns:
        --------
        dict
            Dictionary with account takeover analysis results
        """
        # Convert data to Series if they're dicts
        if isinstance(login_data, dict):
            login_data = pd.Series(login_data)
        
        if user_data is not None and isinstance(user_data, dict):
            user_data = pd.Series(user_data)
        
        if session_data is not None and isinstance(session_data, dict):
            session_data = pd.Series(session_data)
        
        # Get user's login history
        user_id = login_data.get('user_id', None)
        login_history = self.get_user_login_history(user_id)
        
        # Build user profile if not already available
        user_profile = self.account_takeover_detector.user_profiles.get(user_id, None)
        if user_profile is None and not login_history.empty:
            user_profile = self.account_takeover_detector.build_user_profile(user_id, login_history)
        
        # Analyze login attempt
        login_results = self.account_takeover_detector.analyze_login_attempt(login_data, user_profile, login_history)
        
        # Check for bot activity if session data is available
        if session_data is not None:
            request_data = {
                'request_id': session_data.get('request_id', None),
                'session_id': session_data.get('session_id', None),
                'ip_address': session_data.get('ip_address', None),
                'timestamp': session_data.get('timestamp', None),
                'user_agent': session_data.get('user_agent', None)
            }
            
            bot_results = self.additional_fraud_detector.detect_bot_activity(request_data, session_data)
            
            # If bot activity is detected, increase account takeover risk
            if bot_results['risk_level'] == 'high':
                login_results['overall_risk_score'] = min(1.0, login_results['overall_risk_score'] + 0.2)
                
                # Update risk level and recommended action based on adjusted risk score
                if login_results['overall_risk_score'] >= 0.7:
                    login_results['risk_level'] = 'high'
                    login_results['recommended_action'] = 'block_and_notify'
                elif login_results['overall_risk_score'] >= 0.4:
                    login_results['risk_level'] = 'medium'
                    login_results['recommended_action'] = 'additional_verification'
        
        # Get user's risk profile
        user_risk_profile = self.get_user_risk_profile(user_id)
        
        # Adjust risk based on user's risk profile
        if user_risk_profile is not None:
            # If user has high average risk score, increase login risk
            if user_risk_profile['average_risk_score'] >= self.risk_thresholds['high_risk']:
                login_results['overall_risk_score'] = min(1.0, login_results['overall_risk_score'] + 0.1)
            
            # Update risk level and recommended action based on adjusted risk score
            if login_results['overall_risk_score'] >= 0.7:
                login_results['risk_level'] = 'high'
                login_results['recommended_action'] = 'block_and_notify'
            elif login_results['overall_risk_score'] >= 0.4:
                login_results['risk_level'] = 'medium'
                login_results['recommended_action'] = 'additional_verification'
        
        return login_results
    
    def generate_risk_report(self, start_date=None, end_date=None):
        """
        Generate a risk report for a specified time period.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for the report
        end_date : str or datetime
            End date for the report
            
        Returns:
        --------
        dict
            Dictionary with risk report data
        """
        # Convert dates to datetime if they're strings
        if start_date is not None and isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        if end_date is not None and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # If no dates specified, use all data
        if start_date is None and end_date is None:
            transactions = self.transaction_history
        else:
            # Filter transactions by date
            transactions = []
            
            for tx in self.transaction_history:
                tx_date = tx.get('timestamp', None)
                
                if tx_date is None:
                    continue
                
                if isinstance(tx_date, str):
                    tx_date = pd.to_datetime(tx_date)
                
                if start_date is not None and tx_date < start_date:
                    continue
                
                if end_date is not None and tx_date > end_date:
                    continue
                
                transactions.append(tx)
        
        # Calculate report metrics
        total_transactions = len(transactions)
        
        if total_transactions == 0:
            return {
                'start_date': start_date,
                'end_date': end_date,
                'total_transactions': 0,
                'risk_levels': {},
                'fraud_types': {},
                'high_risk_users': []
            }
        
        # Count transactions by risk level
        risk_levels = {
            'high': 0,
            'medium': 0,
            'low': 0,
            'very_low': 0
        }
        
        for tx in transactions:
            risk_level = tx.get('risk_level', 'unknown')
            if risk_level in risk_levels:
                risk_levels[risk_level] += 1
        
        # Calculate percentages
        risk_level_percentages = {
            level: count / total_transactions * 100
            for level, count in risk_levels.items()
        }
        
        # Identify high-risk users
        user_risk_counts = defaultdict(int)
        
        for tx in transactions:
            user_id = tx.get('user_id', None)
            risk_level = tx.get('risk_level', None)
            
            if user_id is not None and risk_level == 'high':
                user_risk_counts[user_id] += 1
        
        high_risk_users = [
            {
                'user_id': user_id,
                'high_risk_transactions': count,
                'risk_profile': self.get_user_risk_profile(user_id)
            }
            for user_id, count in user_risk_counts.items()
            if count >= 2  # Users with at least 2 high-risk transactions
        ]
        
        # Sort high-risk users by number of high-risk transactions
        high_risk_users.sort(key=lambda x: x['high_risk_transactions'], reverse=True)
        
        # Identify common fraud types
        fraud_types = defaultdict(int)
        
        for user_id, profile in self.user_risk_profiles.items():
            for factor in profile['risk_factors']:
                fraud_types[factor] += 1
        
        # Create report
        report = {
            'start_date': start_date,
            'end_date': end_date,
            'total_transactions': total_transactions,
            'risk_levels': {
                'counts': risk_levels,
                'percentages': risk_level_percentages
            },
            'fraud_types': dict(fraud_types),
            'high_risk_users': high_risk_users
        }
        
        return report
    
    def plot_risk_distribution(self, save_path=None):
        """
        Plot the distribution of risk scores.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot (if None, plot is displayed)
            
        Returns:
        --------
        None
        """
        if not self.transaction_history:
            print("No transaction data available for plotting.")
            return
        
        # Extract risk scores
        risk_scores = [tx.get('risk_score', 0) for tx in self.transaction_history if 'risk_score' in tx]
        
        if not risk_scores:
            print("No risk score data available for plotting.")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(risk_scores, bins=20, kde=True)
        
        # Add threshold lines
        plt.axvline(x=self.risk_thresholds['high_risk'], color='red', linestyle='--', 
                   label=f"High Risk Threshold ({self.risk_thresholds['high_risk']})")
        plt.axvline(x=self.risk_thresholds['medium_risk'], color='orange', linestyle='--', 
                   label=f"Medium Risk Threshold ({self.risk_thresholds['medium_risk']})")
        plt.axvline(x=self.risk_thresholds['low_risk'], color='green', linestyle='--', 
                   label=f"Low Risk Threshold ({self.risk_thresholds['low_risk']})")
        
        plt.title('Distribution of Transaction Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_fraud_types(self, save_path=None):
        """
        Plot the distribution of fraud types.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot (if None, plot is displayed)
            
        Returns:
        --------
        None
        """
        # Count fraud types
        fraud_types = defaultdict(int)
        
        for user_id, profile in self.user_risk_profiles.items():
            for factor in profile['risk_factors']:
                fraud_types[factor] += 1
        
        if not fraud_types:
            print("No fraud type data available for plotting.")
            return
        
        # Convert to DataFrame for plotting
        fraud_df = pd.DataFrame({
            'fraud_type': list(fraud_types.keys()),
            'count': list(fraud_types.values())
        }).sort_values('count', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart
        sns.barplot(x='fraud_type', y='count', data=fraud_df)
        
        plt.title('Distribution of Fraud Types')
        plt.xlabel('Fraud Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_system_state(self, filename='unified_risk_system.pkl'):
        """
        Save the system state to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the system state to
        """
        system_path = os.path.join(self.model_dir, filename)
        
        system_data = {
            'risk_weights': self.risk_weights,
            'risk_thresholds': self.risk_thresholds,
            'transaction_history': self.transaction_history,
            'user_risk_profiles': self.user_risk_profiles
        }
        
        with open(system_path, 'wb') as f:
            pickle.dump(system_data, f)
    
    def load_system_state(self, filename='unified_risk_system.pkl'):
        """
        Load the system state from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the system state from
            
        Returns:
        --------
        self : UnifiedRiskScoringSystem
            Returns self for method chaining
        """
        system_path = os.path.join(self.model_dir, filename)
        
        with open(system_path, 'rb') as f:
            system_data = pickle.load(f)
        
        self.risk_weights = system_data['risk_weights']
        self.risk_thresholds = system_data['risk_thresholds']
        self.transaction_history = system_data['transaction_history']
        self.user_risk_profiles = system_data['user_risk_profiles']
        
        return self
    
    def create_sample_data(self, n_transactions=500, save_to_csv=True):
        """
        Create sample transaction data for demonstration purposes.
        
        Parameters:
        -----------
        n_transactions : int
            Number of transactions to generate
        save_to_csv : bool
            Whether to save the generated data to CSV files
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with sample transaction data
        """
        np.random.seed(42)
        
        # Generate transaction data
        transaction_ids = [f"T{i:06d}" for i in range(1, n_transactions + 1)]
        user_ids = [f"U{np.random.randint(1, 100):05d}" for _ in range(n_transactions)]
        
        # Generate timestamps over the last 30 days
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)
        timestamps = pd.date_range(start=start_date, end=end_date, periods=n_transactions)
        
        # Generate transaction amounts
        amounts = np.random.uniform(10, 1000, n_transactions).round(2)
        
        # Generate payment methods
        payment_methods = np.random.choice(['credit_card', 'debit_card', 'digital_wallet', 'bank_transfer'], n_transactions)
        
        # Generate product categories
        product_categories = np.random.choice(['electronics', 'clothing', 'home', 'books', 'jewelry', 'digital', 'gift_card'], n_transactions)
        
        # Generate IP addresses
        ip_addresses = [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_transactions)]
        
        # Generate device IDs
        device_ids = [f"D{np.random.randint(1, 50):05d}" for _ in range(n_transactions)]
        
        # Generate risk scores (mostly low, some medium, few high)
        risk_scores = np.random.choice(
            [np.random.uniform(0, 0.2), np.random.uniform(0.4, 0.7), np.random.uniform(0.7, 1.0)],
            n_transactions,
            p=[0.7, 0.2, 0.1]
        )
        
        # Determine risk levels
        risk_levels = []
        for score in risk_scores:
            if score >= self.risk_thresholds['high_risk']:
                risk_levels.append('high')
            elif score >= self.risk_thresholds['medium_risk']:
                risk_levels.append('medium')
            elif score >= self.risk_thresholds['low_risk']:
                risk_levels.append('low')
            else:
                risk_levels.append('very_low')
        
        # Create DataFrame
        transactions_data = {
            'transaction_id': transaction_ids,
            'user_id': user_ids,
            'timestamp': timestamps,
            'amount': amounts,
            'payment_method': payment_methods,
            'product_category': product_categories,
            'ip_address': ip_addresses,
            'device_id': device_ids,
            'risk_score': risk_scores,
            'risk_level': risk_levels
        }
        
        transactions_df = pd.DataFrame(transactions_data)
        
        # Save to CSV if requested
        if save_to_csv:
            data_dir = os.path.dirname(self.model_dir)
            data_dir = os.path.join(data_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            transactions_df.to_csv(os.path.join(data_dir, 'sample_unified_transactions.csv'), index=False)
        
        # Add to transaction history
        self.transaction_history = transactions_df.to_dict('records')
        
        # Build user risk profiles
        for user_id in set(user_ids):
            user_transactions = transactions_df[transactions_df['user_id'] == user_id]
            
            high_risk_count = sum(user_transactions['risk_level'] == 'high')
            medium_risk_count = sum(user_transactions['risk_level'] == 'medium')
            average_risk_score = user_transactions['risk_score'].mean()
            last_transaction = user_transactions.iloc[-1]['timestamp']
            
            # Generate random risk factors
            risk_factors = []
            if high_risk_count > 0 or medium_risk_count > 0:
                potential_factors = ['credit_card_fraud', 'account_takeover', 'friendly_fraud', 'promotion_abuse', 'refund_fraud', 'bot_activity']
                n_factors = np.random.randint(1, 4)
                risk_factors = np.random.choice(potential_factors, size=min(n_factors, len(potential_factors)), replace=False).tolist()
            
            self.user_risk_profiles[user_id] = {
                'user_id': user_id,
                'transaction_count': len(user_transactions),
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'average_risk_score': average_risk_score,
                'last_transaction_timestamp': last_transaction,
                'risk_factors': risk_factors
            }
        
        return transactions_df
