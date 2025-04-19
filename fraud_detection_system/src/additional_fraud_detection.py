"""
Additional Fraud Detection Module for E-commerce

This module implements detection mechanisms for various additional fraud types
in e-commerce systems, including promotion abuse, refund fraud, bot attacks,
and synthetic identity detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
import json
import re
import hashlib
from collections import defaultdict, Counter
import ipaddress

class AdditionalFraudDetector:
    """
    A class for detecting various additional types of fraud in e-commerce systems.
    
    This class implements methods for identifying promotion/coupon abuse, refund fraud,
    bot/automated attacks, and synthetic identity fraud.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the AdditionalFraudDetector.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.promotion_abuse_model = None
        self.bot_detection_model = None
        self.synthetic_identity_model = None
        self.feature_importance = {}
        
        # Databases for tracking
        self.promotion_usage_db = defaultdict(list)  # {promotion_code: [usage_records]}
        self.refund_history_db = defaultdict(list)   # {user_id: [refund_records]}
        self.ip_reputation_db = {}                   # {ip_address: reputation_score}
        self.user_agent_patterns = {}                # {pattern: bot_likelihood}
        
        # Thresholds for detection
        self.thresholds = {
            'promotion_usage_limit': 1,          # Max uses per user
            'promotion_multi_account_limit': 3,  # Max accounts from same IP/device
            'high_refund_ratio': 0.3,            # Ratio of refunds to purchases
            'refund_frequency_days': 30,         # Time window for refund frequency
            'max_refunds_in_window': 3,          # Max refunds in time window
            'bot_request_rate_limit': 10,        # Requests per minute
            'bot_session_duration_min': 10,      # Minimum session duration in seconds
            'synthetic_identity_score': 0.7      # Threshold for synthetic identity
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
        self : AdditionalFraudDetector
            Returns self for method chaining
        """
        self.thresholds.update(thresholds)
        return self
    
    #
    # Promotion Abuse Detection
    #
    
    def detect_promotion_abuse(self, promotion_data, user_data=None, order_data=None):
        """
        Detect potential promotion or coupon abuse.
        
        Parameters:
        -----------
        promotion_data : dict or pd.Series
            Data about the promotion usage
        user_data : dict or pd.Series
            Data about the user
        order_data : dict or pd.Series
            Data about the order
            
        Returns:
        --------
        dict
            Analysis results with risk scores
        """
        # Convert data to Series if they're dicts
        if isinstance(promotion_data, dict):
            promotion_data = pd.Series(promotion_data)
        
        if user_data is not None and isinstance(user_data, dict):
            user_data = pd.Series(user_data)
        
        if order_data is not None and isinstance(order_data, dict):
            order_data = pd.Series(order_data)
        
        # Initialize results
        results = {
            'promotion_id': promotion_data.get('promotion_id', None),
            'user_id': promotion_data.get('user_id', None),
            'order_id': promotion_data.get('order_id', None),
            'timestamp': promotion_data.get('timestamp', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # Check 1: Usage frequency
        usage_risk = self.check_promotion_usage_frequency(promotion_data)
        results['checks']['usage_frequency'] = usage_risk
        results['risk_scores']['usage_frequency'] = usage_risk['risk_score']
        
        # Check 2: Multi-account abuse
        if user_data is not None:
            multi_account_risk = self.check_multi_account_abuse(promotion_data, user_data)
            results['checks']['multi_account'] = multi_account_risk
            results['risk_scores']['multi_account'] = multi_account_risk['risk_score']
        
        # Check 3: Order manipulation
        if order_data is not None:
            order_risk = self.check_order_manipulation(promotion_data, order_data)
            results['checks']['order_manipulation'] = order_risk
            results['risk_scores']['order_manipulation'] = order_risk['risk_score']
        
        # Check 4: Timing patterns
        timing_risk = self.check_promotion_timing(promotion_data)
        results['checks']['timing_patterns'] = timing_risk
        results['risk_scores']['timing_patterns'] = timing_risk['risk_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'usage_frequency': 0.3,
            'multi_account': 0.3,
            'order_manipulation': 0.2,
            'timing_patterns': 0.2
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
            results['recommended_action'] = 'block_promotion'
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
            results['recommended_action'] = 'manual_review'
        else:
            results['risk_level'] = 'low'
            results['recommended_action'] = 'allow'
        
        # Update promotion usage database
        self.update_promotion_usage(promotion_data)
        
        return results
    
    def check_promotion_usage_frequency(self, promotion_data):
        """
        Check if a user is using a promotion code too frequently.
        
        Parameters:
        -----------
        promotion_data : pd.Series
            Data about the promotion usage
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        promotion_code = promotion_data.get('promotion_code', None)
        user_id = promotion_data.get('user_id', None)
        
        if promotion_code is None or user_id is None:
            return {
                'usage_count': 0,
                'exceeds_limit': False,
                'risk_score': 0.1
            }
        
        # Count previous uses by this user
        usage_count = 0
        
        for usage in self.promotion_usage_db.get(promotion_code, []):
            if usage['user_id'] == user_id:
                usage_count += 1
        
        # Check if usage exceeds limit
        exceeds_limit = usage_count >= self.thresholds['promotion_usage_limit']
        
        # Calculate risk score
        if exceeds_limit:
            risk_score = 0.8  # High risk
        elif usage_count > 0:
            risk_score = 0.4  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'usage_count': usage_count,
            'exceeds_limit': exceeds_limit,
            'risk_score': risk_score
        }
    
    def check_multi_account_abuse(self, promotion_data, user_data):
        """
        Check if multiple accounts from the same source are using the same promotion.
        
        Parameters:
        -----------
        promotion_data : pd.Series
            Data about the promotion usage
        user_data : pd.Series
            Data about the user
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        promotion_code = promotion_data.get('promotion_code', None)
        ip_address = user_data.get('ip_address', None)
        device_id = user_data.get('device_id', None)
        
        if promotion_code is None or (ip_address is None and device_id is None):
            return {
                'accounts_from_source': 0,
                'exceeds_limit': False,
                'risk_score': 0.2
            }
        
        # Count accounts from the same source using this promotion
        ip_accounts = set()
        device_accounts = set()
        
        for usage in self.promotion_usage_db.get(promotion_code, []):
            if ip_address is not None and usage.get('ip_address') == ip_address:
                ip_accounts.add(usage['user_id'])
            
            if device_id is not None and usage.get('device_id') == device_id:
                device_accounts.add(usage['user_id'])
        
        # Use the larger count
        accounts_from_source = max(len(ip_accounts), len(device_accounts))
        
        # Check if usage exceeds limit
        exceeds_limit = accounts_from_source >= self.thresholds['promotion_multi_account_limit']
        
        # Calculate risk score
        if exceeds_limit:
            risk_score = 0.9  # Very high risk
        elif accounts_from_source > 1:
            risk_score = 0.5  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'accounts_from_source': accounts_from_source,
            'exceeds_limit': exceeds_limit,
            'risk_score': risk_score
        }
    
    def check_order_manipulation(self, promotion_data, order_data):
        """
        Check for order manipulation to abuse promotions.
        
        Parameters:
        -----------
        promotion_data : pd.Series
            Data about the promotion usage
        order_data : pd.Series
            Data about the order
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract relevant data
        promotion_min_order = promotion_data.get('minimum_order_value', 0)
        order_amount = order_data.get('amount', 0)
        
        # Check for cart padding (order amount just above the minimum)
        cart_padding = False
        padding_ratio = 0.0
        
        if promotion_min_order > 0:
            # Calculate how close the order is to the minimum
            padding_amount = order_amount - promotion_min_order
            padding_ratio = padding_amount / promotion_min_order
            
            # If order is within 5% of minimum, flag as potential padding
            cart_padding = 0 <= padding_ratio <= 0.05
        
        # Check for item return risk
        return_risk_items = []
        
        if 'items' in order_data:
            items = order_data['items']
            
            if isinstance(items, list):
                # Look for patterns that suggest items will be returned
                for item in items:
                    # Example: Check if there are multiple quantities of the same expensive item
                    if item.get('quantity', 1) > 1 and item.get('price', 0) > 100:
                        return_risk_items.append(item.get('item_id', None))
        
        # Calculate risk score
        risk_score = 0.0
        
        if cart_padding:
            risk_score += 0.5  # Medium-high risk for cart padding
        
        if return_risk_items:
            risk_score += 0.4  # Medium risk for return risk items
        
        # Cap the risk score at 1.0
        risk_score = min(1.0, risk_score)
        
        return {
            'cart_padding': cart_padding,
            'padding_ratio': padding_ratio,
            'return_risk_items': return_risk_items,
            'risk_score': risk_score
        }
    
    def check_promotion_timing(self, promotion_data):
        """
        Check for suspicious timing patterns in promotion usage.
        
        Parameters:
        -----------
        promotion_data : pd.Series
            Data about the promotion usage
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        promotion_code = promotion_data.get('promotion_code', None)
        timestamp = promotion_data.get('timestamp', None)
        
        if promotion_code is None or timestamp is None:
            return {
                'suspicious_timing': False,
                'risk_score': 0.1
            }
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Check for suspicious timing patterns
        suspicious_timing = False
        time_since_launch = None
        
        # If we have promotion launch time, check how quickly it was used
        if 'launch_time' in promotion_data:
            launch_time = promotion_data['launch_time']
            
            if isinstance(launch_time, str):
                launch_time = pd.to_datetime(launch_time)
            
            # Calculate time since launch in minutes
            time_since_launch = (timestamp - launch_time).total_seconds() / 60
            
            # If used within 5 minutes of launch, might be suspicious
            if time_since_launch < 5:
                suspicious_timing = True
        
        # Check for rapid successive uses
        rapid_succession = False
        
        if promotion_code in self.promotion_usage_db:
            recent_usages = [
                usage for usage in self.promotion_usage_db[promotion_code]
                if (timestamp - pd.to_datetime(usage['timestamp'])).total_seconds() / 60 < 10
            ]
            
            # If there are multiple recent usages, might be suspicious
            rapid_succession = len(recent_usages) >= 3
        
        # Calculate risk score
        risk_score = 0.0
        
        if suspicious_timing:
            risk_score += 0.4  # Medium risk for suspicious timing
        
        if rapid_succession:
            risk_score += 0.5  # Medium-high risk for rapid succession
        
        # Cap the risk score at 1.0
        risk_score = min(1.0, risk_score)
        
        return {
            'suspicious_timing': suspicious_timing,
            'rapid_succession': rapid_succession,
            'time_since_launch': time_since_launch,
            'risk_score': risk_score
        }
    
    def update_promotion_usage(self, promotion_data):
        """
        Update the promotion usage database with a new usage record.
        
        Parameters:
        -----------
        promotion_data : pd.Series
            Data about the promotion usage
            
        Returns:
        --------
        None
        """
        promotion_code = promotion_data.get('promotion_code', None)
        
        if promotion_code is None:
            return
        
        # Create usage record
        usage_record = {
            'user_id': promotion_data.get('user_id', None),
            'order_id': promotion_data.get('order_id', None),
            'timestamp': promotion_data.get('timestamp', None),
            'ip_address': promotion_data.get('ip_address', None),
            'device_id': promotion_data.get('device_id', None),
            'amount': promotion_data.get('amount', None)
        }
        
        # Add to database
        self.promotion_usage_db[promotion_code].append(usage_record)
    
    #
    # Refund Fraud Detection
    #
    
    def detect_refund_fraud(self, refund_data, order_data=None, user_data=None):
        """
        Detect potential refund fraud.
        
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
            Analysis results with risk scores
        """
        # Convert data to Series if they're dicts
        if isinstance(refund_data, dict):
            refund_data = pd.Series(refund_data)
        
        if order_data is not None and isinstance(order_data, dict):
            order_data = pd.Series(order_data)
        
        if user_data is not None and isinstance(user_data, dict):
            user_data = pd.Series(user_data)
        
        # Initialize results
        results = {
            'refund_id': refund_data.get('refund_id', None),
            'order_id': refund_data.get('order_id', None),
            'user_id': refund_data.get('user_id', None),
            'timestamp': refund_data.get('timestamp', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # Check 1: Refund frequency
        frequency_risk = self.check_refund_frequency(refund_data)
        results['checks']['refund_frequency'] = frequency_risk
        results['risk_scores']['refund_frequency'] = frequency_risk['risk_score']
        
        # Check 2: Refund timing
        if order_data is not None:
            timing_risk = self.check_refund_timing(refund_data, order_data)
            results['checks']['refund_timing'] = timing_risk
            results['risk_scores']['refund_timing'] = timing_risk['risk_score']
        
        # Check 3: Refund reason analysis
        reason_risk = self.analyze_refund_reason(refund_data)
        results['checks']['refund_reason'] = reason_risk
        results['risk_scores']['refund_reason'] = reason_risk['risk_score']
        
        # Check 4: User refund history
        if user_data is not None:
            history_risk = self.check_user_refund_history(refund_data, user_data)
            results['checks']['user_history'] = history_risk
            results['risk_scores']['user_history'] = history_risk['risk_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'refund_frequency': 0.3,
            'refund_timing': 0.2,
            'refund_reason': 0.2,
            'user_history': 0.3
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
            results['recommended_action'] = 'deny_refund'
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
            results['recommended_action'] = 'manual_review'
        else:
            results['risk_level'] = 'low'
            results['recommended_action'] = 'approve_refund'
        
        # Update refund history database
        self.update_refund_history(refund_data)
        
        return results
    
    def check_refund_frequency(self, refund_data):
        """
        Check if a user is requesting refunds too frequently.
        
        Parameters:
        -----------
        refund_data : pd.Series
            Data about the refund request
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        user_id = refund_data.get('user_id', None)
        timestamp = refund_data.get('timestamp', None)
        
        if user_id is None or timestamp is None:
            return {
                'refund_count': 0,
                'exceeds_limit': False,
                'risk_score': 0.1
            }
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Calculate time window
        time_window = timestamp - pd.Timedelta(days=self.thresholds['refund_frequency_days'])
        
        # Count refunds in the time window
        refund_count = 0
        
        for refund in self.refund_history_db.get(user_id, []):
            refund_time = pd.to_datetime(refund['timestamp'])
            
            if refund_time >= time_window:
                refund_count += 1
        
        # Check if frequency exceeds limit
        exceeds_limit = refund_count >= self.thresholds['max_refunds_in_window']
        
        # Calculate risk score
        if exceeds_limit:
            risk_score = 0.8  # High risk
        elif refund_count > 1:
            risk_score = 0.4  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'refund_count': refund_count,
            'time_window_days': self.thresholds['refund_frequency_days'],
            'exceeds_limit': exceeds_limit,
            'risk_score': risk_score
        }
    
    def check_refund_timing(self, refund_data, order_data):
        """
        Check the timing of a refund request relative to the order.
        
        Parameters:
        -----------
        refund_data : pd.Series
            Data about the refund request
        order_data : pd.Series
            Data about the original order
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract timestamps
        refund_time = refund_data.get('timestamp', None)
        order_time = order_data.get('timestamp', None)
        delivery_time = order_data.get('delivery_date', None)
        
        if refund_time is None or order_time is None:
            return {
                'days_since_order': None,
                'days_since_delivery': None,
                'suspicious_timing': False,
                'risk_score': 0.2
            }
        
        # Convert timestamps to datetime if they're strings
        if isinstance(refund_time, str):
            refund_time = pd.to_datetime(refund_time)
        
        if isinstance(order_time, str):
            order_time = pd.to_datetime(order_time)
        
        if delivery_time is not None and isinstance(delivery_time, str):
            delivery_time = pd.to_datetime(delivery_time)
        
        # Calculate time differences
        days_since_order = (refund_time - order_time).total_seconds() / (24 * 3600)
        
        days_since_delivery = None
        if delivery_time is not None:
            days_since_delivery = (refund_time - delivery_time).total_seconds() / (24 * 3600)
        
        # Check for suspicious timing
        suspicious_timing = False
        
        # Very quick refund request after order
        if days_since_order < 1:
            suspicious_timing = True
        
        # Very quick refund request after delivery
        if days_since_delivery is not None and days_since_delivery < 0.5:  # Less than 12 hours
            suspicious_timing = True
        
        # Refund request just before return policy expiration
        if 'return_policy_days' in order_data:
            return_policy_days = order_data['return_policy_days']
            days_before_expiration = return_policy_days - days_since_order
            
            if 0 <= days_before_expiration <= 1:  # Within 1 day of expiration
                suspicious_timing = True
        
        # Calculate risk score
        if suspicious_timing:
            risk_score = 0.7  # High risk
        elif days_since_order < 2:
            risk_score = 0.4  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'days_since_order': days_since_order,
            'days_since_delivery': days_since_delivery,
            'suspicious_timing': suspicious_timing,
            'risk_score': risk_score
        }
    
    def analyze_refund_reason(self, refund_data):
        """
        Analyze the reason given for a refund request.
        
        Parameters:
        -----------
        refund_data : pd.Series
            Data about the refund request
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract refund reason
        reason = refund_data.get('reason', 'unknown')
        
        # Define risk levels for different reasons
        reason_risk = {
            'changed_mind': 0.6,          # High risk - just changed mind
            'not_as_described': 0.3,      # Medium-low risk - legitimate complaint
            'defective': 0.2,             # Low risk - legitimate complaint
            'wrong_item': 0.2,            # Low risk - legitimate complaint
            'arrived_late': 0.4,          # Medium risk - may be legitimate
            'no_longer_needed': 0.7,      # High risk - suspicious reason
            'better_price_elsewhere': 0.5, # Medium-high risk - suspicious reason
            'accidental_purchase': 0.6,   # High risk - suspicious reason
            'unknown': 0.5                # Medium-high risk - no clear reason
        }
        
        # Get risk score for this reason
        risk_score = reason_risk.get(reason, 0.5)
        
        # Check for additional details
        has_details = 'reason_details' in refund_data and refund_data['reason_details'] is not None
        
        # Adjust risk score based on details
        if has_details:
            # If details are provided, slightly reduce risk
            risk_score = max(0.1, risk_score - 0.1)
        
        return {
            'reason': reason,
            'has_details': has_details,
            'risk_score': risk_score
        }
    
    def check_user_refund_history(self, refund_data, user_data):
        """
        Check a user's refund history for patterns of abuse.
        
        Parameters:
        -----------
        refund_data : pd.Series
            Data about the refund request
        user_data : pd.Series
            Data about the user
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        user_id = refund_data.get('user_id', None)
        
        if user_id is None:
            return {
                'total_refunds': 0,
                'refund_ratio': 0.0,
                'high_refund_ratio': False,
                'risk_score': 0.2
            }
        
        # Get user's order and refund counts
        total_orders = user_data.get('order_count', 0)
        
        # Count total refunds
        total_refunds = len(self.refund_history_db.get(user_id, []))
        
        # Calculate refund ratio
        refund_ratio = total_refunds / max(1, total_orders)
        
        # Check if ratio exceeds threshold
        high_refund_ratio = refund_ratio > self.thresholds['high_refund_ratio']
        
        # Check for repeat refund reasons
        reason_counts = Counter()
        
        for refund in self.refund_history_db.get(user_id, []):
            reason_counts[refund.get('reason', 'unknown')] += 1
        
        # Find most common reason
        most_common_reason = reason_counts.most_common(1)
        repeat_reason = most_common_reason[0][1] >= 3 if most_common_reason else False
        
        # Calculate risk score
        risk_score = 0.0
        
        if high_refund_ratio:
            risk_score += 0.6  # High risk for high refund ratio
        
        if repeat_reason:
            risk_score += 0.3  # Medium risk for repeat reasons
        
        if total_refunds > 5:
            risk_score += 0.2  # Additional risk for many refunds
        
        # Cap the risk score at 1.0
        risk_score = min(1.0, risk_score)
        
        return {
            'total_refunds': total_refunds,
            'total_orders': total_orders,
            'refund_ratio': refund_ratio,
            'high_refund_ratio': high_refund_ratio,
            'repeat_reason': repeat_reason,
            'risk_score': risk_score
        }
    
    def update_refund_history(self, refund_data):
        """
        Update the refund history database with a new refund record.
        
        Parameters:
        -----------
        refund_data : pd.Series
            Data about the refund request
            
        Returns:
        --------
        None
        """
        user_id = refund_data.get('user_id', None)
        
        if user_id is None:
            return
        
        # Create refund record
        refund_record = {
            'refund_id': refund_data.get('refund_id', None),
            'order_id': refund_data.get('order_id', None),
            'timestamp': refund_data.get('timestamp', None),
            'amount': refund_data.get('amount', None),
            'reason': refund_data.get('reason', 'unknown'),
            'status': refund_data.get('status', 'pending')
        }
        
        # Add to database
        self.refund_history_db[user_id].append(refund_record)
    
    #
    # Bot Attack Detection
    #
    
    def detect_bot_activity(self, request_data, session_data=None):
        """
        Detect potential bot or automated attack activity.
        
        Parameters:
        -----------
        request_data : dict or pd.Series
            Data about the current request
        session_data : dict or pd.Series
            Data about the user session
            
        Returns:
        --------
        dict
            Analysis results with risk scores
        """
        # Convert data to Series if they're dicts
        if isinstance(request_data, dict):
            request_data = pd.Series(request_data)
        
        if session_data is not None and isinstance(session_data, dict):
            session_data = pd.Series(session_data)
        
        # Initialize results
        results = {
            'request_id': request_data.get('request_id', None),
            'session_id': request_data.get('session_id', None),
            'ip_address': request_data.get('ip_address', None),
            'timestamp': request_data.get('timestamp', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # Check 1: Request rate analysis
        rate_risk = self.analyze_request_rate(request_data, session_data)
        results['checks']['request_rate'] = rate_risk
        results['risk_scores']['request_rate'] = rate_risk['risk_score']
        
        # Check 2: User agent analysis
        agent_risk = self.analyze_user_agent(request_data)
        results['checks']['user_agent'] = agent_risk
        results['risk_scores']['user_agent'] = agent_risk['risk_score']
        
        # Check 3: Behavioral analysis
        if session_data is not None:
            behavior_risk = self.analyze_session_behavior(session_data)
            results['checks']['behavior'] = behavior_risk
            results['risk_scores']['behavior'] = behavior_risk['risk_score']
        
        # Check 4: IP reputation
        ip_risk = self.check_ip_reputation(request_data)
        results['checks']['ip_reputation'] = ip_risk
        results['risk_scores']['ip_reputation'] = ip_risk['risk_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'request_rate': 0.3,
            'user_agent': 0.2,
            'behavior': 0.3,
            'ip_reputation': 0.2
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
            results['recommended_action'] = 'block_and_captcha'
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
            results['recommended_action'] = 'rate_limit'
        else:
            results['risk_level'] = 'low'
            results['recommended_action'] = 'allow'
        
        return results
    
    def analyze_request_rate(self, request_data, session_data=None):
        """
        Analyze request rate for potential bot activity.
        
        Parameters:
        -----------
        request_data : pd.Series
            Data about the current request
        session_data : pd.Series
            Data about the user session
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        ip_address = request_data.get('ip_address', None)
        session_id = request_data.get('session_id', None)
        timestamp = request_data.get('timestamp', None)
        
        if ip_address is None or timestamp is None:
            return {
                'requests_per_minute': 0,
                'exceeds_limit': False,
                'risk_score': 0.2
            }
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Calculate time window (1 minute)
        time_window = timestamp - pd.Timedelta(minutes=1)
        
        # Count requests in the time window
        requests_per_minute = 0
        
        if session_data is not None and 'requests' in session_data:
            requests = session_data['requests']
            
            if isinstance(requests, list):
                for req in requests:
                    req_time = pd.to_datetime(req['timestamp'])
                    
                    if req_time >= time_window:
                        requests_per_minute += 1
        
        # Check if rate exceeds limit
        exceeds_limit = requests_per_minute > self.thresholds['bot_request_rate_limit']
        
        # Calculate risk score
        if exceeds_limit:
            # Scale risk based on how much the limit is exceeded
            excess_factor = requests_per_minute / self.thresholds['bot_request_rate_limit']
            risk_score = min(0.9, 0.5 + (excess_factor - 1) * 0.1)
        elif requests_per_minute > self.thresholds['bot_request_rate_limit'] / 2:
            risk_score = 0.4  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'requests_per_minute': requests_per_minute,
            'rate_limit': self.thresholds['bot_request_rate_limit'],
            'exceeds_limit': exceeds_limit,
            'risk_score': risk_score
        }
    
    def analyze_user_agent(self, request_data):
        """
        Analyze user agent string for bot indicators.
        
        Parameters:
        -----------
        request_data : pd.Series
            Data about the current request
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        user_agent = request_data.get('user_agent', '')
        
        if not user_agent:
            return {
                'is_bot': True,
                'bot_indicators': ['missing_user_agent'],
                'risk_score': 0.8  # High risk for missing user agent
            }
        
        # Check for known bot patterns
        bot_indicators = []
        
        # Known bot keywords
        bot_keywords = [
            'bot', 'crawl', 'spider', 'scrape', 'headless', 'phantomjs', 'selenium',
            'automation', 'http-client', 'python-requests', 'curl', 'wget'
        ]
        
        for keyword in bot_keywords:
            if keyword.lower() in user_agent.lower():
                bot_indicators.append(f'contains_{keyword}')
        
        # Check for missing browser information
        if not any(browser in user_agent.lower() for browser in ['chrome', 'firefox', 'safari', 'edge', 'opera']):
            bot_indicators.append('missing_browser')
        
        # Check for unusual user agent length
        if len(user_agent) < 20:
            bot_indicators.append('short_user_agent')
        
        # Check for inconsistent browser/OS combinations
        if ('windows' in user_agent.lower() and 'safari' in user_agent.lower() and 'chrome' not in user_agent.lower()):
            bot_indicators.append('inconsistent_browser_os')
        
        if ('mac' in user_agent.lower() and 'edge' in user_agent.lower() and 'edg/' not in user_agent.lower()):
            bot_indicators.append('inconsistent_browser_os')
        
        # Determine if it's a bot
        is_bot = len(bot_indicators) > 0
        
        # Calculate risk score
        if is_bot:
            risk_score = min(0.9, 0.5 + len(bot_indicators) * 0.1)
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'is_bot': is_bot,
            'bot_indicators': bot_indicators,
            'risk_score': risk_score
        }
    
    def analyze_session_behavior(self, session_data):
        """
        Analyze session behavior for bot indicators.
        
        Parameters:
        -----------
        session_data : pd.Series
            Data about the user session
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract session data
        session_duration = session_data.get('duration_seconds', 0)
        page_views = session_data.get('page_views', 0)
        mouse_movements = session_data.get('mouse_movements', 0)
        clicks = session_data.get('clicks', 0)
        
        # Check for behavioral indicators
        behavior_indicators = []
        
        # Check for very short session duration
        if session_duration < self.thresholds['bot_session_duration_min']:
            behavior_indicators.append('short_session')
        
        # Check for lack of mouse movements
        if page_views > 1 and mouse_movements < 5:
            behavior_indicators.append('no_mouse_movement')
        
        # Check for unusual click patterns
        if page_views > 0 and clicks == 0:
            behavior_indicators.append('no_clicks')
        
        # Check for unusual navigation patterns
        if 'navigation_path' in session_data:
            nav_path = session_data['navigation_path']
            
            if isinstance(nav_path, list) and len(nav_path) > 1:
                # Check for too rapid page transitions
                for i in range(1, len(nav_path)):
                    time_diff = (pd.to_datetime(nav_path[i]['timestamp']) - 
                                pd.to_datetime(nav_path[i-1]['timestamp'])).total_seconds()
                    
                    if time_diff < 1:  # Less than 1 second between pages
                        behavior_indicators.append('rapid_navigation')
                        break
        
        # Determine if behavior is suspicious
        suspicious_behavior = len(behavior_indicators) > 0
        
        # Calculate risk score
        if suspicious_behavior:
            risk_score = min(0.9, 0.4 + len(behavior_indicators) * 0.1)
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'suspicious_behavior': suspicious_behavior,
            'behavior_indicators': behavior_indicators,
            'risk_score': risk_score
        }
    
    def check_ip_reputation(self, request_data):
        """
        Check IP reputation for potential bot or malicious activity.
        
        Parameters:
        -----------
        request_data : pd.Series
            Data about the current request
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        ip_address = request_data.get('ip_address', None)
        
        if ip_address is None:
            return {
                'ip_reputation': 'unknown',
                'risk_score': 0.5  # Medium risk for unknown IP
            }
        
        # Check if IP is in our database
        reputation_score = self.ip_reputation_db.get(ip_address, None)
        
        if reputation_score is not None:
            # Use stored reputation score
            risk_score = reputation_score
        else:
            # Check for suspicious IP characteristics
            
            # Check if IP is in a data center range
            is_datacenter = False
            try:
                ip = ipaddress.ip_address(ip_address)
                
                # Example datacenter ranges (simplified)
                datacenter_ranges = [
                    ipaddress.ip_network('13.32.0.0/15'),  # AWS CloudFront
                    ipaddress.ip_network('13.224.0.0/14'),  # AWS CloudFront
                    ipaddress.ip_network('35.184.0.0/13'),  # Google Cloud
                    ipaddress.ip_network('104.196.0.0/14')  # Google Cloud
                ]
                
                for network in datacenter_ranges:
                    if ip in network:
                        is_datacenter = True
                        break
            except:
                pass
            
            # Check if IP is a Tor exit node (simplified example)
            is_tor = ip_address in ['101.36.72.0', '171.25.193.77']  # Example Tor exit nodes
            
            # Check if IP is a proxy (simplified example)
            is_proxy = ip_address in ['104.129.18.0', '154.16.45.0']  # Example proxy IPs
            
            # Calculate risk score
            risk_score = 0.2  # Base risk
            
            if is_datacenter:
                risk_score += 0.3
            
            if is_tor:
                risk_score += 0.4
            
            if is_proxy:
                risk_score += 0.3
            
            # Cap the risk score at 1.0
            risk_score = min(1.0, risk_score)
            
            # Store in database for future reference
            self.ip_reputation_db[ip_address] = risk_score
        
        # Determine reputation category
        if risk_score >= 0.7:
            reputation = 'high_risk'
        elif risk_score >= 0.4:
            reputation = 'medium_risk'
        else:
            reputation = 'low_risk'
        
        return {
            'ip_reputation': reputation,
            'risk_score': risk_score
        }
    
    #
    # Synthetic Identity Detection
    #
    
    def detect_synthetic_identity(self, user_data, registration_data=None):
        """
        Detect potential synthetic identity fraud.
        
        Parameters:
        -----------
        user_data : dict or pd.Series
            Data about the user
        registration_data : dict or pd.Series
            Data about the user registration process
            
        Returns:
        --------
        dict
            Analysis results with risk scores
        """
        # Convert data to Series if they're dicts
        if isinstance(user_data, dict):
            user_data = pd.Series(user_data)
        
        if registration_data is not None and isinstance(registration_data, dict):
            registration_data = pd.Series(registration_data)
        
        # Initialize results
        results = {
            'user_id': user_data.get('user_id', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # Check 1: Identity consistency
        consistency_risk = self.check_identity_consistency(user_data)
        results['checks']['identity_consistency'] = consistency_risk
        results['risk_scores']['identity_consistency'] = consistency_risk['risk_score']
        
        # Check 2: Registration behavior
        if registration_data is not None:
            registration_risk = self.analyze_registration_behavior(registration_data)
            results['checks']['registration_behavior'] = registration_risk
            results['risk_scores']['registration_behavior'] = registration_risk['risk_score']
        
        # Check 3: Digital footprint
        footprint_risk = self.analyze_digital_footprint(user_data)
        results['checks']['digital_footprint'] = footprint_risk
        results['risk_scores']['digital_footprint'] = footprint_risk['risk_score']
        
        # Check 4: Pattern matching
        pattern_risk = self.check_synthetic_patterns(user_data)
        results['checks']['pattern_matching'] = pattern_risk
        results['risk_scores']['pattern_matching'] = pattern_risk['risk_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'identity_consistency': 0.3,
            'registration_behavior': 0.2,
            'digital_footprint': 0.2,
            'pattern_matching': 0.3
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
        if results['overall_risk_score'] >= self.thresholds['synthetic_identity_score']:
            results['risk_level'] = 'high'
            results['recommended_action'] = 'additional_verification'
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
            results['recommended_action'] = 'monitor'
        else:
            results['risk_level'] = 'low'
            results['recommended_action'] = 'allow'
        
        return results
    
    def check_identity_consistency(self, user_data):
        """
        Check for consistency in identity information.
        
        Parameters:
        -----------
        user_data : pd.Series
            Data about the user
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract identity data
        name = user_data.get('name', '')
        email = user_data.get('email', '')
        phone = user_data.get('phone', '')
        address = user_data.get('address', '')
        
        # Check for inconsistencies
        inconsistencies = []
        
        # Check if email contains name
        if name and email:
            name_parts = name.lower().split()
            email_username = email.split('@')[0].lower()
            
            if not any(part in email_username for part in name_parts if len(part) > 2):
                inconsistencies.append('email_name_mismatch')
        
        # Check for mismatched country codes
        if phone and address:
            # Extract country from address (simplified)
            address_country = None
            for country in ['USA', 'UK', 'Canada', 'Australia']:
                if country in address:
                    address_country = country
                    break
            
            # Extract country code from phone (simplified)
            phone_country = None
            if phone.startswith('+1'):
                phone_country = 'USA'
            elif phone.startswith('+44'):
                phone_country = 'UK'
            elif phone.startswith('+61'):
                phone_country = 'Australia'
            
            if address_country and phone_country and address_country != phone_country:
                inconsistencies.append('phone_address_country_mismatch')
        
        # Check for temporary email domains
        if email:
            temp_email_domains = ['temp-mail.org', 'guerrillamail.com', 'mailinator.com', 'yopmail.com']
            if any(domain in email for domain in temp_email_domains):
                inconsistencies.append('temporary_email')
        
        # Calculate risk score
        if len(inconsistencies) >= 2:
            risk_score = 0.8  # High risk
        elif len(inconsistencies) == 1:
            risk_score = 0.5  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'inconsistencies': inconsistencies,
            'risk_score': risk_score
        }
    
    def analyze_registration_behavior(self, registration_data):
        """
        Analyze user registration behavior for synthetic identity indicators.
        
        Parameters:
        -----------
        registration_data : pd.Series
            Data about the user registration process
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract registration data
        registration_time = registration_data.get('timestamp', None)
        form_fill_time = registration_data.get('form_fill_time_seconds', None)
        field_changes = registration_data.get('field_changes', None)
        
        # Check for suspicious behaviors
        suspicious_behaviors = []
        
        # Check for very quick form filling
        if form_fill_time is not None and form_fill_time < 10:  # Less than 10 seconds
            suspicious_behaviors.append('quick_form_fill')
        
        # Check for lack of field changes/corrections
        if field_changes is not None and field_changes == 0:
            suspicious_behaviors.append('no_field_changes')
        
        # Check for registration during unusual hours
        if registration_time is not None:
            if isinstance(registration_time, str):
                registration_time = pd.to_datetime(registration_time)
            
            hour = registration_time.hour
            if hour >= 1 and hour <= 5:  # Between 1 AM and 5 AM
                suspicious_behaviors.append('unusual_hour')
        
        # Check for copy-paste behavior
        if 'copy_paste_detected' in registration_data and registration_data['copy_paste_detected']:
            suspicious_behaviors.append('copy_paste_detected')
        
        # Calculate risk score
        if len(suspicious_behaviors) >= 2:
            risk_score = 0.7  # High risk
        elif len(suspicious_behaviors) == 1:
            risk_score = 0.4  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'suspicious_behaviors': suspicious_behaviors,
            'risk_score': risk_score
        }
    
    def analyze_digital_footprint(self, user_data):
        """
        Analyze user's digital footprint for synthetic identity indicators.
        
        Parameters:
        -----------
        user_data : pd.Series
            Data about the user
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract digital footprint data
        email = user_data.get('email', '')
        social_profiles = user_data.get('social_profiles', [])
        account_age_days = user_data.get('account_age_days', 0)
        
        # Check for suspicious indicators
        suspicious_indicators = []
        
        # Check for new email account
        if 'email_age_days' in user_data and user_data['email_age_days'] < 30:
            suspicious_indicators.append('new_email')
        
        # Check for lack of social profiles
        if not social_profiles or len(social_profiles) == 0:
            suspicious_indicators.append('no_social_profiles')
        
        # Check for new account
        if account_age_days < 7:
            suspicious_indicators.append('new_account')
        
        # Check for digital presence
        if 'digital_presence_score' in user_data:
            digital_presence = user_data['digital_presence_score']
            if digital_presence < 0.3:  # Low digital presence
                suspicious_indicators.append('low_digital_presence')
        
        # Calculate risk score
        if len(suspicious_indicators) >= 3:
            risk_score = 0.8  # High risk
        elif len(suspicious_indicators) >= 1:
            risk_score = 0.5  # Medium risk
        else:
            risk_score = 0.2  # Low-medium risk
        
        return {
            'suspicious_indicators': suspicious_indicators,
            'risk_score': risk_score
        }
    
    def check_synthetic_patterns(self, user_data):
        """
        Check for known patterns of synthetic identities.
        
        Parameters:
        -----------
        user_data : pd.Series
            Data about the user
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Extract user data
        name = user_data.get('name', '')
        email = user_data.get('email', '')
        phone = user_data.get('phone', '')
        address = user_data.get('address', '')
        dob = user_data.get('date_of_birth', None)
        
        # Check for synthetic patterns
        synthetic_patterns = []
        
        # Check for sequential digits in phone number
        if phone:
            # Remove non-digit characters
            digits = ''.join(filter(str.isdigit, phone))
            
            # Check for sequential patterns
            for i in range(len(digits) - 3):
                if digits[i:i+4] in ['1234', '2345', '3456', '4567', '5678', '6789', '9876', '8765', '7654', '6543', '5432', '4321']:
                    synthetic_patterns.append('sequential_phone_digits')
                    break
        
        # Check for common synthetic name patterns
        if name:
            # Check for names that are too generic or common in synthetic identities
            generic_first_names = ['John', 'Jane', 'Michael', 'David', 'Mary']
            generic_last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown']
            
            name_parts = name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = name_parts[-1]
                
                if first_name in generic_first_names and last_name in generic_last_names:
                    synthetic_patterns.append('generic_name')
        
        # Check for age inconsistency
        if dob and 'registration_date' in user_data:
            if isinstance(dob, str):
                dob = pd.to_datetime(dob)
            
            registration_date = user_data['registration_date']
            if isinstance(registration_date, str):
                registration_date = pd.to_datetime(registration_date)
            
            age = (registration_date - dob).days / 365.25
            
            # Check for round ages (exactly 25, 30, 35, etc.)
            if age % 5 == 0:
                synthetic_patterns.append('round_age')
        
        # Check for address patterns
        if address:
            # Check for addresses with round numbers
            address_numbers = re.findall(r'\b\d+\b', address)
            if address_numbers and all(int(num) % 100 == 0 for num in address_numbers):
                synthetic_patterns.append('round_address_numbers')
        
        # Calculate risk score
        if len(synthetic_patterns) >= 2:
            risk_score = 0.8  # High risk
        elif len(synthetic_patterns) == 1:
            risk_score = 0.5  # Medium risk
        else:
            risk_score = 0.2  # Low-medium risk
        
        return {
            'synthetic_patterns': synthetic_patterns,
            'risk_score': risk_score
        }
    
    #
    # Model Training and Evaluation
    #
    
    def train_bot_detection_model(self, X_train, y_train, optimize_hyperparams=False, class_weight='balanced'):
        """
        Train a machine learning model for bot detection.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (bot=1, human=0)
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        class_weight : str or dict
            Class weights to handle imbalanced data
            
        Returns:
        --------
        self : AdditionalFraudDetector
            Returns self for method chaining
        """
        # Create and train model
        self.bot_detection_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight=class_weight
        )
        
        self.bot_detection_model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance['bot_detection'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.bot_detection_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def train_synthetic_identity_model(self, X_train, y_train, optimize_hyperparams=False, class_weight='balanced'):
        """
        Train a machine learning model for synthetic identity detection.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (synthetic=1, genuine=0)
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        class_weight : str or dict
            Class weights to handle imbalanced data
            
        Returns:
        --------
        self : AdditionalFraudDetector
            Returns self for method chaining
        """
        # Create and train model
        self.synthetic_identity_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight=class_weight
        )
        
        self.synthetic_identity_model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance['synthetic_identity'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.synthetic_identity_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def evaluate_model(self, model_type, X_test, y_test):
        """
        Evaluate a trained model on test data.
        
        Parameters:
        -----------
        model_type : str
            Type of model to evaluate ('bot_detection' or 'synthetic_identity')
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if model_type == 'bot_detection':
            model = self.bot_detection_model
        elif model_type == 'synthetic_identity':
            model = self.synthetic_identity_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model is None:
            raise ValueError(f"Model not trained. Call train_{model_type}_model() first.")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
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
    
    def plot_feature_importance(self, model_type, top_n=20, save_path=None):
        """
        Plot feature importance from a trained model.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('bot_detection' or 'synthetic_identity')
        top_n : int
            Number of top features to display
        save_path : str
            Path to save the plot (if None, plot is displayed)
            
        Returns:
        --------
        None
        """
        if model_type not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_type}. Train a model first.")
        
        # Get top N features
        top_features = self.feature_importance[model_type].head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Features for {model_type.replace("_", " ").title()}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_model(self, filename='additional_fraud_models.pkl'):
        """
        Save the trained models and configuration to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'bot_detection_model': self.bot_detection_model,
            'synthetic_identity_model': self.synthetic_identity_model,
            'feature_importance': self.feature_importance,
            'thresholds': self.thresholds,
            'promotion_usage_db': dict(self.promotion_usage_db),
            'refund_history_db': dict(self.refund_history_db),
            'ip_reputation_db': self.ip_reputation_db,
            'user_agent_patterns': self.user_agent_patterns
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='additional_fraud_models.pkl'):
        """
        Load trained models and configuration from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
            
        Returns:
        --------
        self : AdditionalFraudDetector
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.bot_detection_model = model_data['bot_detection_model']
        self.synthetic_identity_model = model_data['synthetic_identity_model']
        self.feature_importance = model_data['feature_importance']
        self.thresholds = model_data['thresholds']
        self.promotion_usage_db = defaultdict(list, model_data['promotion_usage_db'])
        self.refund_history_db = defaultdict(list, model_data['refund_history_db'])
        self.ip_reputation_db = model_data['ip_reputation_db']
        self.user_agent_patterns = model_data['user_agent_patterns']
        
        return self
    
    def create_sample_data(self, save_to_csv=True):
        """
        Create sample data for demonstration purposes.
        
        Parameters:
        -----------
        save_to_csv : bool
            Whether to save the generated data to CSV files
            
        Returns:
        --------
        tuple
            (promotions_df, refunds_df, bot_requests_df, synthetic_users_df) DataFrames
        """
        np.random.seed(42)
        
        # Generate promotion abuse data
        n_promotions = 200
        
        promotion_ids = [f"P{i:05d}" for i in range(1, n_promotions + 1)]
        user_ids = [f"U{np.random.randint(1, 100):05d}" for _ in range(n_promotions)]
        order_ids = [f"O{i:06d}" for i in range(1, n_promotions + 1)]
        
        promotions_data = {
            'promotion_id': promotion_ids,
            'promotion_code': [f"PROMO{np.random.randint(1, 20):02d}" for _ in range(n_promotions)],
            'user_id': user_ids,
            'order_id': order_ids,
            'timestamp': pd.date_range(start='2023-01-01', periods=n_promotions, freq='2H'),
            'amount': np.random.uniform(10, 200, n_promotions).round(2),
            'discount_amount': np.random.uniform(5, 50, n_promotions).round(2),
            'minimum_order_value': np.random.choice([0, 50, 100], n_promotions),
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_promotions)],
            'device_id': [f"D{np.random.randint(1, 50):05d}" for _ in range(n_promotions)],
            'is_abuse': np.random.choice([True, False], n_promotions, p=[0.2, 0.8])
        }
        
        # Add some patterns for abusive cases
        for i in range(n_promotions):
            if promotions_data['is_abuse'][i]:
                # Make some abusive patterns
                if np.random.random() < 0.5:
                    # Multiple uses of same code by same user
                    idx = np.random.randint(0, i)
                    promotions_data['promotion_code'][i] = promotions_data['promotion_code'][idx]
                    promotions_data['user_id'][i] = promotions_data['user_id'][idx]
                else:
                    # Multiple accounts from same IP
                    idx = np.random.randint(0, i)
                    promotions_data['promotion_code'][i] = promotions_data['promotion_code'][idx]
                    promotions_data['ip_address'][i] = promotions_data['ip_address'][idx]
        
        promotions_df = pd.DataFrame(promotions_data)
        
        # Generate refund fraud data
        n_refunds = 150
        
        refund_ids = [f"R{i:05d}" for i in range(1, n_refunds + 1)]
        refund_order_ids = [f"O{np.random.randint(1, 1000):06d}" for _ in range(n_refunds)]
        refund_user_ids = [f"U{np.random.randint(1, 100):05d}" for _ in range(n_refunds)]
        
        refund_reasons = ['defective', 'not_as_described', 'wrong_item', 'arrived_late', 'changed_mind', 'no_longer_needed', 'accidental_purchase']
        
        refunds_data = {
            'refund_id': refund_ids,
            'order_id': refund_order_ids,
            'user_id': refund_user_ids,
            'timestamp': pd.date_range(start='2023-02-01', periods=n_refunds, freq='12H'),
            'amount': np.random.uniform(10, 500, n_refunds).round(2),
            'reason': np.random.choice(refund_reasons, n_refunds),
            'status': np.random.choice(['pending', 'approved', 'rejected'], n_refunds, p=[0.2, 0.7, 0.1]),
            'is_fraud': np.random.choice([True, False], n_refunds, p=[0.15, 0.85])
        }
        
        # Add some patterns for fraudulent cases
        for i in range(n_refunds):
            if refunds_data['is_fraud'][i]:
                # Make some fraudulent patterns
                if np.random.random() < 0.6:
                    # Frequent refunds by same user
                    idx = np.random.randint(0, i)
                    refunds_data['user_id'][i] = refunds_data['user_id'][idx]
                    # Use suspicious reasons
                    refunds_data['reason'][i] = np.random.choice(['changed_mind', 'no_longer_needed', 'accidental_purchase'])
        
        refunds_df = pd.DataFrame(refunds_data)
        
        # Generate bot activity data
        n_requests = 300
        
        request_ids = [f"REQ{i:06d}" for i in range(1, n_requests + 1)]
        session_ids = [f"S{np.random.randint(1, 50):05d}" for _ in range(n_requests)]
        
        # Generate some realistic and some bot user agents
        real_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        bot_user_agents = [
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "python-requests/2.25.1",
            "curl/7.64.1",
            "Scrapy/2.5.0 (+https://scrapy.org)",
            ""  # Empty user agent
        ]
        
        requests_data = {
            'request_id': request_ids,
            'session_id': session_ids,
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_requests)],
            'timestamp': pd.date_range(start='2023-03-01', periods=n_requests, freq='5min'),
            'user_agent': np.random.choice(real_user_agents + bot_user_agents, n_requests),
            'endpoint': np.random.choice(['/api/products', '/api/cart', '/api/checkout', '/api/search', '/api/user'], n_requests),
            'response_code': np.random.choice([200, 400, 403, 404, 500], n_requests, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
            'is_bot': np.random.choice([True, False], n_requests, p=[0.2, 0.8])
        }
        
        # Add some patterns for bot cases
        for i in range(n_requests):
            if requests_data['is_bot'][i]:
                # Make some bot patterns
                if np.random.random() < 0.7:
                    # Use bot user agent
                    requests_data['user_agent'][i] = np.random.choice(bot_user_agents)
                
                # Create rapid request patterns
                if i > 0 and np.random.random() < 0.8:
                    # Make requests come from same session in rapid succession
                    requests_data['session_id'][i] = requests_data['session_id'][i-1]
                    requests_data['ip_address'][i] = requests_data['ip_address'][i-1]
                    # Adjust timestamp to be very close to previous
                    requests_data['timestamp'][i] = requests_data['timestamp'][i-1] + pd.Timedelta(seconds=np.random.randint(1, 5))
        
        bot_requests_df = pd.DataFrame(requests_data)
        
        # Generate synthetic identity data
        n_users = 100
        
        synthetic_user_ids = [f"U{i:05d}" for i in range(1, n_users + 1)]
        
        # Generate some realistic and some synthetic names
        real_names = [
            "Emily Johnson", "Michael Williams", "Sophia Martinez", "Daniel Thompson",
            "Olivia Rodriguez", "James Anderson", "Emma Wilson", "Benjamin Davis"
        ]
        
        synthetic_names = [
            "John Smith", "Jane Doe", "Mary Johnson", "David Brown", "Michael Williams"
        ]
        
        synthetic_users_data = {
            'user_id': synthetic_user_ids,
            'name': np.random.choice(real_names + synthetic_names, n_users),
            'email': [f"user{i}@{np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'temp-mail.org'])}" for i in range(1, n_users + 1)],
            'phone': [f"+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_users)],
            'address': [f"{np.random.randint(1, 9999)} {np.random.choice(['Main', 'Oak', 'Maple', 'Cedar'])} St, {np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'])}" for _ in range(n_users)],
            'registration_date': pd.date_range(start='2023-01-01', periods=n_users, freq='1D'),
            'account_age_days': np.random.randint(1, 365, n_users),
            'order_count': np.random.randint(0, 20, n_users),
            'is_synthetic': np.random.choice([True, False], n_users, p=[0.2, 0.8])
        }
        
        # Add some patterns for synthetic cases
        for i in range(n_users):
            if synthetic_users_data['is_synthetic'][i]:
                # Make some synthetic patterns
                if np.random.random() < 0.7:
                    # Use synthetic name
                    synthetic_users_data['name'][i] = np.random.choice(synthetic_names)
                
                if np.random.random() < 0.6:
                    # Use temporary email
                    synthetic_users_data['email'][i] = f"user{i}@{np.random.choice(['temp-mail.org', 'guerrillamail.com', 'mailinator.com'])}"
                
                if np.random.random() < 0.5:
                    # Use sequential digits in phone
                    synthetic_users_data['phone'][i] = f"+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-1234"
                
                # New account
                synthetic_users_data['account_age_days'][i] = np.random.randint(1, 10)
                
                # Few or no orders
                synthetic_users_data['order_count'][i] = np.random.randint(0, 2)
        
        synthetic_users_df = pd.DataFrame(synthetic_users_data)
        
        # Save to CSV if requested
        if save_to_csv:
            data_dir = os.path.dirname(self.model_dir)
            data_dir = os.path.join(data_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            promotions_df.to_csv(os.path.join(data_dir, 'sample_promotions.csv'), index=False)
            refunds_df.to_csv(os.path.join(data_dir, 'sample_refunds.csv'), index=False)
            bot_requests_df.to_csv(os.path.join(data_dir, 'sample_bot_requests.csv'), index=False)
            synthetic_users_df.to_csv(os.path.join(data_dir, 'sample_synthetic_users.csv'), index=False)
        
        return promotions_df, refunds_df, bot_requests_df, synthetic_users_df
