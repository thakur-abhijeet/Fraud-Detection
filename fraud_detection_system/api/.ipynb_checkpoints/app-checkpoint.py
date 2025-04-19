"""
Flask API for E-commerce Fraud Detection System

This module implements a RESTful API for the fraud detection system,
providing endpoints for transaction analysis, account takeover prevention,
friendly fraud detection, and other fraud detection mechanisms.
"""

from flask import Flask, request, jsonify
import sys
import os
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta

# Add parent directory to path to import fraud detection modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fraud detection modules
from src.data_preprocessing import DataPreprocessor
from src.credit_card_fraud_detection import CreditCardFraudDetector
from src.account_takeover_prevention import AccountTakeoverDetector
from src.friendly_fraud_detection import FriendlyFraudDetector
from src.additional_fraud_detection import AdditionalFraudDetector
from src.unified_risk_scoring import UnifiedRiskScoringSystem

app = Flask(__name__)

# Initialize fraud detection components
data_preprocessor = DataPreprocessor()
credit_card_detector = CreditCardFraudDetector()
account_takeover_detector = AccountTakeoverDetector()
friendly_fraud_detector = FriendlyFraudDetector()
additional_fraud_detector = AdditionalFraudDetector()
unified_risk_system = UnifiedRiskScoringSystem()

# Global variables to store loaded data
transactions_df = None
users_df = None
login_history_df = None
chargebacks_df = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'message': 'Fraud Detection System API is running'
    })

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """
    Load data from specified file paths.
    
    Expected JSON payload:
    {
        "transactions_file": "/path/to/transactions.csv",
        "users_file": "/path/to/users.csv",
        "login_history_file": "/path/to/login_history.csv",
        "chargebacks_file": "/path/to/chargebacks.csv"
    }
    
    Note: All file paths are optional. If not provided, the system will use sample data.
    """
    global transactions_df, users_df, login_history_df, chargebacks_df
    
    try:
        data = request.get_json()
        
        # Load transactions data
        if 'transactions_file' in data and os.path.exists(data['transactions_file']):
            transactions_df = data_preprocessor.load_data(data['transactions_file'])
            transactions_df = data_preprocessor.preprocess_transaction_data(transactions_df)
        else:
            # Use sample data
            transactions_df = data_preprocessor.generate_sample_transaction_data()
        
        # Load users data
        if 'users_file' in data and os.path.exists(data['users_file']):
            users_df = data_preprocessor.load_data(data['users_file'])
            users_df = data_preprocessor.preprocess_user_data(users_df)
        else:
            # Use sample data
            users_df = data_preprocessor.generate_sample_user_data()
        
        # Load login history data
        if 'login_history_file' in data and os.path.exists(data['login_history_file']):
            login_history_df = data_preprocessor.load_data(data['login_history_file'])
            login_history_df = data_preprocessor.preprocess_login_data(login_history_df)
        else:
            # Use sample data
            login_history_df = data_preprocessor.generate_sample_login_data(users_df)
        
        # Load chargebacks data
        if 'chargebacks_file' in data and os.path.exists(data['chargebacks_file']):
            chargebacks_df = data_preprocessor.load_data(data['chargebacks_file'])
            chargebacks_df = data_preprocessor.preprocess_chargeback_data(chargebacks_df)
        else:
            # Use sample data
            chargebacks_df = data_preprocessor.generate_sample_chargeback_data(transactions_df)
        
        # Initialize fraud detection components with data
        credit_card_detector.train_ml_model(transactions_df)
        account_takeover_detector.train_login_behavior_model(login_history_df)
        friendly_fraud_detector.train_chargeback_model(transactions_df, chargebacks_df)
        
        # Initialize unified risk system
        unified_risk_system.load_models()
        
        # Create sample data for the unified risk system
        unified_risk_system.create_sample_data(n_transactions=500, save_to_csv=True)
        
        return jsonify({
            'status': 'success',
            'message': 'Data loaded successfully',
            'data_summary': {
                'transactions': len(transactions_df) if transactions_df is not None else 0,
                'users': len(users_df) if users_df is not None else 0,
                'login_history': len(login_history_df) if login_history_df is not None else 0,
                'chargebacks': len(chargebacks_df) if chargebacks_df is not None else 0
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze_transaction', methods=['POST'])
def analyze_transaction():
    """
    Analyze a transaction for potential fraud.
    
    Expected JSON payload:
    {
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
        },
        "session_data": {
            "session_id": "S12345",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'transaction_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'transaction_data is required'
            }), 400
        
        transaction_data = data['transaction_data']
        user_data = data.get('user_data', None)
        session_data = data.get('session_data', None)
        
        # Analyze transaction using unified risk system
        results = unified_risk_system.analyze_transaction(transaction_data, user_data, session_data)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'transaction_id': transaction_data.get('transaction_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze_login', methods=['POST'])
def analyze_login():
    """
    Analyze a login attempt for potential account takeover.
    
    Expected JSON payload:
    {
        "login_data": {
            "user_id": "U12345",
            "timestamp": "2025-04-01T14:30:00",
            "ip_address": "192.168.1.1",
            "device_id": "D12345",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "login_success": true
        },
        "user_data": {
            "user_id": "U12345",
            "account_age_days": 120,
            "previous_purchases": 5
        },
        "session_data": {
            "session_id": "S12345",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'login_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'login_data is required'
            }), 400
        
        login_data = data['login_data']
        user_data = data.get('user_data', None)
        session_data = data.get('session_data', None)
        
        # Analyze login attempt using unified risk system
        results = unified_risk_system.analyze_login_attempt(login_data, user_data, session_data)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'user_id': login_data.get('user_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze_refund', methods=['POST'])
def analyze_refund():
    """
    Analyze a refund request for potential fraud.
    
    Expected JSON payload:
    {
        "refund_data": {
            "refund_id": "R12345",
            "transaction_id": "T123456",
            "user_id": "U12345",
            "timestamp": "2025-04-05T10:15:00",
            "amount": 299.99,
            "reason": "product_not_as_described"
        },
        "order_data": {
            "transaction_id": "T123456",
            "user_id": "U12345",
            "timestamp": "2025-04-01T14:30:00",
            "amount": 299.99,
            "delivery_status": "delivered",
            "delivery_date": "2025-04-03T12:00:00"
        },
        "user_data": {
            "user_id": "U12345",
            "account_age_days": 120,
            "previous_purchases": 5
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'refund_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'refund_data is required'
            }), 400
        
        refund_data = data['refund_data']
        order_data = data.get('order_data', None)
        user_data = data.get('user_data', None)
        
        # Analyze refund request using unified risk system
        results = unified_risk_system.analyze_refund_request(refund_data, order_data, user_data)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'refund_id': refund_data.get('refund_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/credit_card_fraud', methods=['POST'])
def analyze_credit_card():
    """
    Analyze a transaction specifically for credit card fraud.
    
    Expected JSON payload:
    {
        "transaction_data": {
            "transaction_id": "T123456",
            "user_id": "U12345",
            "timestamp": "2025-04-01T14:30:00",
            "amount": 299.99,
            "payment_method": "credit_card",
            "card_number_hash": "1a2b3c4d5e6f7g8h",
            "billing_country": "US",
            "shipping_country": "US"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'transaction_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'transaction_data is required'
            }), 400
        
        transaction_data = data['transaction_data']
        
        # Get user's transaction history
        user_id = transaction_data.get('user_id', None)
        user_transactions = pd.DataFrame()
        
        if user_id is not None and transactions_df is not None:
            user_transactions = transactions_df[transactions_df['user_id'] == user_id]
        
        # Analyze transaction for credit card fraud
        results = credit_card_detector.analyze_transaction(transaction_data, user_transactions)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'transaction_id': transaction_data.get('transaction_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/account_takeover', methods=['POST'])
def analyze_account_takeover():
    """
    Analyze a login attempt specifically for account takeover.
    
    Expected JSON payload:
    {
        "login_data": {
            "user_id": "U12345",
            "timestamp": "2025-04-01T14:30:00",
            "ip_address": "192.168.1.1",
            "device_id": "D12345",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "login_success": true
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'login_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'login_data is required'
            }), 400
        
        login_data = data['login_data']
        
        # Get user's login history
        user_id = login_data.get('user_id', None)
        login_history = pd.DataFrame()
        
        if user_id is not None and login_history_df is not None:
            login_history = login_history_df[login_history_df['user_id'] == user_id]
        
        # Build user profile if not already available
        user_profile = account_takeover_detector.user_profiles.get(user_id, None)
        if user_profile is None and not login_history.empty:
            user_profile = account_takeover_detector.build_user_profile(user_id, login_history)
        
        # Analyze login attempt for account takeover
        results = account_takeover_detector.analyze_login_attempt(login_data, user_profile, login_history)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'user_id': login_data.get('user_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/friendly_fraud', methods=['POST'])
def analyze_friendly_fraud():
    """
    Analyze a transaction for potential friendly fraud (chargeback fraud).
    
    Expected JSON payload:
    {
        "transaction_data": {
            "transaction_id": "T123456",
            "user_id": "U12345",
            "timestamp": "2025-04-01T14:30:00",
            "amount": 299.99,
            "payment_method": "credit_card",
            "product_category": "electronics"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'transaction_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'transaction_data is required'
            }), 400
        
        transaction_data = data['transaction_data']
        
        # Get user's transaction and chargeback history
        user_id = transaction_data.get('user_id', None)
        user_transactions = pd.DataFrame()
        user_chargebacks = pd.DataFrame()
        
        if user_id is not None:
            if transactions_df is not None:
                user_transactions = transactions_df[transactions_df['user_id'] == user_id]
            
            if chargebacks_df is not None:
                user_chargebacks = chargebacks_df[chargebacks_df['user_id'] == user_id]
        
        # Build customer profile
        customer_profile = friendly_fraud_detector.customer_profiles.get(user_id, None)
        if customer_profile is None and not user_transactions.empty:
            customer_profile = friendly_fraud_detector.build_customer_profile(user_id, user_transactions, user_chargebacks)
        
        # Extract features for chargeback prediction
        chargeback_features = friendly_fraud_detector.extract_chargeback_features(transaction_data, customer_profile)
        
        # Predict chargeback risk
        chargeback_risk = friendly_fraud_detector.predict_chargeback_risk(chargeback_features)
        
        results = {
            'chargeback_risk': chargeback_risk,
            'customer_profile': customer_profile
        }
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'transaction_id': transaction_data.get('transaction_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/promotion_abuse', methods=['POST'])
def analyze_promotion_abuse():
    """
    Analyze a promotion usage for potential abuse.
    
    Expected JSON payload:
    {
        "promotion_data": {
            "promotion_id": "P12345",
            "promotion_code": "SUMMER20",
            "user_id": "U12345",
            "order_id": "O12345",
            "timestamp": "2025-04-01T14:30:00",
            "amount": 299.99
        },
        "user_data": {
            "user_id": "U12345",
            "account_age_days": 120,
            "previous_purchases": 5
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'promotion_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'promotion_data is required'
            }), 400
        
        promotion_data = data['promotion_data']
        user_data = data.get('user_data', None)
        transaction_data = data.get('transaction_data', None)
        
        # Analyze promotion usage for abuse
        results = additional_fraud_detector.detect_promotion_abuse(promotion_data, user_data, transaction_data)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'promotion_id': promotion_data.get('promotion_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/bot_detection', methods=['POST'])
def analyze_bot_activity():
    """
    Analyze a request for potential bot activity.
    
    Expected JSON payload:
    {
        "request_data": {
            "request_id": "R12345",
            "session_id": "S12345",
            "ip_address": "192.168.1.1",
            "timestamp": "2025-04-01T14:30:00",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        "session_data": {
            "session_id": "S12345",
            "user_id": "U12345",
            "start_time": "2025-04-01T14:25:00",
            "page_views": 5,
            "actions": 3
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if 'request_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'request_data is required'
            }), 400
        
        request_data = data['request_data']
        session_data = data.get('session_data', None)
        
        # Analyze request for bot activity
        results = additional_fraud_detector.detect_bot_activity(request_data, session_data)
        
        # Convert any non-serializable objects to strings
        serializable_results = json_serialize(results)
        
        return jsonify({
            'status': 'success',
            'request_id': request_data.get('request_id', None),
            'analysis_results': serializable_results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/risk_report', methods=['GET'])
def generate_risk_report():
    """
    Generate a risk report for a specified time period.
    
    Query parameters:
    - start_date: Start date for the report (format: YYYY-MM-DD)
    - end_date: End date for the report (format: YYYY-MM-DD)
    """
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date', None)
        end_date_str = request.args.get('end_date', None)
        
        # Convert date strings to datetime objects
        start_date = None
        end_date = None
        
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            # Set end_date to end of day
            end_date = end_date.replace(hour=23, minute=59, second=59)
        
        # Generate risk report
        report = unified_risk_system.generate_risk_report(start_date, end_date)
        
        # Convert any non-serializable objects to strings
        serializable_report = json_serialize(report)
        
        return jsonify({
            'status': 'success',
            'start_date': start_date_str,
            'end_date': end_date_str,
            'report': serializable_report
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/user_risk_profile', methods=['GET'])
def get_user_risk_profile():
    """
    Get a user's risk profile.
    
    Query parameters:
    - user_id: User identifier
    """
    try:
        # Get query parameters
        user_id = request.args.get('user_id', None)
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400
        
        # Get user's risk profile
        risk_profile = unified_risk_system.get_user_risk_profile(user_id)
        
        if risk_profile is None:
            return jsonify({
                'status': 'error',
                'message': f'No risk profile found for user {user_id}'
            }), 404
        
        # Convert any non-serializable objects to strings
        serializable_profile = json_serialize(risk_profile)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'risk_profile': serializable_profile
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/risk_thresholds', methods=['GET', 'POST'])
def manage_risk_thresholds():
    """
    Get or update risk thresholds.
    
    For POST requests, expected JSON payload:
    {
        "high_risk": 0.8,
        "medium_risk": 0.5,
        "low_risk": 0.3
    }
    """
    try:
        if request.method == 'GET':
            # Get current risk thresholds
            return jsonify({
                'status': 'success',
                'risk_thresholds': unified_risk_system.risk_thresholds
            })
        
        elif request.method == 'POST':
            # Update risk thresholds
            data = request.get_json()
            
            # Validate thresholds
            if 'high_risk' in data and (data['high_risk'] < 0 or data['high_risk'] > 1):
                return jsonify({
                    'status': 'error',
                    'message': 'high_risk threshold must be between 0 and 1'
                }), 400
            
            if 'medium_risk' in data and (data['medium_risk'] < 0 or data['medium_risk'] > 1):
                return jsonify({
                    'status': 'error',
                    'message': 'medium_risk threshold must be between 0 and 1'
                }), 400
            
            if 'low_risk' in data and (data['low_risk'] < 0 or data['low_risk'] > 1):
                return jsonify({
                    'status': 'error',
                    'message': 'low_risk threshold must be between 0 and 1'
                }), 400
            
            # Update thresholds
            unified_risk_system.set_risk_thresholds(data)
            
            return jsonify({
                'status': 'success',
                'message': 'Risk thresholds updated successfully',
                'risk_thresholds': unified_risk_system.risk_thresholds
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/risk_weights', methods=['GET', 'POST'])
def manage_risk_weights():
    """
    Get or update risk weights.
    
    For POST requests, expected JSON payload:
    {
        "credit_card_fraud": 0.3,
        "account_takeover": 0.25,
        "friendly_fraud": 0.2,
        "promotion_abuse": 0.1,
        "refund_fraud": 0.1,
        "bot_activity": 0.05
    }
    """
    try:
        if request.method == 'GET':
            # Get current risk weights
            return jsonify({
                'status': 'success',
                'risk_weights': unified_risk_system.risk_weights
            })
        
        elif request.method == 'POST':
            # Update risk weights
            data = request.get_json()
            
            # Validate weights
            for key, value in data.items():
                if value < 0:
                    return jsonify({
                        'status': 'error',
                        'message': f'Weight for {key} must be non-negative'
                    }), 400
            
            # Update weights
            unified_risk_system.set_risk_weights(data)
            
            return jsonify({
                'status': 'success',
                'message': 'Risk weights updated successfully',
                'risk_weights': unified_risk_system.risk_weights
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

def json_serialize(obj):
    """
    Convert non-serializable objects to serializable format.
    
    Parameters:
    -----------
    obj : object
        Object to serialize
        
    Returns:
    --------
    object
        Serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return json_serialize(obj.to_dict())
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
