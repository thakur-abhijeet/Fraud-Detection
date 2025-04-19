"""
Test script for the Fraud Detection System API

This script tests the main endpoints of the fraud detection system API
to ensure they are working correctly.
"""

import requests
import json
import time
import sys

# Base URL for the API
BASE_URL = "http://localhost:5001"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/api/health")
    
    if response.status_code == 200:
        print("✅ Health check successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Health check failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_load_data():
    """Test the load data endpoint"""
    print("\n=== Testing Load Data ===")
    response = requests.post(
        f"{BASE_URL}/api/load_data",
        json={}  # Empty JSON to use sample data
    )
    
    if response.status_code == 200:
        print("✅ Load data successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Load data failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_analyze_transaction():
    """Test the analyze transaction endpoint"""
    print("\n=== Testing Transaction Analysis ===")
    response = requests.post(
        f"{BASE_URL}/api/analyze_transaction",
        json={
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
    )
    
    if response.status_code == 200:
        print("✅ Transaction analysis successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Transaction analysis failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_analyze_login():
    """Test the analyze login endpoint"""
    print("\n=== Testing Login Analysis ===")
    response = requests.post(
        f"{BASE_URL}/api/analyze_login",
        json={
            "login_data": {
                "user_id": "U12345",
                "timestamp": "2025-04-01T14:30:00",
                "ip_address": "192.168.1.1",
                "device_id": "D12345",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "login_success": True
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
    )
    
    if response.status_code == 200:
        print("✅ Login analysis successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Login analysis failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_analyze_refund():
    """Test the analyze refund endpoint"""
    print("\n=== Testing Refund Analysis ===")
    response = requests.post(
        f"{BASE_URL}/api/analyze_refund",
        json={
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
    )
    
    if response.status_code == 200:
        print("✅ Refund analysis successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Refund analysis failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_credit_card_fraud():
    """Test the credit card fraud endpoint"""
    print("\n=== Testing Credit Card Fraud Detection ===")
    response = requests.post(
        f"{BASE_URL}/api/credit_card_fraud",
        json={
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
    )
    
    if response.status_code == 200:
        print("✅ Credit card fraud detection successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Credit card fraud detection failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_account_takeover():
    """Test the account takeover endpoint"""
    print("\n=== Testing Account Takeover Detection ===")
    response = requests.post(
        f"{BASE_URL}/api/account_takeover",
        json={
            "login_data": {
                "user_id": "U12345",
                "timestamp": "2025-04-01T14:30:00",
                "ip_address": "192.168.1.1",
                "device_id": "D12345",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "login_success": True
            }
        }
    )
    
    if response.status_code == 200:
        print("✅ Account takeover detection successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Account takeover detection failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_friendly_fraud():
    """Test the friendly fraud endpoint"""
    print("\n=== Testing Friendly Fraud Detection ===")
    response = requests.post(
        f"{BASE_URL}/api/friendly_fraud",
        json={
            "transaction_data": {
                "transaction_id": "T123456",
                "user_id": "U12345",
                "timestamp": "2025-04-01T14:30:00",
                "amount": 299.99,
                "payment_method": "credit_card",
                "product_category": "electronics"
            }
        }
    )
    
    if response.status_code == 200:
        print("✅ Friendly fraud detection successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Friendly fraud detection failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_promotion_abuse():
    """Test the promotion abuse endpoint"""
    print("\n=== Testing Promotion Abuse Detection ===")
    response = requests.post(
        f"{BASE_URL}/api/promotion_abuse",
        json={
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
    )
    
    if response.status_code == 200:
        print("✅ Promotion abuse detection successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Promotion abuse detection failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_bot_detection():
    """Test the bot detection endpoint"""
    print("\n=== Testing Bot Detection ===")
    response = requests.post(
        f"{BASE_URL}/api/bot_detection",
        json={
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
    )
    
    if response.status_code == 200:
        print("✅ Bot detection successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Bot detection failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_risk_report():
    """Test the risk report endpoint"""
    print("\n=== Testing Risk Report Generation ===")
    response = requests.get(
        f"{BASE_URL}/api/risk_report",
        params={
            "start_date": "2025-03-01",
            "end_date": "2025-04-01"
        }
    )
    
    if response.status_code == 200:
        print("✅ Risk report generation successful")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Risk report generation failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_risk_thresholds():
    """Test the risk thresholds endpoint"""
    print("\n=== Testing Risk Thresholds ===")
    
    # Get current thresholds
    get_response = requests.get(f"{BASE_URL}/api/risk_thresholds")
    
    if get_response.status_code == 200:
        print("✅ Get risk thresholds successful")
        print(f"Response: {json.dumps(get_response.json(), indent=2)}")
        
        # Update thresholds
        post_response = requests.post(
            f"{BASE_URL}/api/risk_thresholds",
            json={
                "high_risk": 0.8,
                "medium_risk": 0.5,
                "low_risk": 0.3
            }
        )
        
        if post_response.status_code == 200:
            print("✅ Update risk thresholds successful")
            print(f"Response: {json.dumps(post_response.json(), indent=2)}")
            return True
        else:
            print(f"❌ Update risk thresholds failed with status code {post_response.status_code}")
            print(f"Response: {post_response.text}")
            return False
    else:
        print(f"❌ Get risk thresholds failed with status code {get_response.status_code}")
        print(f"Response: {get_response.text}")
        return False

def run_all_tests():
    """Run all tests and return the number of failures"""
    print("Starting Fraud Detection System API Tests...")
    
    # Wait for the API to start
    print("Waiting for API to start...")
    time.sleep(2)
    
    # Run tests
    tests = [
        test_health_check,
        test_load_data,
        test_analyze_transaction,
        test_analyze_login,
        test_analyze_refund,
        test_credit_card_fraud,
        test_account_takeover,
        test_friendly_fraud,
        test_promotion_abuse,
        test_bot_detection,
        test_risk_report,
        test_risk_thresholds
    ]
    
    failures = 0
    for test in tests:
        if not test():
            failures += 1
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {len(tests) - failures}")
    print(f"Failed: {failures}")
    
    return failures

if __name__ == "__main__":
    failures = run_all_tests()
    sys.exit(failures)  # Return the number of failures as exit code
