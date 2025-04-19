# Fraud Detection System API Documentation

This document provides detailed information about the API endpoints available for the e-commerce fraud detection system.

## Base URL

When running locally, the API is available at:

```
http://localhost:5001
```

## API Endpoints

### Health Check

Check if the API is running.

**Endpoint:** `/api/health`  
**Method:** GET  
**Response:**
```json
{
  "status": "healthy",
  "message": "Fraud Detection System API is running"
}
```

### Load Data

Load data from specified file paths or use sample data.

**Endpoint:** `/api/load_data`  
**Method:** POST  
**Request Body:**
```json
{
  "transactions_file": "/path/to/transactions.csv",
  "users_file": "/path/to/users.csv",
  "login_history_file": "/path/to/login_history.csv",
  "chargebacks_file": "/path/to/chargebacks.csv"
}
```
**Note:** All file paths are optional. If not provided, the system will use sample data.

**Response:**
```json
{
  "status": "success",
  "message": "Data loaded successfully",
  "data_summary": {
    "transactions": 1000,
    "users": 500,
    "login_history": 2000,
    "chargebacks": 100
  }
}
```

### Analyze Transaction

Analyze a transaction for potential fraud.

**Endpoint:** `/api/analyze_transaction`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "transaction_id": "T123456",
  "analysis_results": {
    "transaction_id": "T123456",
    "user_id": "U12345",
    "timestamp": "2025-04-01T14:30:00",
    "amount": 299.99,
    "fraud_checks": {
      "credit_card_fraud": {
        "velocity_checks": {
          "amount_1h": {
            "value": 299.99,
            "threshold": 1000,
            "exceeded": false
          },
          "count_1h": {
            "value": 1,
            "threshold": 3,
            "exceeded": false
          }
        },
        "amount_anomaly": false,
        "overall_risk_score": 0.15
      },
      "account_takeover": {
        "device_anomaly": false,
        "location_anomaly": false,
        "overall_risk_score": 0.1
      }
    },
    "risk_scores": {
      "credit_card_fraud": 0.15,
      "account_takeover": 0.1
    },
    "overall_risk_score": 0.13,
    "risk_level": "low",
    "recommended_action": "allow"
  }
}
```

### Analyze Login

Analyze a login attempt for potential account takeover.

**Endpoint:** `/api/analyze_login`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "user_id": "U12345",
  "analysis_results": {
    "device_check": {
      "device_known": true,
      "risk_score": 0.1
    },
    "location_check": {
      "location_anomaly": false,
      "distance_from_known": 5.2,
      "risk_score": 0.1
    },
    "time_check": {
      "time_anomaly": false,
      "risk_score": 0.1
    },
    "velocity_check": {
      "login_attempts_1h": 1,
      "threshold": 5,
      "exceeded": false,
      "risk_score": 0.1
    },
    "overall_risk_score": 0.1,
    "risk_level": "low",
    "recommended_action": "allow"
  }
}
```

### Analyze Refund

Analyze a refund request for potential fraud.

**Endpoint:** `/api/analyze_refund`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "refund_id": "R12345",
  "analysis_results": {
    "refund_frequency_check": {
      "refund_count_30d": 1,
      "threshold": 3,
      "exceeded": false,
      "risk_score": 0.1
    },
    "time_to_refund_check": {
      "days_since_delivery": 2,
      "threshold": 30,
      "risk_score": 0.2
    },
    "reason_check": {
      "reason": "product_not_as_described",
      "risk_score": 0.3
    },
    "overall_risk_score": 0.2,
    "risk_level": "low",
    "recommended_action": "approve_refund"
  }
}
```

### Credit Card Fraud Detection

Analyze a transaction specifically for credit card fraud.

**Endpoint:** `/api/credit_card_fraud`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "transaction_id": "T123456",
  "analysis_results": {
    "velocity_checks": {
      "amount_1h": {
        "value": 299.99,
        "threshold": 1000,
        "exceeded": false
      },
      "count_1h": {
        "value": 1,
        "threshold": 3,
        "exceeded": false
      },
      "countries_1h": {
        "value": 1,
        "threshold": 2,
        "exceeded": false
      }
    },
    "bin_country_check": {
      "match": true,
      "risk_score": 0.1
    },
    "address_verification": {
      "billing_shipping_match": true,
      "risk_score": 0.1
    },
    "amount_anomaly": {
      "is_anomalous": false,
      "z_score": 0.5,
      "risk_score": 0.1
    },
    "ml_prediction": {
      "fraud_probability": 0.05,
      "risk_score": 0.05
    },
    "overall_risk_score": 0.1,
    "risk_level": "low",
    "recommended_action": "allow"
  }
}
```

### Account Takeover Detection

Analyze a login attempt specifically for account takeover.

**Endpoint:** `/api/account_takeover`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "user_id": "U12345",
  "analysis_results": {
    "device_check": {
      "device_known": true,
      "risk_score": 0.1
    },
    "location_check": {
      "location_anomaly": false,
      "distance_from_known": 5.2,
      "risk_score": 0.1
    },
    "time_check": {
      "time_anomaly": false,
      "risk_score": 0.1
    },
    "velocity_check": {
      "login_attempts_1h": 1,
      "threshold": 5,
      "exceeded": false,
      "risk_score": 0.1
    },
    "password_change_check": {
      "recent_password_change": false,
      "risk_score": 0.1
    },
    "overall_risk_score": 0.1,
    "risk_level": "low",
    "recommended_action": "allow"
  }
}
```

### Friendly Fraud Detection

Analyze a transaction for potential friendly fraud (chargeback fraud).

**Endpoint:** `/api/friendly_fraud`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "transaction_id": "T123456",
  "analysis_results": {
    "chargeback_risk": 0.15,
    "customer_profile": {
      "user_id": "U12345",
      "transaction_count": 10,
      "chargeback_count": 0,
      "average_order_value": 250.0,
      "account_age_days": 120
    }
  }
}
```

### Promotion Abuse Detection

Analyze a promotion usage for potential abuse.

**Endpoint:** `/api/promotion_abuse`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "promotion_id": "P12345",
  "analysis_results": {
    "usage_frequency_check": {
      "usage_count": 1,
      "threshold": 3,
      "exceeded": false,
      "risk_score": 0.1
    },
    "multi_account_check": {
      "related_accounts": 0,
      "threshold": 2,
      "exceeded": false,
      "risk_score": 0.1
    },
    "order_manipulation_check": {
      "manipulation_detected": false,
      "risk_score": 0.1
    },
    "overall_risk_score": 0.1,
    "risk_level": "low",
    "recommended_action": "allow"
  }
}
```

### Bot Detection

Analyze a request for potential bot activity.

**Endpoint:** `/api/bot_detection`  
**Method:** POST  
**Request Body:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "request_id": "R12345",
  "analysis_results": {
    "user_agent_check": {
      "is_bot": false,
      "bot_indicators": [],
      "risk_score": 0.1
    },
    "request_rate_check": {
      "requests_per_minute": 1.0,
      "threshold": 10.0,
      "exceeded": false,
      "risk_score": 0.1
    },
    "session_behavior_check": {
      "behavior_anomaly": false,
      "risk_score": 0.1
    },
    "overall_risk_score": 0.1,
    "risk_level": "low",
    "recommended_action": "allow"
  }
}
```

### Risk Report

Generate a risk report for a specified time period.

**Endpoint:** `/api/risk_report`  
**Method:** GET  
**Query Parameters:**
- `start_date` (optional): Start date for the report (format: YYYY-MM-DD)
- `end_date` (optional): End date for the report (format: YYYY-MM-DD)

**Response:**
```json
{
  "status": "success",
  "start_date": "2025-03-01",
  "end_date": "2025-04-01",
  "report": {
    "total_transactions": 1000,
    "risk_levels": {
      "counts": {
        "high": 50,
        "medium": 150,
        "low": 300,
        "very_low": 500
      },
      "percentages": {
        "high": 5.0,
        "medium": 15.0,
        "low": 30.0,
        "very_low": 50.0
      }
    },
    "fraud_types": {
      "credit_card_fraud": 30,
      "account_takeover": 20,
      "friendly_fraud": 15,
      "promotion_abuse": 10,
      "refund_fraud": 5,
      "bot_activity": 2
    },
    "high_risk_users": [
      {
        "user_id": "U12345",
        "high_risk_transactions": 3,
        "risk_profile": {
          "user_id": "U12345",
          "transaction_count": 10,
          "high_risk_count": 3,
          "medium_risk_count": 2,
          "average_risk_score": 0.6,
          "risk_factors": ["credit_card_fraud", "friendly_fraud"]
        }
      }
    ]
  }
}
```

### User Risk Profile

Get a user's risk profile.

**Endpoint:** `/api/user_risk_profile`  
**Method:** GET  
**Query Parameters:**
- `user_id`: User identifier

**Response:**
```json
{
  "status": "success",
  "user_id": "U12345",
  "risk_profile": {
    "user_id": "U12345",
    "transaction_count": 10,
    "high_risk_count": 1,
    "medium_risk_count": 2,
    "average_risk_score": 0.3,
    "last_transaction_timestamp": "2025-04-01T14:30:00",
    "risk_factors": ["credit_card_fraud", "friendly_fraud"]
  }
}
```

### Risk Thresholds

Get or update risk thresholds.

**Endpoint:** `/api/risk_thresholds`  
**Method:** GET  
**Response:**
```json
{
  "status": "success",
  "risk_thresholds": {
    "high_risk": 0.7,
    "medium_risk": 0.4,
    "low_risk": 0.2
  }
}
```

**Endpoint:** `/api/risk_thresholds`  
**Method:** POST  
**Request Body:**
```json
{
  "high_risk": 0.8,
  "medium_risk": 0.5,
  "low_risk": 0.3
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Risk thresholds updated successfully",
  "risk_thresholds": {
    "high_risk": 0.8,
    "medium_risk": 0.5,
    "low_risk": 0.3
  }
}
```

### Risk Weights

Get or update risk weights.

**Endpoint:** `/api/risk_weights`  
**Method:** GET  
**Response:**
```json
{
  "status": "success",
  "risk_weights": {
    "credit_card_fraud": 0.3,
    "account_takeover": 0.25,
    "friendly_fraud": 0.2,
    "promotion_abuse": 0.1,
    "refund_fraud": 0.1,
    "bot_activity": 0.05
  }
}
```

**Endpoint:** `/api/risk_weights`  
**Method:** POST  
**Request Body:**
```json
{
  "credit_card_fraud": 0.4,
  "account_takeover": 0.3,
  "friendly_fraud": 0.15,
  "promotion_abuse": 0.05,
  "refund_fraud": 0.05,
  "bot_activity": 0.05
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Risk weights updated successfully",
  "risk_weights": {
    "credit_card_fraud": 0.4,
    "account_takeover": 0.3,
    "friendly_fraud": 0.15,
    "promotion_abuse": 0.05,
    "refund_fraud": 0.05,
    "bot_activity": 0.05
  }
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages in case of failure.

**Example Error Response:**
```json
{
  "status": "error",
  "message": "Data not loaded. Call /api/load_data first.",
  "traceback": "..."
}
```

## Usage Examples

### Python Example

```python
import requests
import json

# Base URL
base_url = "http://localhost:5001"

# Load data
response = requests.post(
    f"{base_url}/api/load_data",
    json={
        "transactions_file": "/path/to/transactions.csv",
        "users_file": "/path/to/users.csv"
    }
)
print(json.dumps(response.json(), indent=2))

# Analyze a transaction
response = requests.post(
    f"{base_url}/api/analyze_transaction",
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
        }
    }
)
print(json.dumps(response.json(), indent=2))

# Get a risk report
response = requests.get(
    f"{base_url}/api/risk_report",
    params={
        "start_date": "2025-03-01",
        "end_date": "2025-04-01"
    }
)
print(json.dumps(response.json(), indent=2))
```

### cURL Example

```bash
# Health check
curl -X GET http://localhost:5001/api/health

# Load data
curl -X POST http://localhost:5001/api/load_data \
  -H "Content-Type: application/json" \
  -d '{"transactions_file": "/path/to/transactions.csv", "users_file": "/path/to/users.csv"}'

# Analyze a transaction
curl -X POST http://localhost:5001/api/analyze_transaction \
  -H "Content-Type: application/json" \
  -d '{
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
    }
  }'

# Get a risk report
curl -X GET "http://localhost:5001/api/risk_report?start_date=2025-03-01&end_date=2025-04-01"
```

## Running the API

To run the API locally:

1. Navigate to the fraud detection system directory
2. Run the following command:

```bash
python api/app.py
```

This will start the API server on port 5001.
