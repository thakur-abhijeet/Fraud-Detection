import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_ip():
    """Generate a random IP address"""
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

def generate_device_id():
    """Generate a random device ID"""
    device_types = ['iPhone', 'Android', 'Windows', 'Mac', 'Linux']
    return f"{random.choice(device_types)}_{random.randint(1000, 9999)}"

def generate_address():
    """Generate a random address"""
    streets = ['Main St', 'Park Ave', 'Broadway', 'Market St', 'State St']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    return f"{random.randint(1, 999)} {random.choice(streets)}, {random.choice(cities)}"

def generate_transaction_data(n_samples=1000, fraud_rate=0.05):
    """Generate sample transaction data with realistic patterns"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate base data
    data = {
        'transaction_id': range(n_samples),
        'timestamp': [
            (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
            for i in range(n_samples)
        ],
        'customer_id': np.random.randint(1, 100, n_samples),
        'product_id': np.random.randint(1, 50, n_samples),
        'payment_method': np.random.choice(
            ['credit_card', 'debit_card', 'paypal'],
            n_samples,
            p=[0.6, 0.3, 0.1]
        ),
        'shipping_address': [generate_address() for _ in range(n_samples)],
        'ip_address': [generate_ip() for _ in range(n_samples)],
        'device_id': [generate_device_id() for _ in range(n_samples)]
    }
    
    # Generate amounts with different distributions for fraud and non-fraud
    fraud_mask = np.random.binomial(1, fraud_rate, n_samples)
    
    # Normal transactions: mostly small amounts
    normal_amounts = np.random.lognormal(mean=4, sigma=0.5, size=n_samples)
    normal_amounts = np.clip(normal_amounts, 10, 1000)
    
    # Fraudulent transactions: mix of very small and very large amounts
    fraud_amounts = np.concatenate([
        np.random.lognormal(mean=2, sigma=0.3, size=int(n_samples * fraud_rate * 0.7)),  # Small amounts
        np.random.lognormal(mean=6, sigma=0.5, size=int(n_samples * fraud_rate * 0.3))   # Large amounts
    ])
    fraud_amounts = np.clip(fraud_amounts, 10, 5000)
    
    # Combine amounts based on fraud mask
    data['amount'] = np.where(fraud_mask, fraud_amounts, normal_amounts)
    
    # Add some patterns to make fraud detection more interesting
    # 1. Multiple transactions from same IP in short time
    for i in range(1, n_samples):
        if random.random() < 0.1:  # 10% chance to copy previous IP
            data['ip_address'][i] = data['ip_address'][i-1]
    
    # 2. Unusual payment methods for certain amounts
    for i in range(n_samples):
        if data['amount'][i] > 1000 and data['payment_method'][i] == 'paypal':
            data['payment_method'][i] = 'credit_card'
    
    # 3. Add some missing values
    for col in ['shipping_address', 'ip_address', 'device_id']:
        mask = np.random.random(n_samples) < 0.05  # 5% missing values
        data[col] = np.where(mask, np.nan, data[col])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add is_fraud column
    df['is_fraud'] = fraud_mask
    
    return df

def main():
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Generate sample data
    df = generate_transaction_data(n_samples=1000, fraud_rate=0.05)
    
    # Save to CSV
    output_path = '../data/sample_transactions.csv'
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"Generated sample data saved to {output_path}")
    print("\nData Summary:")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print("\nAmount Statistics:")
    print(df['amount'].describe())
    print("\nPayment Method Distribution:")
    print(df['payment_method'].value_counts(normalize=True))

if __name__ == "__main__":
    main() 