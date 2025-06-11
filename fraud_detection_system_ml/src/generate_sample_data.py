import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import argparse
from utils import setup_logging, load_config
import logging

def generate_ip():
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

def generate_device_id():
    device_types = ['iPhone', 'Android', 'Windows', 'Mac', 'Linux']
    return f"{random.choice(device_types)}_{random.randint(1000, 9999)}"

def generate_address():
    streets = ['Main St', 'Park Ave', 'Broadway', 'Market St', 'State St']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    return f"{random.randint(1, 999)} {random.choice(streets)}, {random.choice(cities)}"

def generate_transaction_data(n_samples=1000, fraud_rate=0.05):
    np.random.seed(42)
    random.seed(42)
    n_fraud = int(n_samples * fraud_rate)
    now = datetime.now()
    data = {
        'transaction_id': range(n_samples),
        'timestamp': [
            (now - timedelta(minutes=i*random.randint(1,3))).strftime('%Y-%m-%d %H:%M:%S')
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

    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    fraud_mask = np.zeros(n_samples, dtype=bool)
    fraud_mask[fraud_indices] = True

    normal_amounts = np.random.lognormal(mean=4, sigma=0.5, size=n_samples)
    normal_amounts = np.clip(normal_amounts, 10, 1000)
    n_small_fraud = int(n_fraud * 0.7)
    n_large_fraud = n_fraud - n_small_fraud
    small_fraud_amounts = np.random.lognormal(mean=2, sigma=0.3, size=n_small_fraud)
    large_fraud_amounts = np.random.lognormal(mean=6, sigma=0.5, size=n_large_fraud)
    fraud_amounts = np.concatenate([small_fraud_amounts, large_fraud_amounts])
    fraud_amounts = np.clip(fraud_amounts, 10, 5000)
    amounts = np.zeros(n_samples)
    amounts[fraud_indices] = fraud_amounts
    amounts[~fraud_mask] = normal_amounts[~fraud_mask]
    data['amount'] = amounts

    # Add patterns for fraud detection
    for i in range(1, n_samples):
        if random.random() < 0.1:
            data['ip_address'][i] = data['ip_address'][i-1]
    for i in range(n_samples):
        if data['amount'][i] > 1000 and data['payment_method'][i] == 'paypal':
            data['payment_method'][i] = 'credit_card'

    df = pd.DataFrame(data)
    for col in ['shipping_address', 'ip_address', 'device_id']:
        mask = np.random.random(n_samples) < 0.05
        df[col] = df[col].mask(mask)
    df['is_fraud'] = fraud_mask

    # ---- ENHANCED FEATURE ENGINEERING ----
    df = df.sort_values(by="timestamp")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['prev_timestamp'] = df.groupby('customer_id')['timestamp'].shift(1)
    df['time_since_last_transaction'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds().fillna(-1)
    df['transactions_past_1hr'] = df.groupby('customer_id')['timestamp'].rolling('1h', on='timestamp').count().reset_index(level=0, drop=True) - 1
    df['transactions_past_1hr'] = df['transactions_past_1hr'].fillna(0)

    return df

def main():
    parser = argparse.ArgumentParser(description="Generate sample transaction data")
    parser.add_argument('--config', default='../config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(log_dir=config['output']['logs_dir'], log_name="generate_sample_data.log")

    os.makedirs(os.path.dirname(config['data']['raw_data_path']), exist_ok=True)
    df = generate_transaction_data(n_samples=1000, fraud_rate=0.05)
    df.to_csv(config['data']['raw_data_path'], index=False)
    logging.info(f"Generated sample data saved to {config['data']['raw_data_path']}")
    logging.info(f"Total transactions: {len(df)}")
    logging.info(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    logging.info(f"Amount statistics:\n{df['amount'].describe()}")
    logging.info(f"Payment method distribution:\n{df['payment_method'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()