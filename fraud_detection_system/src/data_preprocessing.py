# Data Preprocessing Module for E-commerce Fraud Detection System
# This module provides utilities for loading, cleaning, and preprocessing data
# for fraud detection in e-commerce transactions.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import datetime
import re
import os
import json
import pickle

class FraudDataPreprocessor:

    # A class for preprocessing e-commerce data for fraud detection.
    # This class handles data loading, cleaning, feature extraction, and splitting
    # for different fraud detection approaches.

    
    def __init__(self, data_dir='../data'):
  
        # Initialize the FraudDataPreprocessor.
        # Parameters:
        # -----------
        # data_dir : str
        #     Directory containing the data files
    
        self.data_dir = data_dir
        self.transactions_df = None
        self.users_df = None
        self.logins_df = None
        self.chargebacks_df = None
        self.devices_df = None
        
        # Scalers and imputers
        self.numeric_scaler = None
        self.numeric_imputer = None
        
        # Feature lists
        self.transaction_features = []
        self.user_features = []
        self.device_features = []
        self.behavioral_features = []
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self, transactions_file=None, users_file=None, 
                 logins_file=None, chargebacks_file=None, devices_file=None):

        # Load data from CSV or JSON files.
        
        # Parameters:
        # -----------
        # transactions_file : str
        #     Filename for transaction data
        # users_file : str
        #     Filename for user data
        # logins_file : str
        #     Filename for login activity data
        # chargebacks_file : str
        #     Filename for chargeback data
        # devices_file : str
        #     Filename for device data
            
        # Returns:
        # --------
        # self : FraudDataPreprocessor
        #     Returns self for method chaining

        # Load data files if provided
        if transactions_file:
            file_path = os.path.join(self.data_dir, transactions_file)
            if file_path.endswith('.csv'):
                self.transactions_df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.transactions_df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format for {transactions_file}")
                
        if users_file:
            file_path = os.path.join(self.data_dir, users_file)
            if file_path.endswith('.csv'):
                self.users_df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.users_df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format for {users_file}")
                
        if logins_file:
            file_path = os.path.join(self.data_dir, logins_file)
            if file_path.endswith('.csv'):
                self.logins_df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.logins_df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format for {logins_file}")
                
        if chargebacks_file:
            file_path = os.path.join(self.data_dir, chargebacks_file)
            if file_path.endswith('.csv'):
                self.chargebacks_df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.chargebacks_df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format for {chargebacks_file}")
                
        if devices_file:
            file_path = os.path.join(self.data_dir, devices_file)
            if file_path.endswith('.csv'):
                self.devices_df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.devices_df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format for {devices_file}")
        
        return self
    
    def clean_data(self, handle_missing=True, remove_duplicates=True, handle_outliers=True):
        """
        Clean the loaded data by handling missing values, removing duplicates, and handling outliers.
        
        Parameters:
        -----------
        handle_missing : bool
            Whether to handle missing values
        remove_duplicates : bool
            Whether to remove duplicate entries
        handle_outliers : bool
            Whether to handle outliers in numerical data
            
        Returns:
        --------
        self : FraudDataPreprocessor
            Returns self for method chaining
        """
        # Clean transactions data
        if self.transactions_df is not None:
            if handle_missing:
                # Initialize imputer for numeric columns
                num_cols = self.transactions_df.select_dtypes(include=['number']).columns
                self.numeric_imputer = SimpleImputer(strategy='median')
                
                if num_cols.any():
                    self.transactions_df[num_cols] = self.numeric_imputer.fit_transform(
                        self.transactions_df[num_cols]
                    )
                
                # Fill missing categorical values with mode or 'Unknown'
                cat_cols = self.transactions_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.transactions_df[col].fillna('Unknown', inplace=True)
            
            if remove_duplicates:
                # Assuming transaction_id is the unique identifier
                if 'transaction_id' in self.transactions_df.columns:
                    self.transactions_df.drop_duplicates(subset=['transaction_id'], keep='first', inplace=True)
            
            if handle_outliers:
                # Handle outliers in amount column if it exists
                if 'amount' in self.transactions_df.columns:
                    # Use Robust Scaler for amount to handle outliers
                    amount_scaler = RobustScaler()
                    self.transactions_df['amount_scaled'] = amount_scaler.fit_transform(
                        self.transactions_df[['amount']]
                    )
        
        # Clean users data
        if self.users_df is not None:
            if handle_missing:
                # Fill missing categorical values
                cat_cols = self.users_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.users_df[col].fillna('Unknown', inplace=True)
                
                # Fill missing numerical values
                num_cols = self.users_df.select_dtypes(include=['number']).columns
                if num_cols.any():
                    imputer = SimpleImputer(strategy='median')
                    self.users_df[num_cols] = imputer.fit_transform(self.users_df[num_cols])
            
            if remove_duplicates:
                # Assuming user_id is the unique identifier
                if 'user_id' in self.users_df.columns:
                    self.users_df.drop_duplicates(subset=['user_id'], keep='first', inplace=True)
        
        # Clean logins data
        if self.logins_df is not None:
            if handle_missing:
                # Fill missing categorical values
                cat_cols = self.logins_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.logins_df[col].fillna('Unknown', inplace=True)
            
            if remove_duplicates:
                # Remove duplicate login entries (same user-timestamp)
                id_cols = [col for col in ['user_id', 'timestamp'] if col in self.logins_df.columns]
                if id_cols:
                    self.logins_df.drop_duplicates(subset=id_cols, keep='first', inplace=True)
        
        # Clean chargebacks data
        if self.chargebacks_df is not None:
            if handle_missing:
                # Fill missing categorical values
                cat_cols = self.chargebacks_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.chargebacks_df[col].fillna('Unknown', inplace=True)
            
            if remove_duplicates:
                # Remove duplicate chargeback entries
                id_cols = [col for col in ['transaction_id', 'chargeback_id'] if col in self.chargebacks_df.columns]
                if id_cols:
                    self.chargebacks_df.drop_duplicates(subset=id_cols, keep='first', inplace=True)
        
        # Clean devices data
        if self.devices_df is not None:
            if handle_missing:
                # Fill missing categorical values
                cat_cols = self.devices_df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    self.devices_df[col].fillna('Unknown', inplace=True)
            
            if remove_duplicates:
                # Remove duplicate device entries
                id_cols = [col for col in ['device_id', 'user_id'] if col in self.devices_df.columns]
                if id_cols:
                    self.devices_df.drop_duplicates(subset=id_cols, keep='first', inplace=True)
        
        return self
    
    def extract_datetime_features(self, df, datetime_col='timestamp', prefix=''):
        """
        Extract features from datetime columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing datetime column
        datetime_col : str
            Name of the datetime column
        prefix : str
            Prefix to add to new feature columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted datetime features
        """
        if datetime_col not in df.columns:
            return df
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
            result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')
        
        # Extract datetime features
        result_df[f'{prefix}hour'] = result_df[datetime_col].dt.hour
        result_df[f'{prefix}day'] = result_df[datetime_col].dt.day
        result_df[f'{prefix}day_of_week'] = result_df[datetime_col].dt.dayofweek
        result_df[f'{prefix}month'] = result_df[datetime_col].dt.month
        result_df[f'{prefix}year'] = result_df[datetime_col].dt.year
        result_df[f'{prefix}is_weekend'] = result_df[datetime_col].dt.dayofweek >= 5
        
        # Extract time periods
        result_df[f'{prefix}time_period'] = pd.cut(
            result_df[datetime_col].dt.hour,
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        return result_df
    
    def extract_location_features(self, df, ip_col=None, country_col=None, city_col=None, zip_col=None):
        """
        Extract features from location data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing location columns
        ip_col : str
            Name of the IP address column
        country_col : str
            Name of the country column
        city_col : str
            Name of the city column
        zip_col : str
            Name of the ZIP/postal code column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted location features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Process IP address if available
        if ip_col and ip_col in result_df.columns:
            # In a real implementation, we would use IP geolocation libraries
            # For this example, we'll just create a dummy feature
            result_df['ip_risk_score'] = np.random.uniform(0, 1, size=len(result_df))
        
        # Process country if available
        if country_col and country_col in result_df.columns:
            # Create country risk score based on fraud statistics
            # In a real implementation, this would be based on actual data
            high_risk_countries = ['CountryA', 'CountryB', 'CountryC']
            medium_risk_countries = ['CountryD', 'CountryE', 'CountryF']
            
            result_df['country_risk'] = 'low'
            result_df.loc[result_df[country_col].isin(medium_risk_countries), 'country_risk'] = 'medium'
            result_df.loc[result_df[country_col].isin(high_risk_countries), 'country_risk'] = 'high'
            
            # One-hot encode country risk
            country_risk_dummies = pd.get_dummies(result_df['country_risk'], prefix='country_risk')
            result_df = pd.concat([result_df, country_risk_dummies], axis=1)
        
        # Process address mismatch if we have both billing and shipping info
        if 'billing_zip' in result_df.columns and 'shipping_zip' in result_df.columns:
            result_df['zip_mismatch'] = (result_df['billing_zip'] != result_df['shipping_zip']).astype(int)
        
        if 'billing_country' in result_df.columns and 'shipping_country' in result_df.columns:
            result_df['country_mismatch'] = (result_df['billing_country'] != result_df['shipping_country']).astype(int)
        
        return result_df
    
    def extract_payment_features(self, df):
        """
        Extract features from payment data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing payment information
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted payment features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract card BIN (first 6 digits) if available
        if 'card_number' in result_df.columns:
            # In a real implementation, we would properly handle card numbers
            # For this example, we'll just create a dummy feature
            result_df['card_bin'] = result_df['card_number'].astype(str).str[:6]
            
            # Remove full card number for security
            result_df.drop('card_number', axis=1, inplace=True)
        
        # Extract payment method risk
        if 'payment_method' in result_df.columns:
            # Create payment method risk score
            high_risk_methods = ['gift_card', 'prepaid_card', 'cryptocurrency']
            medium_risk_methods = ['credit_card', 'digital_wallet']
            low_risk_methods = ['bank_transfer', 'debit_card']
            
            result_df['payment_risk'] = 'medium'
            result_df.loc[result_df['payment_method'].isin(low_risk_methods), 'payment_risk'] = 'low'
            result_df.loc[result_df['payment_method'].isin(high_risk_methods), 'payment_risk'] = 'high'
            
            # One-hot encode payment risk
            payment_risk_dummies = pd.get_dummies(result_df['payment_risk'], prefix='payment_risk')
            result_df = pd.concat([result_df, payment_risk_dummies], axis=1)
        
        # Calculate transaction velocity features if timestamp is available
        if 'user_id' in result_df.columns and 'timestamp' in result_df.columns:
            # Convert timestamp to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], errors='coerce')
            
            # Sort by user_id and timestamp
            result_df = result_df.sort_values(['user_id', 'timestamp'])
            
            # Calculate time difference between consecutive transactions for the same user
            result_df['prev_timestamp'] = result_df.groupby('user_id')['timestamp'].shift(1)
            result_df['time_since_last_transaction'] = (result_df['timestamp'] - result_df['prev_timestamp']).dt.total_seconds() / 3600  # in hours
            
            # Fill NaN values (first transaction for each user)
            result_df['time_since_last_transaction'].fillna(1000, inplace=True)  # Large value to indicate no previous transaction
            
            # Create velocity features
            result_df['high_velocity'] = (result_df['time_since_last_transaction'] < 1).astype(int)  # Less than 1 hour
            result_df['medium_velocity'] = ((result_df['time_since_last_transaction'] >= 1) & 
                                          (result_df['time_since_last_transaction'] < 24)).astype(int)  # Between 1 and 24 hours
        
        return result_df
    
    def extract_user_behavior_features(self, transactions_df=None, logins_df=None):
        """
        Extract user behavior features from transaction and login data.
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            DataFrame containing transaction data
        logins_df : pd.DataFrame
            DataFrame containing login data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with user behavior features
        """
        if transactions_df is None:
            transactions_df = self.transactions_df
        
        if logins_df is None:
            logins_df = self.logins_df
        
        if transactions_df is None or 'user_id' not in transactions_df.columns:
            raise ValueError("Transaction data with user_id column is required")
        
        # Create a DataFrame with one row per user
        user_ids = transactions_df['user_id'].unique()
        user_behavior_df = pd.DataFrame({'user_id': user_ids})
        
        # Extract transaction behavior features
        if 'timestamp' in transactions_df.columns:
            # Convert timestamp to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(transactions_df['timestamp']):
                transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'], errors='coerce')
            
            # Calculate transaction frequency
            tx_counts = transactions_df.groupby('user_id').size().reset_index(name='transaction_count')
            user_behavior_df = user_behavior_df.merge(tx_counts, on='user_id', how='left')
            
            # Calculate average transaction amount
            if 'amount' in transactions_df.columns:
                avg_amounts = transactions_df.groupby('user_id')['amount'].mean().reset_index(name='avg_transaction_amount')
                user_behavior_df = user_behavior_df.merge(avg_amounts, on='user_id', how='left')
                
                # Calculate transaction amount variance
                var_amounts = transactions_df.groupby('user_id')['amount'].var().reset_index(name='var_transaction_amount')
                user_behavior_df = user_behavior_df.merge(var_amounts, on='user_id', how='left')
                
                # Calculate max transaction amount
                max_amounts = transactions_df.groupby('user_id')['amount'].max().reset_index(name='max_transaction_amount')
                user_behavior_df = user_behavior_df.merge(max_amounts, on='user_id', how='left')
            
            # Calculate time between transactions
            transactions_df = transactions_df.sort_values(['user_id', 'timestamp'])
            transactions_df['prev_timestamp'] = transactions_df.groupby('user_id')['timestamp'].shift(1)
            transactions_df['time_diff'] = (transactions_df['timestamp'] - transactions_df['prev_timestamp']).dt.total_seconds() / 3600  # in hours
            
            # Calculate average time between transactions
            avg_time_diffs = transactions_df.groupby('user_id')['time_diff'].mean().reset_index(name='avg_hours_between_transactions')
            user_behavior_df = user_behavior_df.merge(avg_time_diffs, on='user_id', how='left')
            
            # Calculate number of different days with transactions
            transactions_df['transaction_date'] = transactions_df['timestamp'].dt.date
            unique_days = transactions_df.groupby('user_id')['transaction_date'].nunique().reset_index(name='unique_transaction_days')
            user_behavior_df = user_behavior_df.merge(unique_days, on='user_id', how='left')
        
        # Extract login behavior features if login data is available
        if logins_df is not None and 'user_id' in logins_df.columns:
            # Calculate login frequency
            login_counts = logins_df.groupby('user_id').size().reset_index(name='login_count')
            user_behavior_df = user_behavior_df.merge(login_counts, on='user_id', how='left')
            
            # Calculate number of different devices used
            if 'device_id' in logins_df.columns:
                device_counts = logins_df.groupby('user_id')['device_id'].nunique().reset_index(name='unique_device_count')
                user_behavior_df = user_behavior_df.merge(device_counts, on='user_id', how='left')
            
            # Calculate number of different IP addresses used
            if 'ip_address' in logins_df.columns:
                ip_counts = logins_df.groupby('user_id')['ip_address'].nunique().reset_index(name='unique_ip_count')
                user_behavior_df = user_behavior_df.merge(ip_counts, on='user_id', how='left')
            
            # Calculate number of different locations
            if 'country' in logins_df.columns:
                country_counts = logins_df.groupby('user_id')['country'].nunique().reset_index(name='unique_country_count')
                user_behavior_df = user_behavior_df.merge(country_counts, on='user_id', how='left')
            
            # Calculate login time variance
            if 'timestamp' in logins_df.columns:
                # Convert timestamp to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(logins_df['timestamp']):
                    logins_df['timestamp'] = pd.to_datetime(logins_df['timestamp'], errors='coerce')
                
                # Extract hour of day
                logins_df['login_hour'] = logins_df['timestamp'].dt.hour
                
                # Calculate variance in login hour
                hour_vars = logins_df.groupby('user_id')['login_hour'].var().reset_index(name='login_hour_variance')
                user_behavior_df = user_behavior_df.merge(hour_vars, on='user_id', how='left')
        
        # Fill missing values
        user_behavior_df.fillna(0, inplace=True)
        
        return user_behavior_df
    
    def extract_device_features(self, df, device_col='device_id', browser_col='browser', os_col='operating_system'):
        """
        Extract features from device data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing device information
        device_col : str
            Name of the device ID column
        browser_col : str
            Name of the browser column
        os_col : str
            Name of the operating system column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted device features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create device type feature if available
        if 'device_type' in result_df.columns:
            # One-hot encode device type
            device_dummies = pd.get_dummies(result_df['device_type'], prefix='device')
            result_df = pd.concat([result_df, device_dummies], axis=1)
        
        # Create browser risk score if available
        if browser_col in result_df.columns:
            # In a real implementation, this would be based on actual data
            high_risk_browsers = ['BrowserA', 'BrowserB']
            medium_risk_browsers = ['BrowserC', 'BrowserD']
            
            result_df['browser_risk'] = 'low'
            result_df.loc[result_df[browser_col].isin(medium_risk_browsers), 'browser_risk'] = 'medium'
            result_df.loc[result_df[browser_col].isin(high_risk_browsers), 'browser_risk'] = 'high'
            
            # One-hot encode browser risk
            browser_risk_dummies = pd.get_dummies(result_df['browser_risk'], prefix='browser_risk')
            result_df = pd.concat([result_df, browser_risk_dummies], axis=1)
        
        # Create OS risk score if available
        if os_col in result_df.columns:
            # In a real implementation, this would be based on actual data
            high_risk_os = ['OSA', 'OSB']
            medium_risk_os = ['OSC', 'OSD']
            
            result_df['os_risk'] = 'low'
            result_df.loc[result_df[os_col].isin(medium_risk_os), 'os_risk'] = 'medium'
            result_df.loc[result_df[os_col].isin(high_risk_os), 'os_risk'] = 'high'
            
            # One-hot encode OS risk
            os_risk_dummies = pd.get_dummies(result_df['os_risk'], prefix='os_risk')
            result_df = pd.concat([result_df, os_risk_dummies], axis=1)
        
        return result_df
    
    def prepare_fraud_detection_data(self):
        """
        Prepare data for fraud detection by combining and transforming all available data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all features for fraud detection
        """
        if self.transactions_df is None:
            raise ValueError("Transaction data is required. Call load_data() first.")
        
        # Start with transaction data
        fraud_data = self.transactions_df.copy()
        
        # Extract datetime features
        if 'timestamp' in fraud_data.columns:
            fraud_data = self.extract_datetime_features(fraud_data, 'timestamp', 'tx_')
        
        # Extract location features
        location_cols = {
            'ip_col': 'ip_address' if 'ip_address' in fraud_data.columns else None,
            'country_col': 'country' if 'country' in fraud_data.columns else None,
            'city_col': 'city' if 'city' in fraud_data.columns else None,
            'zip_col': 'zip_code' if 'zip_code' in fraud_data.columns else None
        }
        fraud_data = self.extract_location_features(fraud_data, **location_cols)
        
        # Extract payment features
        fraud_data = self.extract_payment_features(fraud_data)
        
        # Merge with user data if available
        if self.users_df is not None and 'user_id' in fraud_data.columns:
            # Select relevant user columns
            user_cols = ['user_id']
            for col in self.users_df.columns:
                if col != 'user_id' and not col.startswith('password') and not col.startswith('secret'):
                    user_cols.append(col)
            
            # Merge with transaction data
            fraud_data = fraud_data.merge(self.users_df[user_cols], on='user_id', how='left')
            
            # Extract user account age if registration_date is available
            if 'registration_date' in self.users_df.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(self.users_df['registration_date']):
                    self.users_df['registration_date'] = pd.to_datetime(self.users_df['registration_date'], errors='coerce')
                
                # Calculate account age at transaction time
                if 'timestamp' in fraud_data.columns:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(fraud_data['timestamp']):
                        fraud_data['timestamp'] = pd.to_datetime(fraud_data['timestamp'], errors='coerce')
                    
                    # Merge registration_date
                    fraud_data = fraud_data.merge(
                        self.users_df[['user_id', 'registration_date']], 
                        on='user_id', 
                        how='left'
                    )
                    
                    # Calculate account age in days
                    fraud_data['account_age_days'] = (fraud_data['timestamp'] - fraud_data['registration_date']).dt.total_seconds() / (24 * 3600)
                    
                    # Create account age risk categories
                    fraud_data['new_account'] = (fraud_data['account_age_days'] < 30).astype(int)
                    fraud_data['very_new_account'] = (fraud_data['account_age_days'] < 7).astype(int)
        
        # Extract user behavior features
        user_behavior_df = self.extract_user_behavior_features()
        if 'user_id' in fraud_data.columns:
            fraud_data = fraud_data.merge(user_behavior_df, on='user_id', how='left')
        
        # Merge with device data if available
        if self.devices_df is not None and 'device_id' in fraud_data.columns:
            # Extract device features
            device_features = self.extract_device_features(
                self.devices_df,
                device_col='device_id',
                browser_col='browser' if 'browser' in self.devices_df.columns else None,
                os_col='operating_system' if 'operating_system' in self.devices_df.columns else None
            )
            
            # Merge with transaction data
            fraud_data = fraud_data.merge(device_features, on='device_id', how='left')
        
        # Add chargeback history if available
        if self.chargebacks_df is not None and 'user_id' in fraud_data.columns:
            # Calculate chargeback counts per user
            if 'user_id' in self.chargebacks_df.columns:
                chargeback_counts = self.chargebacks_df.groupby('user_id').size().reset_index(name='chargeback_count')
                fraud_data = fraud_data.merge(chargeback_counts, on='user_id', how='left')
                fraud_data['chargeback_count'].fillna(0, inplace=True)
                
                # Create chargeback risk features
                fraud_data['has_chargeback_history'] = (fraud_data['chargeback_count'] > 0).astype(int)
                fraud_data['multiple_chargebacks'] = (fraud_data['chargeback_count'] > 1).astype(int)
        
        # Handle categorical variables
        cat_cols = fraud_data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            # Skip ID columns and timestamp columns
            if col.endswith('_id') or col.endswith('timestamp') or col.endswith('date'):
                continue
            
            # One-hot encode categorical variables with low cardinality
            if fraud_data[col].nunique() < 10:  # Adjust threshold as needed
                dummies = pd.get_dummies(fraud_data[col], prefix=col)
                fraud_data = pd.concat([fraud_data, dummies], axis=1)
        
        # Scale numerical features
        num_cols = fraud_data.select_dtypes(include=['number']).columns
        # Exclude ID columns and binary features
        num_cols = [col for col in num_cols if not col.endswith('_id') and fraud_data[col].nunique() > 2]
        
        if num_cols:
            self.numeric_scaler = StandardScaler()
            fraud_data[num_cols] = self.numeric_scaler.fit_transform(fraud_data[num_cols])
        
        # Fill any remaining missing values
        fraud_data.fillna(0, inplace=True)
        
        # Store feature lists for later use
        self.transaction_features = [col for col in fraud_data.columns if col.startswith('tx_') or 
                                    col in ['amount', 'amount_scaled', 'payment_method']]
        
        self.user_features = [col for col in fraud_data.columns if col.startswith('user_') or 
                             col in ['account_age_days', 'new_account', 'very_new_account']]
        
        self.device_features = [col for col in fraud_data.columns if col.startswith('device_') or 
                               col.startswith('browser_') or col.startswith('os_')]
        
        self.behavioral_features = [col for col in fraud_data.columns if col in user_behavior_df.columns and 
                                   col != 'user_id']
        
        return fraud_data
    
    def split_data(self, df, target_col='is_fraud', test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to split
        target_col : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test) DataFrames and Series
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filename='fraud_preprocessor.pkl'):
        """
        Save the preprocessor to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the preprocessor to
        """
        model_dir = os.path.join(os.path.dirname(self.data_dir), 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, filename)
        
        # Save only the necessary components, not the data
        preprocessor_data = {
            'numeric_scaler': self.numeric_scaler,
            'numeric_imputer': self.numeric_imputer,
            'transaction_features': self.transaction_features,
            'user_features': self.user_features,
            'device_features': self.device_features,
            'behavioral_features': self.behavioral_features
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
    
    def load_preprocessor(self, filename='fraud_preprocessor.pkl'):
        """
        Load the preprocessor from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the preprocessor from
            
        Returns:
        --------
        self : FraudDataPreprocessor
            Returns self for method chaining
        """
        model_dir = os.path.join(os.path.dirname(self.data_dir), 'model')
        model_path = os.path.join(model_dir, filename)
        
        with open(model_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.numeric_scaler = preprocessor_data['numeric_scaler']
        self.numeric_imputer = preprocessor_data['numeric_imputer']
        self.transaction_features = preprocessor_data['transaction_features']
        self.user_features = preprocessor_data['user_features']
        self.device_features = preprocessor_data['device_features']
        self.behavioral_features = preprocessor_data['behavioral_features']
        
        return self
    
    def create_sample_data(self, n_transactions=1000, n_users=100, n_devices=150, 
                          fraud_ratio=0.05, save_to_csv=True):
        """
        Create sample data for demonstration purposes.
        
        Parameters:
        -----------
        n_transactions : int
            Number of transactions to generate
        n_users : int
            Number of users to generate
        n_devices : int
            Number of devices to generate
        fraud_ratio : float
            Proportion of fraudulent transactions
        save_to_csv : bool
            Whether to save the generated data to CSV files
            
        Returns:
        --------
        tuple
            (transactions_df, users_df, logins_df, chargebacks_df, devices_df) DataFrames
        """
        np.random.seed(42)
        
        # Generate user data
        user_ids = [f"U{i:05d}" for i in range(1, n_users + 1)]
        
        users_data = {
            'user_id': user_ids,
            'name': [f"User {i}" for i in range(1, n_users + 1)],
            'email': [f"user{i}@example.com" for i in range(1, n_users + 1)],
            'registration_date': pd.date_range(start='2020-01-01', periods=n_users, freq='D'),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'CountryA', 'CountryB'], n_users),
            'phone': [f"+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_users)]
        }
        
        users_df = pd.DataFrame(users_data)
        
        # Generate device data
        device_ids = [f"D{i:05d}" for i in range(1, n_devices + 1)]
        
        devices_data = {
            'device_id': device_ids,
            'user_id': np.random.choice(user_ids, n_devices),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_devices),
            'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge', 'BrowserA'], n_devices),
            'operating_system': np.random.choice(['Windows', 'MacOS', 'iOS', 'Android', 'Linux', 'OSA'], n_devices),
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_devices)]
        }
        
        devices_df = pd.DataFrame(devices_data)
        
        # Generate login data
        n_logins = n_users * 10  # Average 10 logins per user
        
        logins_data = {
            'login_id': [f"L{i:06d}" for i in range(1, n_logins + 1)],
            'user_id': np.random.choice(user_ids, n_logins),
            'timestamp': pd.date_range(start='2023-01-01', periods=n_logins, freq='H'),
            'device_id': np.random.choice(device_ids, n_logins),
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_logins)],
            'success': np.random.choice([True, False], n_logins, p=[0.95, 0.05]),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'CountryA', 'CountryB'], n_logins)
        }
        
        logins_df = pd.DataFrame(logins_data)
        
        # Generate transaction data
        transaction_ids = [f"T{i:06d}" for i in range(1, n_transactions + 1)]
        
        # Determine which transactions are fraudulent
        n_fraud = int(n_transactions * fraud_ratio)
        is_fraud = np.zeros(n_transactions, dtype=bool)
        is_fraud[:n_fraud] = True
        np.random.shuffle(is_fraud)
        
        # Generate transaction data with fraud patterns
        transactions_data = {
            'transaction_id': transaction_ids,
            'user_id': np.random.choice(user_ids, n_transactions),
            'timestamp': pd.date_range(start='2023-01-01', periods=n_transactions, freq='30min'),
            'amount': np.zeros(n_transactions),
            'payment_method': np.random.choice(['credit_card', 'debit_card', 'digital_wallet', 'bank_transfer', 'gift_card'], n_transactions),
            'device_id': np.random.choice(device_ids, n_transactions),
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_transactions)],
            'billing_country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France'], n_transactions),
            'shipping_country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France'], n_transactions),
            'is_fraud': is_fraud
        }
        
        # Set transaction amounts (higher for fraudulent transactions)
        for i in range(n_transactions):
            if is_fraud[i]:
                # Fraudulent transactions tend to have higher amounts
                transactions_data['amount'][i] = np.random.uniform(100, 1000)
                
                # Increase likelihood of country mismatch for fraudulent transactions
                if np.random.random() < 0.7:
                    transactions_data['shipping_country'][i] = np.random.choice(['CountryA', 'CountryB', 'CountryC'])
                
                # Increase likelihood of unusual payment methods for fraudulent transactions
                if np.random.random() < 0.6:
                    transactions_data['payment_method'][i] = np.random.choice(['gift_card', 'cryptocurrency', 'prepaid_card'])
            else:
                # Legitimate transactions have more normal amounts
                transactions_data['amount'][i] = np.random.uniform(10, 500)
        
        transactions_df = pd.DataFrame(transactions_data)
        
        # Generate chargeback data
        # About 80% of fraudulent transactions and 1% of legitimate transactions result in chargebacks
        fraud_tx_ids = transactions_df[transactions_df['is_fraud']]['transaction_id'].values
        legit_tx_ids = transactions_df[~transactions_df['is_fraud']]['transaction_id'].values
        
        fraud_chargebacks = np.random.choice(fraud_tx_ids, size=int(len(fraud_tx_ids) * 0.8), replace=False)
        legit_chargebacks = np.random.choice(legit_tx_ids, size=int(len(legit_tx_ids) * 0.01), replace=False)
        
        chargeback_tx_ids = np.concatenate([fraud_chargebacks, legit_chargebacks])
        n_chargebacks = len(chargeback_tx_ids)
        
        chargebacks_data = {
            'chargeback_id': [f"C{i:05d}" for i in range(1, n_chargebacks + 1)],
            'transaction_id': chargeback_tx_ids,
            'timestamp': pd.date_range(start='2023-02-01', periods=n_chargebacks, freq='D'),
            'reason': np.random.choice(['unauthorized', 'product_not_received', 'product_not_as_described', 'duplicate', 'fraud'], n_chargebacks),
            'status': np.random.choice(['pending', 'approved', 'rejected'], n_chargebacks)
        }
        
        # Add user_id to chargebacks by merging with transactions
        chargebacks_df = pd.DataFrame(chargebacks_data)
        tx_user_map = transactions_df[['transaction_id', 'user_id']].set_index('transaction_id')
        chargebacks_df['user_id'] = chargebacks_df['transaction_id'].map(tx_user_map['user_id'])
        
        # Save to CSV if requested
        if save_to_csv:
            transactions_df.to_csv(os.path.join(self.data_dir, 'sample_transactions.csv'), index=False)
            users_df.to_csv(os.path.join(self.data_dir, 'sample_users.csv'), index=False)
            logins_df.to_csv(os.path.join(self.data_dir, 'sample_logins.csv'), index=False)
            chargebacks_df.to_csv(os.path.join(self.data_dir, 'sample_chargebacks.csv'), index=False)
            devices_df.to_csv(os.path.join(self.data_dir, 'sample_devices.csv'), index=False)
        
        # Update instance variables
        self.transactions_df = transactions_df
        self.users_df = users_df
        self.logins_df = logins_df
        self.chargebacks_df = chargebacks_df
        self.devices_df = devices_df
        
        return transactions_df, users_df, logins_df, chargebacks_df, devices_df
