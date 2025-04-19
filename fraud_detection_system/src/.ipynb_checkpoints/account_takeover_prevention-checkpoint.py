"""
Account Takeover Prevention Module for E-commerce

This module implements detection mechanisms for account takeover attempts
in e-commerce systems, including login behavior analysis, device fingerprinting,
and location-based anomaly detection.
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

class AccountTakeoverDetector:
    """
    A class for detecting account takeover attempts in e-commerce systems.
    
    This class implements various methods for identifying potentially fraudulent
    login attempts and suspicious account activities.
    """
    
    def __init__(self, model_dir='../model'):
        """
        Initialize the AccountTakeoverDetector.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load model files
        """
        self.model_dir = model_dir
        self.login_behavior_model = None
        self.location_anomaly_model = None
        self.device_fingerprint_db = {}
        self.user_profiles = {}
        self.feature_importance = None
        
        # Thresholds for detection
        self.thresholds = {
            'login_attempts_1h': 5,       # Max login attempts in 1 hour
            'failed_ratio_threshold': 0.5, # Ratio of failed to total login attempts
            'location_distance_km': 500,   # Max distance between login locations (km)
            'new_device_risk': 0.6,        # Risk score for new device
            'password_change_window': 24,  # Hours to monitor after password change
            'suspicious_ip_risk': 0.7      # Risk score for suspicious IP
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
        self : AccountTakeoverDetector
            Returns self for method chaining
        """
        self.thresholds.update(thresholds)
        return self
    
    def train_login_behavior_model(self, X_train, y_train, optimize_hyperparams=False, class_weight='balanced'):
        """
        Train a machine learning model for login behavior analysis.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (suspicious=1, legitimate=0)
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        class_weight : str or dict
            Class weights to handle imbalanced data
            
        Returns:
        --------
        self : AccountTakeoverDetector
            Returns self for method chaining
        """
        # Create and train model
        self.login_behavior_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight=class_weight
        )
        
        self.login_behavior_model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.login_behavior_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def train_location_anomaly_model(self, user_locations, eps=100, min_samples=2):
        """
        Train a location anomaly detection model using clustering.
        
        Parameters:
        -----------
        user_locations : pd.DataFrame
            DataFrame with user locations (latitude, longitude)
        eps : float
            Maximum distance between points in a cluster
        min_samples : int
            Minimum number of samples in a cluster
            
        Returns:
        --------
        self : AccountTakeoverDetector
            Returns self for method chaining
        """
        # Create a DBSCAN clustering model for location anomaly detection
        self.location_anomaly_model = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Fit the model
        self.location_anomaly_model.fit(user_locations[['latitude', 'longitude']])
        
        return self
    
    def calculate_location_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two geographic coordinates using the Haversine formula.
        
        Parameters:
        -----------
        lat1, lon1 : float
            Latitude and longitude of the first point
        lat2, lon2 : float
            Latitude and longitude of the second point
            
        Returns:
        --------
        float
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def generate_device_fingerprint(self, device_data):
        """
        Generate a unique fingerprint for a device based on its characteristics.
        
        Parameters:
        -----------
        device_data : dict
            Dictionary with device characteristics
            
        Returns:
        --------
        str
            Device fingerprint hash
        """
        # Create a string representation of device data
        device_str = json.dumps(device_data, sort_keys=True)
        
        # Generate a hash
        fingerprint = hashlib.sha256(device_str.encode()).hexdigest()
        
        return fingerprint
    
    def is_known_device(self, user_id, device_fingerprint):
        """
        Check if a device is known for a specific user.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        device_fingerprint : str
            Device fingerprint hash
            
        Returns:
        --------
        bool
            True if the device is known for the user, False otherwise
        """
        if user_id not in self.device_fingerprint_db:
            self.device_fingerprint_db[user_id] = []
            return False
        
        return device_fingerprint in self.device_fingerprint_db[user_id]
    
    def add_known_device(self, user_id, device_fingerprint):
        """
        Add a device to the list of known devices for a user.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        device_fingerprint : str
            Device fingerprint hash
            
        Returns:
        --------
        self : AccountTakeoverDetector
            Returns self for method chaining
        """
        if user_id not in self.device_fingerprint_db:
            self.device_fingerprint_db[user_id] = []
        
        if device_fingerprint not in self.device_fingerprint_db[user_id]:
            self.device_fingerprint_db[user_id].append(device_fingerprint)
        
        return self
    
    def build_user_profile(self, user_id, login_history):
        """
        Build a behavioral profile for a user based on login history.
        
        Parameters:
        -----------
        user_id : str
            User identifier
        login_history : pd.DataFrame
            DataFrame with user's login history
            
        Returns:
        --------
        dict
            User profile with behavioral patterns
        """
        if login_history.empty:
            return {
                'user_id': user_id,
                'login_times': [],
                'login_days': [],
                'devices': [],
                'locations': [],
                'typical_ips': [],
                'last_login': None
            }
        
        # Extract login times and days
        login_history['hour'] = login_history['timestamp'].dt.hour
        login_history['day_of_week'] = login_history['timestamp'].dt.dayofweek
        
        # Calculate common login times and days
        login_times = login_history['hour'].value_counts().to_dict()
        login_days = login_history['day_of_week'].value_counts().to_dict()
        
        # Extract devices and locations
        devices = login_history['device_id'].unique().tolist() if 'device_id' in login_history.columns else []
        
        locations = []
        if 'latitude' in login_history.columns and 'longitude' in login_history.columns:
            for _, row in login_history.iterrows():
                locations.append((row['latitude'], row['longitude']))
        
        # Extract typical IPs
        typical_ips = login_history['ip_address'].value_counts().to_dict() if 'ip_address' in login_history.columns else {}
        
        # Get last login
        last_login = login_history['timestamp'].max()
        
        # Create user profile
        profile = {
            'user_id': user_id,
            'login_times': login_times,
            'login_days': login_days,
            'devices': devices,
            'locations': locations,
            'typical_ips': typical_ips,
            'last_login': last_login
        }
        
        # Store the profile
        self.user_profiles[user_id] = profile
        
        return profile
    
    def analyze_login_attempt(self, login_data, user_profile=None, login_history=None):
        """
        Analyze a login attempt for suspicious activity.
        
        Parameters:
        -----------
        login_data : dict or pd.Series
            Data about the current login attempt
        user_profile : dict
            User's behavioral profile
        login_history : pd.DataFrame
            User's login history
            
        Returns:
        --------
        dict
            Analysis results with risk scores
        """
        # Convert login_data to Series if it's a dict
        if isinstance(login_data, dict):
            login_data = pd.Series(login_data)
        
        # Initialize results
        results = {
            'login_id': login_data.get('login_id', None),
            'user_id': login_data.get('user_id', None),
            'timestamp': login_data.get('timestamp', None),
            'checks': {},
            'risk_scores': {},
            'overall_risk_score': None
        }
        
        # If no user profile is provided, try to get it from stored profiles
        if user_profile is None and 'user_id' in login_data:
            user_profile = self.user_profiles.get(login_data['user_id'], None)
        
        # If no login history is provided, create an empty DataFrame
        if login_history is None:
            login_history = pd.DataFrame(columns=login_data.index)
        
        # Check 1: Login time analysis
        time_risk = self.analyze_login_time(login_data, user_profile)
        results['checks']['login_time'] = time_risk
        results['risk_scores']['login_time'] = time_risk['risk_score']
        
        # Check 2: Device fingerprinting
        if 'device_id' in login_data or 'device_data' in login_data:
            device_risk = self.analyze_device(login_data, user_profile)
            results['checks']['device'] = device_risk
            results['risk_scores']['device'] = device_risk['risk_score']
        
        # Check 3: Location analysis
        if ('latitude' in login_data and 'longitude' in login_data) or 'ip_address' in login_data:
            location_risk = self.analyze_location(login_data, user_profile)
            results['checks']['location'] = location_risk
            results['risk_scores']['location'] = location_risk['risk_score']
        
        # Check 4: Login velocity and failed attempts
        if not login_history.empty:
            velocity_risk = self.analyze_login_velocity(login_data, login_history)
            results['checks']['velocity'] = velocity_risk
            results['risk_scores']['velocity'] = velocity_risk['risk_score']
        
        # Check 5: Password change/recovery analysis
        if 'event_type' in login_data:
            password_risk = self.analyze_password_events(login_data, login_history)
            results['checks']['password'] = password_risk
            results['risk_scores']['password'] = password_risk['risk_score']
        
        # Calculate overall risk score (weighted average of all risk scores)
        weights = {
            'login_time': 0.15,
            'device': 0.25,
            'location': 0.25,
            'velocity': 0.2,
            'password': 0.15
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
            results['recommended_action'] = 'block_and_notify'
        elif results['overall_risk_score'] >= 0.4:
            results['risk_level'] = 'medium'
            results['recommended_action'] = 'additional_verification'
        else:
            results['risk_level'] = 'low'
            results['recommended_action'] = 'allow'
        
        return results
    
    def analyze_login_time(self, login_data, user_profile):
        """
        Analyze the login time for anomalies.
        
        Parameters:
        -----------
        login_data : pd.Series
            Data about the current login attempt
        user_profile : dict
            User's behavioral profile
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Default risk for new users or missing data
        if user_profile is None or 'login_times' not in user_profile or not user_profile['login_times']:
            return {
                'unusual_time': False,
                'unusual_day': False,
                'risk_score': 0.3  # Moderate risk for new users
            }
        
        # Extract current login time and day
        if isinstance(login_data['timestamp'], str):
            login_time = pd.to_datetime(login_data['timestamp'])
        else:
            login_time = login_data['timestamp']
        
        current_hour = login_time.hour
        current_day = login_time.dayofweek
        
        # Check if the login time is common for this user
        login_times = user_profile['login_times']
        login_days = user_profile['login_days']
        
        # Calculate total logins
        total_logins = sum(login_times.values())
        
        # Calculate frequency of current hour and day
        hour_frequency = login_times.get(current_hour, 0) / total_logins if total_logins > 0 else 0
        day_frequency = login_days.get(current_day, 0) / total_logins if total_logins > 0 else 0
        
        # Determine if time is unusual
        unusual_time = hour_frequency < 0.1  # Less than 10% of logins at this hour
        unusual_day = day_frequency < 0.1    # Less than 10% of logins on this day
        
        # Calculate risk score
        risk_score = 0.0
        
        if unusual_time and unusual_day:
            risk_score = 0.8  # High risk
        elif unusual_time:
            risk_score = 0.6  # Medium-high risk
        elif unusual_day:
            risk_score = 0.4  # Medium risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'unusual_time': unusual_time,
            'unusual_day': unusual_day,
            'hour_frequency': hour_frequency,
            'day_frequency': day_frequency,
            'risk_score': risk_score
        }
    
    def analyze_device(self, login_data, user_profile):
        """
        Analyze the device used for login.
        
        Parameters:
        -----------
        login_data : pd.Series
            Data about the current login attempt
        user_profile : dict
            User's behavioral profile
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Generate device fingerprint
        device_fingerprint = None
        
        if 'device_data' in login_data:
            device_fingerprint = self.generate_device_fingerprint(login_data['device_data'])
        elif 'device_id' in login_data:
            device_fingerprint = str(login_data['device_id'])
        
        if device_fingerprint is None:
            return {
                'known_device': False,
                'risk_score': 0.5  # Medium risk for unknown device data
            }
        
        # Check if device is known for this user
        known_device = False
        
        if user_profile is not None and 'devices' in user_profile:
            known_device = device_fingerprint in user_profile['devices'] or self.is_known_device(login_data['user_id'], device_fingerprint)
        
        # Calculate risk score
        if known_device:
            risk_score = 0.1  # Low risk for known device
        else:
            risk_score = self.thresholds['new_device_risk']  # Higher risk for new device
        
        return {
            'known_device': known_device,
            'device_fingerprint': device_fingerprint,
            'risk_score': risk_score
        }
    
    def analyze_location(self, login_data, user_profile):
        """
        Analyze the login location for anomalies.
        
        Parameters:
        -----------
        login_data : pd.Series
            Data about the current login attempt
        user_profile : dict
            User's behavioral profile
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Default risk for new users or missing data
        if user_profile is None or 'locations' not in user_profile or not user_profile['locations']:
            return {
                'location_anomaly': False,
                'distance_from_known': None,
                'risk_score': 0.4  # Moderate risk for new users
            }
        
        # Extract current location
        current_lat, current_lon = None, None
        
        if 'latitude' in login_data and 'longitude' in login_data:
            current_lat = login_data['latitude']
            current_lon = login_data['longitude']
        elif 'ip_address' in login_data:
            # In a real implementation, this would use IP geolocation
            # For this example, we'll just use a placeholder
            current_lat, current_lon = 40.7128, -74.0060  # New York City coordinates
        
        if current_lat is None or current_lon is None:
            return {
                'location_anomaly': False,
                'distance_from_known': None,
                'risk_score': 0.3  # Moderate risk for missing location
            }
        
        # Calculate distance from known locations
        min_distance = float('inf')
        
        for lat, lon in user_profile['locations']:
            distance = self.calculate_location_distance(current_lat, current_lon, lat, lon)
            min_distance = min(min_distance, distance)
        
        # Determine if location is anomalous
        location_anomaly = min_distance > self.thresholds['location_distance_km']
        
        # Calculate risk score
        if location_anomaly:
            # Scale risk based on distance
            risk_score = min(0.9, 0.5 + (min_distance / (2 * self.thresholds['location_distance_km'])) * 0.4)
        else:
            risk_score = 0.1  # Low risk for known location
        
        return {
            'location_anomaly': location_anomaly,
            'distance_from_known': min_distance,
            'risk_score': risk_score
        }
    
    def analyze_login_velocity(self, login_data, login_history):
        """
        Analyze login velocity and failed attempts.
        
        Parameters:
        -----------
        login_data : pd.Series
            Data about the current login attempt
        login_history : pd.DataFrame
            User's login history
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Ensure timestamp is datetime
        if isinstance(login_data['timestamp'], str):
            current_time = pd.to_datetime(login_data['timestamp'])
        else:
            current_time = login_data['timestamp']
        
        # Calculate time windows
        one_hour_ago = current_time - pd.Timedelta(hours=1)
        
        # Ensure login_history timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(login_history['timestamp'].iloc[0]):
            login_history['timestamp'] = pd.to_datetime(login_history['timestamp'])
        
        # Get login attempts in the last hour
        recent_logins = login_history[login_history['timestamp'] >= one_hour_ago]
        
        # Count login attempts and failed attempts
        login_attempts = len(recent_logins) + 1  # Include current attempt
        
        failed_attempts = 0
        if 'success' in recent_logins.columns:
            failed_attempts = sum(~recent_logins['success'])
        
        # Calculate failed ratio
        failed_ratio = failed_attempts / login_attempts if login_attempts > 0 else 0
        
        # Determine if velocity is anomalous
        velocity_anomaly = login_attempts > self.thresholds['login_attempts_1h']
        failed_anomaly = failed_ratio > self.thresholds['failed_ratio_threshold']
        
        # Calculate risk score
        risk_score = 0.0
        
        if velocity_anomaly and failed_anomaly:
            risk_score = 0.9  # Very high risk
        elif velocity_anomaly:
            risk_score = 0.7  # High risk
        elif failed_anomaly:
            risk_score = 0.6  # Medium-high risk
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'login_attempts_1h': login_attempts,
            'failed_attempts_1h': failed_attempts,
            'failed_ratio': failed_ratio,
            'velocity_anomaly': velocity_anomaly,
            'failed_anomaly': failed_anomaly,
            'risk_score': risk_score
        }
    
    def analyze_password_events(self, login_data, login_history):
        """
        Analyze password change and recovery events.
        
        Parameters:
        -----------
        login_data : pd.Series
            Data about the current login attempt
        login_history : pd.DataFrame
            User's login history
            
        Returns:
        --------
        dict
            Analysis results with risk score
        """
        # Default for regular login
        if 'event_type' not in login_data or login_data['event_type'] not in ['password_change', 'password_reset', 'login_after_reset']:
            return {
                'recent_password_change': False,
                'risk_score': 0.1  # Low risk for regular login
            }
        
        # Ensure timestamp is datetime
        if isinstance(login_data['timestamp'], str):
            current_time = pd.to_datetime(login_data['timestamp'])
        else:
            current_time = login_data['timestamp']
        
        # Check for recent password changes or resets
        password_window = current_time - pd.Timedelta(hours=self.thresholds['password_change_window'])
        
        # Ensure login_history timestamps are datetime
        if not login_history.empty and not pd.api.types.is_datetime64_any_dtype(login_history['timestamp'].iloc[0]):
            login_history['timestamp'] = pd.to_datetime(login_history['timestamp'])
        
        # Look for password events in history
        recent_password_events = login_history[
            (login_history['timestamp'] >= password_window) & 
            (login_history['event_type'].isin(['password_change', 'password_reset']))
        ]
        
        recent_password_change = len(recent_password_events) > 0 or login_data['event_type'] in ['password_change', 'password_reset']
        
        # Calculate risk score
        risk_score = 0.0
        
        if login_data['event_type'] == 'password_reset':
            risk_score = 0.7  # High risk for password reset
        elif login_data['event_type'] == 'password_change':
            risk_score = 0.5  # Medium risk for password change
        elif login_data['event_type'] == 'login_after_reset':
            risk_score = 0.6  # Medium-high risk for login after reset
        elif recent_password_change:
            risk_score = 0.4  # Medium risk for login after recent password change
        else:
            risk_score = 0.1  # Low risk
        
        return {
            'event_type': login_data['event_type'],
            'recent_password_change': recent_password_change,
            'risk_score': risk_score
        }
    
    def predict_login_risk(self, login_features):
        """
        Predict the risk of a login attempt using the ML model.
        
        Parameters:
        -----------
        login_features : pd.DataFrame
            Features of the login attempt
            
        Returns:
        --------
        float
            Risk score (0-1)
        """
        if self.login_behavior_model is None:
            raise ValueError("Model not trained. Call train_login_behavior_model() first.")
        
        # Ensure login_features is a DataFrame
        if isinstance(login_features, pd.Series):
            login_features = pd.DataFrame([login_features])
        
        # Predict probability
        risk_score = self.login_behavior_model.predict_proba(login_features)[0, 1]
        
        return risk_score
    
    def detect_location_anomaly(self, latitude, longitude, user_locations=None):
        """
        Detect if a location is anomalous using the location anomaly model.
        
        Parameters:
        -----------
        latitude : float
            Latitude of the location
        longitude : float
            Longitude of the location
        user_locations : pd.DataFrame
            DataFrame with user's previous locations
            
        Returns:
        --------
        dict
            Dictionary with anomaly detection results
        """
        if self.location_anomaly_model is None and user_locations is None:
            return {
                'is_anomaly': False,
                'confidence': 0.5
            }
        
        # If we have a trained model, use it
        if self.location_anomaly_model is not None:
            # Create a point for the current location
            point = np.array([[latitude, longitude]])
            
            # Predict cluster
            cluster = self.location_anomaly_model.fit_predict(point)[0]
            
            # If cluster is -1, it's an outlier
            is_anomaly = cluster == -1
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': 0.8 if is_anomaly else 0.2
            }
        
        # If we don't have a model but have user locations, calculate distances
        elif user_locations is not None:
            min_distance = float('inf')
            
            for _, row in user_locations.iterrows():
                distance = self.calculate_location_distance(
                    latitude, longitude, row['latitude'], row['longitude']
                )
                min_distance = min(min_distance, distance)
            
            # Determine if location is anomalous based on distance
            is_anomaly = min_distance > self.thresholds['location_distance_km']
            
            # Calculate confidence based on distance
            if is_anomaly:
                confidence = min(0.9, 0.5 + (min_distance / (2 * self.thresholds['location_distance_km'])) * 0.4)
            else:
                confidence = 0.2
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'distance': min_distance
            }
        
        # Default response if no model or user locations
        return {
            'is_anomaly': False,
            'confidence': 0.5
        }
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the login behavior model on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels (suspicious=1, legitimate=0)
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.login_behavior_model is None:
            raise ValueError("Model not trained. Call train_login_behavior_model() first.")
        
        # Make predictions
        y_pred = self.login_behavior_model.predict(X_test)
        y_pred_proba = self.login_behavior_model.predict_proba(X_test)[:, 1]
        
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
        plt.title(f'Top {top_n} Features for Account Takeover Detection')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_model(self, filename='account_takeover_model.pkl'):
        """
        Save the trained models and configuration to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save the model to
        """
        model_path = os.path.join(self.model_dir, filename)
        
        model_data = {
            'login_behavior_model': self.login_behavior_model,
            'location_anomaly_model': self.location_anomaly_model,
            'feature_importance': self.feature_importance,
            'thresholds': self.thresholds,
            'user_profiles': self.user_profiles,
            'device_fingerprint_db': self.device_fingerprint_db
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename='account_takeover_model.pkl'):
        """
        Load trained models and configuration from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
            
        Returns:
        --------
        self : AccountTakeoverDetector
            Returns self for method chaining
        """
        model_path = os.path.join(self.model_dir, filename)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.login_behavior_model = model_data['login_behavior_model']
        self.location_anomaly_model = model_data['location_anomaly_model']
        self.feature_importance = model_data['feature_importance']
        self.thresholds = model_data['thresholds']
        self.user_profiles = model_data['user_profiles']
        self.device_fingerprint_db = model_data['device_fingerprint_db']
        
        return self
    
    def extract_login_features(self, login_data, login_history=None, user_profile=None):
        """
        Extract features from login data for ML model prediction.
        
        Parameters:
        -----------
        login_data : pd.Series
            Data about the current login attempt
        login_history : pd.DataFrame
            User's login history
        user_profile : dict
            User's behavioral profile
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted features
        """
        # Convert login_data to Series if it's a dict
        if isinstance(login_data, dict):
            login_data = pd.Series(login_data)
        
        # Initialize features dictionary
        features = {}
        
        # Extract time features
        if 'timestamp' in login_data:
            if isinstance(login_data['timestamp'], str):
                timestamp = pd.to_datetime(login_data['timestamp'])
            else:
                timestamp = login_data['timestamp']
            
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.dayofweek
            features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
            features['is_business_hours'] = 1 if (timestamp.hour >= 9 and timestamp.hour <= 17) else 0
            features['is_night'] = 1 if (timestamp.hour >= 22 or timestamp.hour <= 5) else 0
        
        # Extract device features
        if 'device_id' in login_data:
            features['known_device'] = 1 if (user_profile and login_data['device_id'] in user_profile.get('devices', [])) else 0
        
        # Extract location features
        if 'latitude' in login_data and 'longitude' in login_data and user_profile and 'locations' in user_profile:
            min_distance = float('inf')
            for lat, lon in user_profile['locations']:
                distance = self.calculate_location_distance(
                    login_data['latitude'], login_data['longitude'], lat, lon
                )
                min_distance = min(min_distance, distance)
            
            features['location_distance'] = min_distance
            features['location_anomaly'] = 1 if min_distance > self.thresholds['location_distance_km'] else 0
        
        # Extract IP features
        if 'ip_address' in login_data and user_profile and 'typical_ips' in user_profile:
            features['known_ip'] = 1 if login_data['ip_address'] in user_profile['typical_ips'] else 0
        
        # Extract velocity features
        if login_history is not None and not login_history.empty and 'timestamp' in login_data:
            # Ensure timestamp is datetime
            if isinstance(login_data['timestamp'], str):
                current_time = pd.to_datetime(login_data['timestamp'])
            else:
                current_time = login_data['timestamp']
            
            # Calculate time windows
            one_hour_ago = current_time - pd.Timedelta(hours=1)
            one_day_ago = current_time - pd.Timedelta(days=1)
            
            # Ensure login_history timestamps are datetime
            if not pd.api.types.is_datetime64_any_dtype(login_history['timestamp'].iloc[0]):
                login_history['timestamp'] = pd.to_datetime(login_history['timestamp'])
            
            # Get login attempts in different time windows
            logins_1h = login_history[login_history['timestamp'] >= one_hour_ago]
            logins_24h = login_history[login_history['timestamp'] >= one_day_ago]
            
            features['login_attempts_1h'] = len(logins_1h)
            features['login_attempts_24h'] = len(logins_24h)
            
            # Calculate failed attempts if available
            if 'success' in login_history.columns:
                features['failed_attempts_1h'] = sum(~logins_1h['success'])
                features['failed_attempts_24h'] = sum(~logins_24h['success'])
                features['failed_ratio_1h'] = features['failed_attempts_1h'] / max(1, features['login_attempts_1h'])
                features['failed_ratio_24h'] = features['failed_attempts_24h'] / max(1, features['login_attempts_24h'])
            
            # Calculate unique IPs and devices if available
            if 'ip_address' in login_history.columns:
                features['unique_ips_1h'] = logins_1h['ip_address'].nunique()
                features['unique_ips_24h'] = logins_24h['ip_address'].nunique()
            
            if 'device_id' in login_history.columns:
                features['unique_devices_1h'] = logins_1h['device_id'].nunique()
                features['unique_devices_24h'] = logins_24h['device_id'].nunique()
        
        # Extract password event features
        if 'event_type' in login_data:
            features['is_password_reset'] = 1 if login_data['event_type'] == 'password_reset' else 0
            features['is_password_change'] = 1 if login_data['event_type'] == 'password_change' else 0
            features['is_login_after_reset'] = 1 if login_data['event_type'] == 'login_after_reset' else 0
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
    
    def create_sample_data(self, n_users=50, n_logins=1000, n_devices=100, 
                          suspicious_ratio=0.05, save_to_csv=True):
        """
        Create sample data for demonstration purposes.
        
        Parameters:
        -----------
        n_users : int
            Number of users to generate
        n_logins : int
            Number of login attempts to generate
        n_devices : int
            Number of devices to generate
        suspicious_ratio : float
            Proportion of suspicious login attempts
        save_to_csv : bool
            Whether to save the generated data to CSV files
            
        Returns:
        --------
        tuple
            (logins_df, users_df, devices_df) DataFrames
        """
        np.random.seed(42)
        
        # Generate user data
        user_ids = [f"U{i:05d}" for i in range(1, n_users + 1)]
        
        users_data = {
            'user_id': user_ids,
            'name': [f"User {i}" for i in range(1, n_users + 1)],
            'email': [f"user{i}@example.com" for i in range(1, n_users + 1)],
            'registration_date': pd.date_range(start='2020-01-01', periods=n_users, freq='D'),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France'], n_users),
            'phone': [f"+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_users)]
        }
        
        users_df = pd.DataFrame(users_data)
        
        # Generate device data
        device_ids = [f"D{i:05d}" for i in range(1, n_devices + 1)]
        
        devices_data = {
            'device_id': device_ids,
            'user_id': np.random.choice(user_ids, n_devices),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_devices),
            'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], n_devices),
            'operating_system': np.random.choice(['Windows', 'MacOS', 'iOS', 'Android', 'Linux'], n_devices),
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_devices)]
        }
        
        devices_df = pd.DataFrame(devices_data)
        
        # Generate login data
        login_ids = [f"L{i:06d}" for i in range(1, n_logins + 1)]
        
        # Determine which logins are suspicious
        n_suspicious = int(n_logins * suspicious_ratio)
        is_suspicious = np.zeros(n_logins, dtype=bool)
        is_suspicious[:n_suspicious] = True
        np.random.shuffle(is_suspicious)
        
        # Generate base login data
        logins_data = {
            'login_id': login_ids,
            'user_id': np.random.choice(user_ids, n_logins),
            'timestamp': pd.date_range(start='2023-01-01', periods=n_logins, freq='30min'),
            'device_id': np.random.choice(device_ids, n_logins),
            'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_logins)],
            'success': np.random.choice([True, False], n_logins, p=[0.9, 0.1]),
            'event_type': np.random.choice(['login', 'password_change', 'password_reset', 'login_after_reset'], n_logins, p=[0.85, 0.05, 0.05, 0.05]),
            'is_suspicious': is_suspicious
        }
        
        # Add location data
        # Define some common locations for each user
        user_locations = {}
        for user_id in user_ids:
            # Generate 2-3 common locations for each user
            n_locations = np.random.randint(2, 4)
            base_lat = np.random.uniform(25, 50)  # Random base latitude
            base_lon = np.random.uniform(-120, -70)  # Random base longitude
            
            locations = []
            for _ in range(n_locations):
                # Add some small random variation to create nearby locations
                lat = base_lat + np.random.uniform(-1, 1)
                lon = base_lon + np.random.uniform(-1, 1)
                locations.append((lat, lon))
            
            user_locations[user_id] = locations
        
        # Assign locations to logins
        latitudes = []
        longitudes = []
        
        for i in range(n_logins):
            user_id = logins_data['user_id'][i]
            
            if is_suspicious[i] and np.random.random() < 0.7:
                # For suspicious logins, often use a location far from the user's common locations
                lat = np.random.uniform(0, 70)
                lon = np.random.uniform(-180, 180)
            else:
                # For normal logins, use one of the user's common locations
                if np.random.random() < 0.9:  # 90% chance of using a common location
                    lat, lon = user_locations[user_id][np.random.randint(0, len(user_locations[user_id]))]
                else:
                    # 10% chance of using a slightly different location
                    base_lat, base_lon = user_locations[user_id][np.random.randint(0, len(user_locations[user_id]))]
                    lat = base_lat + np.random.uniform(-0.5, 0.5)
                    lon = base_lon + np.random.uniform(-0.5, 0.5)
            
            latitudes.append(lat)
            longitudes.append(lon)
        
        logins_data['latitude'] = latitudes
        logins_data['longitude'] = longitudes
        
        # Create DataFrame
        logins_df = pd.DataFrame(logins_data)
        
        # Modify suspicious logins to have more suspicious patterns
        for i in range(n_logins):
            if is_suspicious[i]:
                # Increase likelihood of failed login
                if np.random.random() < 0.6:
                    logins_df.at[i, 'success'] = False
                
                # Increase likelihood of unusual event type
                if np.random.random() < 0.4:
                    logins_df.at[i, 'event_type'] = np.random.choice(['password_reset', 'login_after_reset'])
                
                # Increase likelihood of unusual device
                if np.random.random() < 0.7:
                    logins_df.at[i, 'device_id'] = np.random.choice(device_ids)
                
                # Increase likelihood of unusual IP
                if np.random.random() < 0.8:
                    logins_df.at[i, 'ip_address'] = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        # Save to CSV if requested
        if save_to_csv:
            data_dir = os.path.join(os.path.dirname(self.model_dir), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            logins_df.to_csv(os.path.join(data_dir, 'sample_logins.csv'), index=False)
            users_df.to_csv(os.path.join(data_dir, 'sample_users.csv'), index=False)
            devices_df.to_csv(os.path.join(data_dir, 'sample_devices.csv'), index=False)
        
        return logins_df, users_df, devices_df
