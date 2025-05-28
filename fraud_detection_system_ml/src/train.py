import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from preprocess import DataPreprocessor
import joblib
import os
import argparse

class FraudDetectionModel:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.preprocessor = DataPreprocessor()
    
    def train(self, X, y):
        """Train both models"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Train LightGBM
        print("Training LightGBM model...")
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Evaluate models
        self._evaluate_models(X_val, y_val)
    
    def _evaluate_models(self, X_val, y_val):
        """Evaluate both models"""
        # XGBoost predictions
        xgb_pred = self.xgb_model.predict(X_val)
        xgb_prob = self.xgb_model.predict_proba(X_val)[:, 1]
        
        # LightGBM predictions
        lgb_pred = self.lgb_model.predict(X_val)
        lgb_prob = self.lgb_model.predict_proba(X_val)[:, 1]
        
        # Print results
        print("\nXGBoost Results:")
        print(classification_report(y_val, xgb_pred))
        print(f"ROC AUC: {roc_auc_score(y_val, xgb_prob):.4f}")
        
        print("\nLightGBM Results:")
        print(classification_report(y_val, lgb_pred))
        print(f"ROC AUC: {roc_auc_score(y_val, lgb_prob):.4f}")
    
    def save_models(self, path):
        """Save models and preprocessor"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.xgb_model, os.path.join(path, 'xgb_model.joblib'))
        joblib.dump(self.lgb_model, os.path.join(path, 'lgb_model.joblib'))
        self.preprocessor.save_preprocessor(os.path.join(path, 'preprocessor.joblib'))
    
    def load_models(self, path):
        """Load models and preprocessor"""
        self.xgb_model = joblib.load(os.path.join(path, 'xgb_model.joblib'))
        self.lgb_model = joblib.load(os.path.join(path, 'lgb_model.joblib'))
        self.preprocessor.load_preprocessor(os.path.join(path, 'preprocessor.joblib'))

def main():
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--input', required=True, help='Input preprocessed CSV file path')
    parser.add_argument('--output', required=True, help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Read data
    df = pd.read_csv(args.input)
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Initialize and train model
    model = FraudDetectionModel()
    model.train(X, y)
    
    # Save models
    model.save_models(args.output)
    print(f"Models saved to {args.output}")

if __name__ == "__main__":
    main()
