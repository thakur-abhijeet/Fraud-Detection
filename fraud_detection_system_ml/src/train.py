import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
from preprocess import DataPreprocessor
import joblib
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Evaluate models and create visualizations
        self._evaluate_models(X_val, y_val)
        self._create_visualizations(X_val, y_val)
    
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
    
    def _create_visualizations(self, X_val, y_val):
        """Create visualizations for model evaluation"""
        # Create output directory for plots
        os.makedirs('plots', exist_ok=True)
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 6))
        
        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'feature': X_val.columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=xgb_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Important Features (XGBoost)')
        
        # LightGBM feature importance
        lgb_importance = pd.DataFrame({
            'feature': X_val.columns,
            'importance': self.lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=lgb_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Important Features (LightGBM)')
        
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 6))
        
        # Calculate ROC curves
        xgb_fpr, xgb_tpr, _ = roc_curve(y_val, self.xgb_model.predict_proba(X_val)[:, 1])
        lgb_fpr, lgb_tpr, _ = roc_curve(y_val, self.lgb_model.predict_proba(X_val)[:, 1])
        
        # Plot ROC curves
        plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {roc_auc_score(y_val, self.xgb_model.predict_proba(X_val)[:, 1]):.3f})')
        plt.plot(lgb_fpr, lgb_tpr, label=f'LightGBM (AUC = {roc_auc_score(y_val, self.lgb_model.predict_proba(X_val)[:, 1]):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig('plots/roc_curves.png')
        plt.close()
        
        # 3. Confusion Matrices
        plt.figure(figsize=(12, 5))
        
        # XGBoost confusion matrix
        plt.subplot(1, 2, 1)
        cm_xgb = confusion_matrix(y_val, self.xgb_model.predict(X_val))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
        plt.title('XGBoost Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # LightGBM confusion matrix
        plt.subplot(1, 2, 2)
        cm_lgb = confusion_matrix(y_val, self.lgb_model.predict(X_val))
        sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues')
        plt.title('LightGBM Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png')
        plt.close()
        
        print("\nVisualizations saved in 'plots' directory:")
        print("- feature_importance.png: Top 10 important features for each model")
        print("- roc_curves.png: ROC curves comparing model performance")
        print("- confusion_matrices.png: Confusion matrices for both models")
    
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
