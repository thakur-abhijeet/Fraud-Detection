import pandas as pd
import numpy as np
from train import FraudDetectionModel
import argparse
import os

def ensemble_predict(model, X):
    """Make predictions using both models and combine results"""
    # Get predictions from both models
    xgb_prob = model.xgb_model.predict_proba(X)[:, 1]
    lgb_prob = model.lgb_model.predict_proba(X)[:, 1]
    
    # Combine predictions (simple average)
    ensemble_prob = (xgb_prob + lgb_prob) / 2
    
    # Convert probabilities to binary predictions using 0.5 threshold
    predictions = (ensemble_prob >= 0.5).astype(int)
    
    return predictions, ensemble_prob

def main():
    parser = argparse.ArgumentParser(description='Make fraud predictions')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--model_dir', required=True, help='Directory containing saved models')
    
    args = parser.parse_args()
    
    # Read data
    df = pd.read_csv(args.input)
    
    # Initialize model and load saved models
    model = FraudDetectionModel()
    model.load_models(args.model_dir)
    
    # Preprocess data
    X = model.preprocessor.preprocess(df, is_training=False)
    
    # Make predictions
    predictions, probabilities = ensemble_predict(model, X)
    
    # Add predictions to dataframe
    df['predicted_fraud'] = predictions
    df['fraud_probability'] = probabilities
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    n_fraud = predictions.sum()
    print(f"\nPrediction Summary:")
    print(f"Total transactions: {len(predictions)}")
    print(f"Predicted fraudulent: {n_fraud} ({n_fraud/len(predictions)*100:.2f}%)")

if __name__ == "__main__":
    main()
