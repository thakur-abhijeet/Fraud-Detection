"""
Tests for credit card fraud detection model.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_model_initialization(credit_card_model):
    """Test model initialization."""
    assert credit_card_model is not None
    assert credit_card_model.model is None
    assert credit_card_model.feature_importance is None

def test_preprocess_data(credit_card_model, sample_transaction):
    """Test data preprocessing."""
    features = credit_card_model.preprocess_data(sample_transaction)
    
    # Check required features are present
    for feature in credit_card_model.required_features:
        assert feature in features.columns

    # Check data types
    assert features['amount'].dtype == np.float64
    assert features['card_present'].dtype == np.int64
    assert features['international'].dtype == np.int64

def test_train_model(credit_card_model):
    """Test model training."""
    # Create sample training data
    X = pd.DataFrame({
        'amount': np.random.uniform(10, 1000, 1000),
        'card_present': np.random.choice([0, 1], 1000),
        'international': np.random.choice([0, 1], 1000),
        'weekend': np.random.choice([0, 1], 1000),
        'night_tx': np.random.choice([0, 1], 1000),
        'high_risk_country': np.random.choice([0, 1], 1000),
        'address_mismatch': np.random.choice([0, 1], 1000)
    })
    
    # Create labels (5% fraud rate)
    y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    
    # Train model
    credit_card_model.train(X, pd.Series(y))
    
    assert credit_card_model.model is not None
    assert credit_card_model.feature_importance is not None
    assert len(credit_card_model.feature_importance) == len(X.columns)

def test_predict(credit_card_model):
    """Test model predictions."""
    # Create sample test data
    X_test = pd.DataFrame({
        'amount': [100.0, 1000.0, 50.0],
        'card_present': [1, 0, 1],
        'international': [0, 1, 0],
        'weekend': [0, 1, 0],
        'night_tx': [0, 1, 0],
        'high_risk_country': [0, 1, 0],
        'address_mismatch': [0, 1, 0]
    })
    
    # Train model if not trained
    if credit_card_model.model is None:
        X = pd.DataFrame(np.random.randn(1000, 7), columns=X_test.columns)
        y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        credit_card_model.train(X, pd.Series(y))
    
    # Get predictions
    predictions = credit_card_model.predict(X_test)
    probabilities = credit_card_model.predict_proba(X_test)
    
    assert len(predictions) == len(X_test)
    assert len(probabilities) == len(X_test)
    assert all(isinstance(p, (np.int64, int)) for p in predictions)
    assert all(0 <= p <= 1 for p in probabilities)

def test_analyze_transaction(credit_card_model, sample_transaction):
    """Test transaction analysis."""
    # Train model if not trained
    if credit_card_model.model is None:
        X = pd.DataFrame(np.random.randn(1000, 7), columns=credit_card_model.required_features[:7])
        y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        credit_card_model.train(X, pd.Series(y))
    
    # Analyze transaction
    results = credit_card_model.analyze_transaction(sample_transaction)
    
    # Check required fields
    assert 'transaction_id' in results
    assert 'fraud_probability' in results
    assert 'risk_level' in results
    assert 'recommended_action' in results
    assert 'risk_factors' in results
    assert 'timestamp' in results
    
    # Check value types and ranges
    assert isinstance(results['fraud_probability'], float)
    assert 0 <= results['fraud_probability'] <= 1
    assert results['risk_level'] in ['low', 'medium', 'high']
    assert results['recommended_action'] in ['allow', 'review', 'block']
    assert isinstance(results['risk_factors'], list)

def test_identify_risk_factors(credit_card_model):
    """Test risk factor identification."""
    # Create high-risk transaction features
    features = pd.Series({
        'amount': 1000.0,
        'card_present': 0,
        'international': 1,
        'weekend': 1,
        'night_tx': 1,
        'high_risk_country': 1,
        'address_mismatch': 1
    })
    
    # Train model if not trained
    if credit_card_model.model is None:
        X = pd.DataFrame(np.random.randn(1000, 7), columns=features.index)
        y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        credit_card_model.train(X, pd.Series(y))
    
    # Get risk factors
    risk_factors = credit_card_model._identify_risk_factors(features, 0.8)
    
    assert isinstance(risk_factors, list)
    assert len(risk_factors) > 0
    
    for factor in risk_factors:
        assert 'factor' in factor
        assert 'impact' in factor
        assert 'description' in factor
        assert isinstance(factor['impact'], float)
        assert 0 <= factor['impact'] <= 1

def test_calculate_feature_impacts(credit_card_model):
    """Test feature impact calculation."""
    # Create sample features
    features = pd.Series({
        'amount': 1000.0,
        'card_present': 0,
        'international': 1,
        'weekend': 1,
        'night_tx': 1,
        'high_risk_country': 1,
        'address_mismatch': 1
    })
    
    # Train model if not trained
    if credit_card_model.model is None:
        X = pd.DataFrame(np.random.randn(1000, 7), columns=features.index)
        y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        credit_card_model.train(X, pd.Series(y))
    
    # Calculate impacts
    impacts = credit_card_model._calculate_feature_impacts(features)
    
    assert isinstance(impacts, dict)
    assert len(impacts) == len(features)
    assert all(isinstance(v, float) for v in impacts.values())
    assert all(0 <= v <= 1 for v in impacts.values())

def test_model_persistence(credit_card_model, tmp_path):
    """Test model saving and loading."""
    # Create and train model
    X = pd.DataFrame(np.random.randn(1000, 7), columns=credit_card_model.required_features[:7])
    y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    credit_card_model.train(X, pd.Series(y))
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    credit_card_model.save_model(str(model_path))
    
    # Create new model instance
    new_model = type(credit_card_model)()
    
    # Load saved model
    new_model.load_model(str(model_path))
    
    # Compare predictions
    X_test = pd.DataFrame(np.random.randn(10, 7), columns=X.columns)
    original_preds = credit_card_model.predict(X_test)
    loaded_preds = new_model.predict(X_test)
    
    np.testing.assert_array_equal(original_preds, loaded_preds)

def test_get_feature_importance(credit_card_model):
    """Test feature importance retrieval."""
    # Train model if not trained
    if credit_card_model.model is None:
        X = pd.DataFrame(np.random.randn(1000, 7), columns=credit_card_model.required_features[:7])
        y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        credit_card_model.train(X, pd.Series(y))
    
    # Get feature importance
    importance = credit_card_model.get_feature_importance()
    top_features = credit_card_model.get_feature_importance(top_n=3)
    
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 7  # Number of features
    assert len(top_features) == 3
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert all(importance['importance'] >= 0)
    assert importance['importance'].sum() == pytest.approx(1.0) 