"""
Base model class for fraud detection models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import os
from datetime import datetime

from ..config import Config

class BaseFraudModel(ABC):
    """Base class for all fraud detection models."""

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize the base fraud detection model.

        Args:
            name: Name of the model
            version: Version of the model (default: "1.0.0")
        """
        self.name = name
        self.version = version
        self.metadata = {
            'name': name,
            'version': version,
            'created_at': None,
            'last_trained': None,
            'saved_at': None,
            'training_samples': 0,
            'performance_metrics': {},
        }
        self.feature_importance = None
        self.config = Config.get_instance()
        self.model: Optional[BaseEstimator] = None

    @abstractmethod
    def preprocess_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess data before model training or prediction.

        Args:
            data: Input data to preprocess

        Returns:
            Preprocessed data
        """
        pass

    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels
        """
        # Update metadata
        if self.metadata['created_at'] is None:
            self.metadata['created_at'] = datetime.utcnow().isoformat()
        
        self.metadata['last_trained'] = datetime.utcnow().isoformat()
        self.metadata['training_samples'] = len(X)
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features to predict on

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Prediction probabilities
        """
        pass

    def save_model(self) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_dir = os.path.join(self.config.model.model_dir, self.name, self.version)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.name}.joblib")

        # Update metadata
        self.metadata['saved_at'] = datetime.utcnow().isoformat()

        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metadata': self.metadata,
            'version': self.version
        }
        joblib.dump(model_data, model_path)

    def load_model(self, version: Optional[str] = None) -> None:
        """
        Load model from disk.
        
        Args:
            version: Optional specific version to load
        """
        load_version = version or self.version
        model_dir = os.path.join(self.config.model.model_dir, self.name, load_version)
        model_path = os.path.join(model_dir, f"{self.name}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found at {model_path}")

        model_data = joblib.load(model_path)
        
        # Validate version compatibility
        if model_data['version'] != load_version:
            raise ValueError(f"Model version mismatch. Expected {load_version}, got {model_data['version']}")

        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.metadata = model_data['metadata']
        self.version = model_data['version']

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Train a model first.")

        if top_n is not None:
            return self.feature_importance.head(top_n)
        return self.feature_importance

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update model performance metrics.

        Args:
            metrics: Dictionary of metric names and values
        """
        self.metadata['performance_metrics'].update(metrics) 