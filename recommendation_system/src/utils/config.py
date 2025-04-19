"""
Configuration management for the recommendation system.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """
    Configuration manager for the recommendation system.
    """
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize the configuration manager.
        
        Parameters:
        -----------
        config_dir : str
            Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self._config = {
            'data': {
                'dir': 'data',
                'products_file': 'products.csv',
                'ratings_file': 'ratings.csv',
                'reviews_file': 'reviews.csv',
                'users_file': 'users.csv'
            },
            'model': {
                'dir': 'model',
                'content_based_model': 'content_based_model.pkl',
                'collaborative_model': 'collaborative_filtering_model.pkl',
                'sentiment_model': 'sentiment_model.pkl',
                'hybrid_model': 'hybrid_recommender.pkl'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': True
            },
            'recommendation': {
                'weights': {
                    'content': 0.3,
                    'collaborative': 0.5,
                    'sentiment': 0.2
                },
                'category_weights': {
                    'electronics': {
                        'content': 0.5,
                        'collaborative': 0.3,
                        'sentiment': 0.2
                    },
                    'home_appliances': {
                        'content': 0.5,
                        'collaborative': 0.3,
                        'sentiment': 0.2
                    }
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/recommendation_system.log',
                'max_bytes': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            }
        }
        
        # Load configuration from file if exists
        config_file = self.config_dir / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                self._config.update(json.load(f))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Parameters:
        -----------
        key : str
            Configuration key (dot notation supported)
        default : Any
            Default value if key not found
            
        Returns:
        --------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Parameters:
        -----------
        key : str
            Configuration key (dot notation supported)
        value : Any
            Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self) -> None:
        """
        Save configuration to file.
        """
        config_file = self.config_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self._config, f, indent=4)
    
    def get_data_path(self, file_type: str) -> Path:
        """
        Get path to data file.
        
        Parameters:
        -----------
        file_type : str
            Type of data file ('products', 'ratings', 'reviews', 'users')
            
        Returns:
        --------
        Path
            Path to data file
        """
        data_dir = Path(self.get('data.dir'))
        data_dir.mkdir(exist_ok=True)
        return data_dir / self.get(f'data.{file_type}_file')
    
    def get_model_path(self, model_type: str) -> Path:
        """
        Get path to model file.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('content_based', 'collaborative', 'sentiment', 'hybrid')
            
        Returns:
        --------
        Path
            Path to model file
        """
        model_dir = Path(self.get('model.dir'))
        model_dir.mkdir(exist_ok=True)
        return model_dir / self.get(f'model.{model_type}_model')

# Create global configuration instance
config = Config() 