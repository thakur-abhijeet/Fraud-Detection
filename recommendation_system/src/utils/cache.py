"""
Caching utilities for the recommendation system.
"""

import pickle
from functools import wraps
from typing import Any, Callable, Optional
from pathlib import Path
import hashlib
import time

class Cache:
    """
    Cache manager for the recommendation system.
    """
    
    def __init__(self, cache_dir: str = 'cache', ttl: int = 3600):
        """
        Initialize the cache manager.
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cache files
        ttl : int
            Time to live for cache entries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get path to cache file.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        Path
            Path to cache file
        """
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        Optional[Any]
            Cached value if exists and not expired, None otherwise
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                timestamp, value = pickle.load(f)
                
            # Check if cache entry has expired
            if time.time() - timestamp > self.ttl:
                cache_path.unlink()
                return None
                
            return value
        except (pickle.PickleError, EOFError):
            cache_path.unlink()
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Parameters:
        -----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((time.time(), value), f)
        except (pickle.PickleError, IOError):
            if cache_path.exists():
                cache_path.unlink()
    
    def delete(self, key: str) -> None:
        """
        Delete value from cache.
        
        Parameters:
        -----------
        key : str
            Cache key
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        for cache_file in self.cache_dir.glob('*.pkl'):
            cache_file.unlink()

def cached(cache: Cache, key_prefix: str = ''):
    """
    Decorator to cache function results.
    
    Parameters:
    -----------
    cache : Cache
        Cache instance to use
    key_prefix : str
        Prefix for cache keys
        
    Returns:
    --------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{key_prefix}{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get result from cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
            
        return wrapper
    return decorator

# Create global cache instance
cache = Cache() 