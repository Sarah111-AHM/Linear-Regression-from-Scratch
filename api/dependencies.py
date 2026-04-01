from functools import lru_cache
import logging
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.model import LinearRegression

logger = logging.getLogger(__name__)

# Global model instance
_model = None


def load_model(model_path: str = "models/linear_regression.pkl") -> LinearRegression:
    """
    Load the trained model from disk.
    Uses LRU cache to ensure single loading.
    """
    global _model
    
    if _model is None:
        logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        _model = LinearRegression()
        _model.load(model_path)
        
        logger.info("Model loaded successfully")
    
    return _model


def get_model() -> LinearRegression:
    """Dependency to get model instance"""
    return load_model()


class ModelManager:
    """Singleton pattern for model management"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def model(self) -> LinearRegression:
        if self._model is None:
            self._model = load_model()
        return self._model
      
