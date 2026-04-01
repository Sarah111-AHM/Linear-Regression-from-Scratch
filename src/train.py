"""
Training script for Linear Regression model
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
from model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_california_housing():
    """Load and prepare California housing dataset"""
    logger.info("Loading California housing dataset...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {housing.feature_names}")
    
    return X, y


def train_model():
    """Train the linear regression model"""
    
    # Load data
    X, y = load_california_housing()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Test samples: {X_test.shape[0]}")
    
    # Create and train model
    model = LinearRegression(
        learning_rate=0.01,
        n_iterations=1000,
        tolerance=1e-6,
        use_scaling=True,
        verbose=True
    )
    
    logger.info("Starting training...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = model._compute_loss(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test R² Score: {r2:.6f}")
    
    # Save model
    model_path = Path("models/linear_regression.pkl")
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    
    # Save training metadata
    metadata = {
        'n_features': X.shape[1],
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0],
        'test_mse': mse,
        'test_r2': r2,
        'learning_rate': model.learning_rate,
        'n_iterations': model.n_iterations,
        'converged': model.converged
    }
    
    import json
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Model saved successfully")
    
    return model


if __name__ == "__main__":
    train_model()
  
