"""
Linear Regression from Scratch using NumPy
Implements Gradient Descent with Vectorization and Feature Scaling
"""

import numpy as np
import pickle
import json
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegression:
    """
    Linear Regression model implemented from scratch using NumPy.
    
    Mathematical Foundation:
    y = X·w + b
    Loss: MSE = (1/n) * Σ(y_true - y_pred)²
    Gradient: ∇w = (-2/n) * Xᵀ·(y_true - y_pred)
    Gradient: ∇b = (-2/n) * Σ(y_true - y_pred)
    
    Features:
    - Vectorized operations (no explicit loops)
    - Feature scaling (Standardization)
    - Gradient Descent optimization
    - Convergence monitoring
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        n_iterations: int = 1000,
        tolerance: float = 1e-6,
        use_scaling: bool = True,
        verbose: bool = True
    ):
        """
        Initialize Linear Regression model.
        
        Args:
            learning_rate: Step size for gradient descent (α)
            n_iterations: Maximum number of training iterations
            tolerance: Convergence threshold for loss improvement
            use_scaling: Whether to standardize features
            verbose: Print training progress
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.use_scaling = use_scaling
        self.verbose = verbose
        
        # Model parameters
        self.weights = None          # Coefficient vector (w)
        self.bias = None             # Intercept term (b)
        
        # Scaling parameters
        self.mean = None
        self.std = None
        
        # Training history
        self.loss_history = []
        self.converged = False
        
    def _standardize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Standardize features: (X - μ) / σ
        
        Args:
            X: Input features (n_samples, n_features)
            fit: If True, compute and store μ and σ; if False, use stored values
            
        Returns:
            Standardized features
        """
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Prevent division by zero
            self.std[self.std == 0] = 1
            
        # Vectorized standardization
        return (X - self.mean) / self.std
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) loss.
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true: Ground truth values (n_samples,)
            y_pred: Predicted values (n_samples,)
            
        Returns:
            MSE loss value
        """
        n_samples = len(y_true)
        # Vectorized MSE computation
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias using vectorization.
        
        Mathematical derivation:
        L = (1/n) * Σ(y - (X·w + b))²
        ∂L/∂w = (-2/n) * Xᵀ·(y - (X·w + b))
        ∂L/∂b = (-2/n) * Σ(y - (X·w + b))
        
        Args:
            X: Input features (n_samples, n_features)
            y_true: Ground truth values (n_samples,)
            y_pred: Predicted values (n_samples,)
            
        Returns:
            Tuple of (dw, db) gradients
        """
        n_samples = len(y_true)
        
        # Error vector (vectorized)
        error = y_true - y_pred  # Shape: (n_samples,)
        
        # Weight gradient: dw = (-2/n) * Xᵀ·error
        # Using matrix multiplication (vectorized)
        dw = (-2 / n_samples) * np.dot(X.T, error)  # Shape: (n_features,)
        
        # Bias gradient: db = (-2/n) * Σ(error)
        db = (-2 / n_samples) * np.sum(error)  # Scalar
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Train the linear regression model using Gradient Descent.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self: Trained model instance
        """
        # Input validation
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Feature scaling
        if self.use_scaling:
            logger.info("Applying feature scaling (standardization)")
            X_scaled = self._standardize(X, fit=True)
        else:
            X_scaled = X.copy()
        
        # Add bias term conceptually (we'll handle separately)
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        logger.info(f"Starting training: {n_samples} samples, {n_features} features")
        logger.info(f"Learning rate: {self.learning_rate}, Max iterations: {self.n_iterations}")
        
        # Gradient Descent loop
        for iteration in range(self.n_iterations):
            # Forward pass: compute predictions
            # y_pred = X·w + b (vectorized)
            y_pred = np.dot(X_scaled, self.weights) + self.bias
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw, db = self._compute_gradients(X_scaled, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < self.tolerance:
                    self.converged = True
                    logger.info(f"Converged at iteration {iteration} with loss: {loss:.6f}")
                    break
            
            # Log progress
            if self.verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                logger.info(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        if not self.converged:
            logger.warning(f"Reached max iterations ({self.n_iterations}) without convergence")
        
        logger.info(f"Training completed. Final loss: {self.loss_history[-1]:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        y_pred = X·w + b (Matrix multiplication)
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Input validation
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Apply the same scaling used during training
        if self.use_scaling:
            X_scaled = self._standardize(X, fit=False)
        else:
            X_scaled = X.copy()
        
        # Vectorized prediction using matrix multiplication
        return np.dot(X_scaled, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination) score.
        
        R² = 1 - (SS_res / SS_tot)
        SS_res = Σ(y - y_pred)²
        SS_tot = Σ(y - y_mean)²
        
        Args:
            X: Input features
            y: True values
            
        Returns:
            R² score (1 is perfect, 0 is baseline)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Handle division by zero
        if ss_tot == 0:
            return 1.0
        
        return 1 - (ss_res / ss_tot)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters and metadata.
        
        Returns:
            Dictionary containing weights, bias, and training metadata
        """
        return {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'loss_history': self.loss_history,
            'converged': self.converged,
            'use_scaling': self.use_scaling
        }
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'mean': self.mean,
            'std': self.std,
            'use_scaling': self.use_scaling,
            'loss_history': self.loss_history,
            'converged': self.converged
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.mean = model_data['mean']
        self.std = model_data['std']
        self.use_scaling = model_data['use_scaling']
        self.loss_history = model_data['loss_history']
        self.converged = model_data['converged']
        
        logger.info(f"Model loaded from {filepath}")
