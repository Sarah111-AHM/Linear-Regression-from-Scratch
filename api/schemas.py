from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np


class PredictionRequest(BaseModel):
    """Input data for prediction"""
    features: List[float] = Field(..., description="Feature values for prediction")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) == 0:
            raise ValueError('Features list cannot be empty')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [3.5, 25.0, 2000.0, 400.0, 1200.0, 450.0, 34.05, -118.24]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch input data for multiple predictions"""
    instances: List[List[float]] = Field(..., description="List of feature vectors")
    
    @validator('instances')
    def validate_instances(cls, v):
        if len(v) == 0:
            raise ValueError('Instances list cannot be empty')
        
        # Check all instances have same length
        first_len = len(v[0])
        for i, instance in enumerate(v):
            if len(instance) != first_len:
                raise ValueError(f'Instance {i} has {len(instance)} features, expected {first_len}')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    [3.5, 25.0, 2000.0, 400.0, 1200.0, 450.0, 34.05, -118.24],
                    [4.2, 15.0, 1800.0, 350.0, 1000.0, 380.0, 34.10, -118.30]
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output"""
    prediction: float = Field(..., description="Predicted value")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    status: str = Field(..., description="Prediction status")


class BatchPredictionResponse(BaseModel):
    """Batch prediction output"""
    predictions: List[float] = Field(..., description="List of predicted values")
    count: int = Field(..., description="Number of predictions")
    status: str = Field(..., description="Prediction status")


class ModelInfoResponse(BaseModel):
    """Model information"""
    model_type: str = Field(..., description="Model type")
    use_scaling: bool = Field(..., description="Whether feature scaling is applied")
    n_features: int = Field(..., description="Number of features expected")
    converged: bool = Field(..., description="Whether model converged during training")
    final_loss: Optional[float] = Field(None, description="Final training loss")
