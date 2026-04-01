"""
FastAPI application for Linear Regression predictions
"""

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uuid
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse
)
from api.dependencies import get_model
from src.model import LinearRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Linear Regression from Scratch",
    description="Production-ready Linear Regression API with pure NumPy implementation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Linear Regression from Scratch API",
        "version": "1.0.0",
        "description": "A production-ready implementation of Linear Regression using only NumPy",
        "endpoints": [
            "/docs - Swagger documentation",
            "/health - Health check",
            "/predict - Single prediction",
            "/predict/batch - Batch predictions",
            "/model/info - Model information"
        ]
    }


@app.get("/health")
async def health_check(model: LinearRegression = Depends(get_model)):
    """
    Health check endpoint
    Verifies that the model is loaded and ready
    """
    try:
        if model.weights is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "error": "Model not loaded"}
            )
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "use_scaling": model.use_scaling,
            "n_features": len(model.weights) if model.weights is not None else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(model: LinearRegression = Depends(get_model)):
    """
    Get information about the loaded model
    """
    return ModelInfoResponse(
        model_type="LinearRegression (from scratch with NumPy)",
        use_scaling=model.use_scaling,
        n_features=len(model.weights) if model.weights is not None else 0,
        converged=model.converged if hasattr(model, 'converged') else False,
        final_loss=model.loss_history[-1] if model.loss_history else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: LinearRegression = Depends(get_model)
):
    """
    Make a single prediction using the trained linear regression model
    
    Mathematical operation: y_pred = X·w + b
    """
    try:
        # Convert features to numpy array
        X = np.array(request.features).reshape(1, -1)
        
        # Validate feature count
        expected_features = len(model.weights)
        if X.shape[1] != expected_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {expected_features} features, got {X.shape[1]}"
            )
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Generate unique ID for the prediction
        prediction_id = str(uuid.uuid4())
        
        logger.info(f"Prediction made: {prediction:.4f} (ID: {prediction_id})")
        
        return PredictionResponse(
            prediction=float(prediction),
            prediction_id=prediction_id,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model: LinearRegression = Depends(get_model)
):
    """
    Make batch predictions for multiple instances
    """
    try:
        # Convert to numpy array
        X = np.array(request.instances)
        
        # Validate feature count for all instances
        expected_features = len(model.weights)
        if X.shape[1] != expected_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {expected_features} features, got {X.shape[1]}"
            )
        
        # Make batch predictions
        predictions = model.predict(X)
        
        logger.info(f"Batch prediction: {len(predictions)} instances processed")
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            count=len(predictions),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Preload model on startup"""
    try:
        get_model()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
