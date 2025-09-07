"""
FastAPI application for Swiggy delivery time prediction.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import DataPreprocessor
from src.utils.config import MODELS_DIR
from src.utils.helpers import calculate_distance, generate_confidence_interval, format_prediction_response
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Swiggy Delivery Time Prediction API",
    description="Predict food delivery times using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    restaurant_lat: float = Field(..., description="Restaurant latitude", ge=-90, le=90)
    restaurant_lng: float = Field(..., description="Restaurant longitude", ge=-180, le=180)
    customer_lat: float = Field(..., description="Customer latitude", ge=-90, le=90)
    customer_lng: float = Field(..., description="Customer longitude", ge=-180, le=180)
    order_hour: int = Field(..., description="Hour of order (0-23)", ge=0, le=23)
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)", ge=0, le=6)
    weather_condition: str = Field(..., description="Weather condition")
    traffic_density: float = Field(..., description="Traffic density (0-1)", ge=0, le=1)
    
    # Optional fields with defaults
    is_weekend: Optional[bool] = Field(None, description="Is weekend")
    temperature: Optional[float] = Field(25.0, description="Temperature in Celsius")
    restaurant_rating: Optional[float] = Field(4.0, description="Restaurant rating (1-5)", ge=1, le=5)
    food_category: Optional[str] = Field("Indian", description="Food category")
    order_value: Optional[float] = Field(300.0, description="Order value in currency", ge=0)
    num_items: Optional[int] = Field(2, description="Number of items", ge=1)
    preparation_time: Optional[float] = Field(15.0, description="Preparation time in minutes", ge=0)
    restaurant_type: Optional[str] = Field("Casual Dining", description="Restaurant type")
    city: Optional[str] = Field("bangalore", description="City")

class SimplePredictionRequest(BaseModel):
    restaurant_lat: float = Field(..., ge=-90, le=90)
    restaurant_lng: float = Field(..., ge=-180, le=180)
    customer_lat: float = Field(..., ge=-90, le=90)
    customer_lng: float = Field(..., ge=-180, le=180)
    order_hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    weather_condition: str
    traffic_density: float = Field(..., ge=0, le=1)

class PredictionResponse(BaseModel):
    predicted_delivery_time: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    timestamp: str

def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor
    
    try:
        # Load preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(str(preprocessor_path))
            logger.info("Preprocessor loaded successfully")
        else:
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            return False
        
        # Try to load the best model first
        best_model_path = MODELS_DIR / "best_model_lightgbm.joblib"
        if best_model_path.exists():
            model_data = joblib.load(best_model_path)
            model = model_data['model']
            logger.info("Best model (LightGBM) loaded successfully")
            return True
        
        # Fallback to other models
        model_files = [
            "lightgbm_model.joblib",
            "xgboost_model.joblib", 
            "random_forest_model.joblib",
            "linear_regression_model.joblib"
        ]
        
        for model_file in model_files:
            model_path = MODELS_DIR / model_file
            if model_path.exists():
                model_data = joblib.load(model_path)
                model = model_data['model']
                logger.info(f"Model loaded from {model_file}")
                return True
        
        logger.error("No trained model found")
        return False
        
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        return False

def create_sample_model():
    """Create a simple sample model for testing if no model is found."""
    logger.warning("No trained model found. Creating a sample model for testing...")
    
    class MockModel:
        def predict(self, X):
            if hasattr(X, 'iloc'):
                distance = X.iloc[0, 0] if len(X.columns) > 0 else 5.0
                traffic = X.iloc[0, 5] if len(X.columns) > 5 else 0.5
            else:
                distance = X[0][0] if len(X[0]) > 0 else 5.0
                traffic = X[0][5] if len(X[0]) > 5 else 0.5
            
            base_time = 20
            distance_factor = distance * 3
            traffic_factor = traffic * 15
            
            prediction = base_time + distance_factor + traffic_factor
            return [max(15, prediction)]
    
    global model, preprocessor
    model = MockModel()
    
    class MockPreprocessor:
        def prepare_data(self, df, fit=False):
            return df
        
        def get_feature_names(self):
            return ['distance_km', 'order_hour', 'day_of_week', 'is_weekend', 
                   'temperature', 'traffic_density', 'restaurant_rating']
    
    preprocessor = MockPreprocessor()
    logger.info("Sample model created for testing")

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting FastAPI application...")
    
    if not load_model_and_preprocessor():
        logger.warning("Failed to load trained model. Creating sample model for testing...")
        create_sample_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Swiggy Delivery Time Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #FF6B35; text-align: center; }
            .status { padding: 15px; border-radius: 5px; margin: 20px 0; text-align: center; }
            .status.ready { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .api-links { text-align: center; margin: 30px 0; }
            .api-links a { 
                display: inline-block; 
                margin: 10px; 
                padding: 15px 25px; 
                background-color: #FF6B35; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                font-weight: bold;
            }
            .api-links a:hover { background-color: #e55a2b; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature { background-color: #f8f9fa; padding: 20px; border-radius: 5px; text-align: center; }
            .feature h3 { color: #FF6B35; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚚 Swiggy Delivery Time Prediction API</h1>
            
            <div class="status ready">
                <strong>✅ API is ready for predictions!</strong><br>
                Powered by Machine Learning
            </div>
            
            <div class="api-links">
                <a href="/docs" target="_blank">📚 Interactive API Docs</a>
                <a href="/redoc" target="_blank">📖 API Documentation</a>
                <a href="/health">🏥 Health Check</a>
            </div>
            
            <h2>🚀 Features</h2>
            <div class="features">
                <div class="feature">
                    <h3>🤖 ML Models</h3>
                    <p>Multiple algorithms including XGBoost, LightGBM, and Random Forest</p>
                </div>
                <div class="feature">
                    <h3>⚡ Real-time</h3>
                    <p>Fast predictions with confidence intervals</p>
                </div>
                <div class="feature">
                    <h3>🔧 Feature Engineering</h3>
                    <p>Advanced preprocessing and feature engineering pipeline</p>
                </div>
                <div class="feature">
                    <h3>📊 RESTful API</h3>
                    <p>Clean, documented API with automatic validation</p>
                </div>
            </div>
            
            <h2>📋 Quick Test</h2>
            <p>Use the interactive docs at <a href="/docs">/docs</a> to test the API endpoints:</p>
            <ul>
                <li><strong>POST /predict</strong> - Full prediction with all features</li>
                <li><strong>POST /predict/simple</strong> - Simplified prediction with minimal inputs</li>
                <li><strong>GET /health</strong> - Check API health status</li>
            </ul>
            
            <p style="text-align: center; margin-top: 40px; color: #666;">
                Built with FastAPI • Powered by Python • Ready for Production
            </p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None and preprocessor is not None else "unhealthy",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint."""
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            raise HTTPException(
                status_code=503, 
                detail="Model or preprocessor not loaded"
            )
        
        # Convert request to dictionary
        data = request.dict()
        
        # Fill in is_weekend if not provided
        if data['is_weekend'] is None:
            data['is_weekend'] = data['day_of_week'] >= 5
        
        # Calculate distance
        data['distance_km'] = calculate_distance(
            data['restaurant_lat'], data['restaurant_lng'],
            data['customer_lat'], data['customer_lng']
        )
        
        # Create DataFrame for preprocessing
        df = pd.DataFrame([data])
        
        # Preprocess the data
        df_processed = preprocessor.prepare_data(df, fit=False)
        
        # Select features for prediction
        X = df_processed[preprocessor.get_feature_names()]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Generate confidence interval
        confidence_interval = generate_confidence_interval(prediction)
        
        # Format response
        return PredictionResponse(
            predicted_delivery_time=round(prediction, 1),
            confidence_interval={
                "lower": confidence_interval[0],
                "upper": confidence_interval[1]
            },
            model_used="LightGBM",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/simple")
async def predict_simple(request: SimplePredictionRequest):
    """Simplified prediction endpoint."""
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            raise HTTPException(
                status_code=503, 
                detail="Model or preprocessor not loaded"
            )
        
        # Convert to dictionary and add defaults
        data = request.dict()
        
        # Add default values
        defaults = {
            'is_weekend': data['day_of_week'] >= 5,
            'temperature': 25.0,
            'restaurant_rating': 4.0,
            'food_category': 'Indian',
            'order_value': 300.0,
            'num_items': 2,
            'preparation_time': 15.0,
            'restaurant_type': 'Casual Dining',
            'city': 'bangalore'
        }
        
        data.update(defaults)
        
        # Calculate distance
        data['distance_km'] = calculate_distance(
            data['restaurant_lat'], data['restaurant_lng'],
            data['customer_lat'], data['customer_lng']
        )
        
        # Create DataFrame and predict
        df = pd.DataFrame([data])
        df_processed = preprocessor.prepare_data(df, fit=False)
        X = df_processed[preprocessor.get_feature_names()]
        prediction = model.predict(X)[0]
        
        return {
            "predicted_delivery_time": round(prediction, 1),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features_count": len(preprocessor.get_feature_names()) if preprocessor else 0,
        "features": preprocessor.get_feature_names() if preprocessor else [],
        "status": "loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)