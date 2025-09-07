"""
Helper functions for the Swiggy delivery prediction project.
"""
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the distance between two coordinates using geopy.
    
    Args:
        lat1, lng1: Restaurant coordinates
        lat2, lng2: Customer coordinates
    
    Returns:
        float: Distance in kilometers
    """
    try:
        restaurant_coords = (lat1, lng1)
        customer_coords = (lat2, lng2)
        distance = geodesic(restaurant_coords, customer_coords).kilometers
        return round(distance, 2)
    except Exception:
        # Fallback to Euclidean distance if geopy fails
        return np.sqrt((lat2 - lat1)**2 + (lng2 - lng1)**2) * 111  # Approximate km per degree

def get_time_features(timestamp: datetime) -> Dict[str, Any]:
    """
    Extract time-based features from a timestamp.
    
    Args:
        timestamp: DateTime object
    
    Returns:
        dict: Time features
    """
    return {
        "hour": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "is_weekend": timestamp.weekday() >= 5,
        "is_peak_hour": timestamp.hour in [12, 13, 19, 20, 21],
        "is_lunch": 11 <= timestamp.hour <= 14,
        "is_dinner": 18 <= timestamp.hour <= 22
    }

def encode_categorical_features(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical features using label encoding.
    
    Args:
        data: DataFrame with categorical features
        categorical_columns: List of categorical column names
    
    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    data_copy = data.copy()
    
    for col in categorical_columns:
        if col in data_copy.columns:
            # Simple label encoding
            unique_values = data_copy[col].unique()
            encoding_map = {val: idx for idx, val in enumerate(unique_values)}
            data_copy[f"{col}_encoded"] = data_copy[col].map(encoding_map)
    
    return data_copy

def save_model(model: Any, model_name: str, model_dir: Path) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        model_dir: Directory to save the model
    
    Returns:
        str: Path to saved model
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}_model.joblib"
    joblib.dump(model, model_path)
    return str(model_path)

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model object
    """
    return joblib.load(model_path)

def save_preprocessing_pipeline(pipeline: Any, pipeline_name: str, model_dir: Path) -> str:
    """
    Save preprocessing pipeline to disk.
    
    Args:
        pipeline: Preprocessing pipeline
        pipeline_name: Name of the pipeline
        model_dir: Directory to save the pipeline
    
    Returns:
        str: Path to saved pipeline
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = model_dir / f"{pipeline_name}_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    return str(pipeline_path)

def validate_input_features(features: Dict[str, Any], required_features: List[str]) -> Dict[str, Any]:
    """
    Validate and clean input features for prediction.
    
    Args:
        features: Input features dictionary
        required_features: List of required feature names
    
    Returns:
        dict: Validated features
    
    Raises:
        ValueError: If required features are missing
    """
    # Check for required features
    missing_features = [f for f in required_features if f not in features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Basic validation
    validated_features = {}
    for feature, value in features.items():
        if feature in required_features:
            # Convert to appropriate type
            if isinstance(value, (int, float)):
                validated_features[feature] = float(value)
            else:
                validated_features[feature] = value
    
    return validated_features

def generate_confidence_interval(prediction: float, std_error: float = 5.0, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Generate confidence interval for prediction.
    
    Args:
        prediction: Predicted value
        std_error: Standard error estimate
        confidence: Confidence level (0.95 for 95%)
    
    Returns:
        tuple: Lower and upper bounds of confidence interval
    """
    # Simple approximation using normal distribution
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
    margin = z_score * std_error
    
    lower_bound = max(0, prediction - margin)  # Delivery time can't be negative
    upper_bound = prediction + margin
    
    return round(lower_bound, 1), round(upper_bound, 1)

def calculate_peak_hour_multiplier(hour: int) -> float:
    """
    Calculate traffic multiplier based on hour of day.
    
    Args:
        hour: Hour of day (0-23)
    
    Returns:
        float: Traffic multiplier
    """
    # Peak hours: lunch (12-14) and dinner (19-21)
    if hour in [12, 13, 19, 20, 21]:
        return 1.3
    elif hour in [11, 14, 18, 22]:
        return 1.1
    elif 6 <= hour <= 10:  # Morning rush
        return 1.2
    else:
        return 1.0

def format_prediction_response(prediction: float, model_name: str, confidence_interval: Tuple[float, float] = None) -> Dict[str, Any]:
    """
    Format prediction response for API.
    
    Args:
        prediction: Predicted delivery time
        model_name: Name of the model used
        confidence_interval: Optional confidence interval
    
    Returns:
        dict: Formatted response
    """
    response = {
        "predicted_delivery_time": round(prediction, 1),
        "model_used": model_name,
        "timestamp": datetime.now().isoformat()
    }
    
    if confidence_interval:
        response["confidence_interval"] = {
            "lower": confidence_interval[0],
            "upper": confidence_interval[1]
        }
    
    return response