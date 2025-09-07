"""
Flask API for Swiggy delivery time prediction.
"""
from flask import Flask, request, jsonify, render_template_string
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
from src.utils.config import MODELS_DIR, API_HOST, API_PORT, DEBUG
from src.utils.helpers import calculate_distance, validate_input_features, generate_confidence_interval, format_prediction_response
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None

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
            model = model_data['model']  # Extract the actual model
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
                model = model_data['model']  # Extract the actual model
                logger.info(f"Model loaded from {model_file}")
                return True
        
        logger.error("No trained model found")
        return False
        
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        return False

@app.route('/')
def home():
    """Home page with API documentation."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Swiggy Delivery Time Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background-color: #f4f4f4; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .method { color: #fff; padding: 5px 10px; border-radius: 3px; }
            .get { background-color: #4CAF50; }
            .post { background-color: #2196F3; }
            pre { background-color: #f9f9f9; padding: 15px; border-radius: 3px; overflow-x: auto; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.ready { background-color: #d4edda; color: #155724; }
            .status.error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚚 Swiggy Delivery Time Prediction API</h1>
            
            <div class="status {{ status_class }}">
                <strong>Status:</strong> {{ status_message }}
            </div>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /health</h3>
                <p>Check API health status</p>
                <pre>GET {{ base_url }}/health</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict</h3>
                <p>Predict delivery time for an order</p>
                <pre>POST {{ base_url }}/predict
Content-Type: application/json

{
    "restaurant_lat": 12.9716,
    "restaurant_lng": 77.5946,
    "customer_lat": 12.9716,
    "customer_lng": 77.5946,
    "order_hour": 14,
    "day_of_week": 1,
    "is_weekend": false,
    "weather_condition": "Clear",
    "temperature": 25.0,
    "traffic_density": 0.6,
    "restaurant_rating": 4.2,
    "food_category": "Indian",
    "order_value": 450.0,
    "num_items": 3,
    "preparation_time": 15.0,
    "restaurant_type": "Casual Dining",
    "city": "bangalore"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict/simple</h3>
                <p>Simplified prediction with minimal required fields</p>
                <pre>POST {{ base_url }}/predict/simple
Content-Type: application/json

{
    "restaurant_lat": 12.9716,
    "restaurant_lng": 77.5946,
    "customer_lat": 12.9716,
    "customer_lng": 77.5946,
    "order_hour": 14,
    "day_of_week": 1,
    "weather_condition": "Clear",
    "traffic_density": 0.6
}</pre>
            </div>
            
            <h2>Response Format</h2>
            <pre>{
    "predicted_delivery_time": 35.5,
    "confidence_interval": {
        "lower": 30.2,
        "upper": 40.8
    },
    "model_used": "LightGBM",
    "timestamp": "2025-09-06T18:30:00"
}</pre>
            
            <h2>Features</h2>
            <ul>
                <li>Real-time delivery time prediction</li>
                <li>Multiple ML models (Linear, Random Forest, XGBoost, LightGBM)</li>
                <li>Confidence intervals for predictions</li>
                <li>Feature engineering and preprocessing</li>
                <li>RESTful API design</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Check if model is loaded
    if model is not None and preprocessor is not None:
        status_class = "ready"
        status_message = "API is ready for predictions"
    else:
        status_class = "error"
        status_message = "Model not loaded. Please check server logs."
    
    base_url = f"http://{API_HOST}:{API_PORT}"
    
    return render_template_string(
        html_template,
        status_class=status_class,
        status_message=status_message,
        base_url=base_url
    )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    health_status = {
        "status": "healthy" if model is not None and preprocessor is not None else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            return jsonify({
                "error": "Model or preprocessor not loaded",
                "message": "Please check server configuration"
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Required features for prediction
        required_features = [
            'restaurant_lat', 'restaurant_lng', 'customer_lat', 'customer_lng',
            'order_hour', 'day_of_week', 'weather_condition', 'traffic_density'
        ]
        
        # Validate required features
        try:
            validated_data = validate_input_features(data, required_features)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Add default values for optional features
        defaults = {
            'is_weekend': validated_data.get('day_of_week', 0) >= 5,
            'temperature': validated_data.get('temperature', 25.0),
            'restaurant_rating': validated_data.get('restaurant_rating', 4.0),
            'food_category': validated_data.get('food_category', 'Indian'),
            'order_value': validated_data.get('order_value', 300.0),
            'num_items': validated_data.get('num_items', 2),
            'preparation_time': validated_data.get('preparation_time', 15.0),
            'restaurant_type': validated_data.get('restaurant_type', 'Casual Dining'),
            'city': validated_data.get('city', 'bangalore')
        }
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in validated_data:
                validated_data[key] = default_value
        
        # Calculate distance
        validated_data['distance_km'] = calculate_distance(
            validated_data['restaurant_lat'], validated_data['restaurant_lng'],
            validated_data['customer_lat'], validated_data['customer_lng']
        )
        
        # Create DataFrame for preprocessing
        df = pd.DataFrame([validated_data])
        
        # Preprocess the data
        df_processed = preprocessor.prepare_data(df, fit=False)
        
        # Select features for prediction
        X = df_processed[preprocessor.get_feature_names()]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Generate confidence interval
        confidence_interval = generate_confidence_interval(prediction)
        
        # Format response
        response = format_prediction_response(
            prediction, 
            "LightGBM",  # Default model name
            confidence_interval
        )
        
        logger.info(f"Prediction made: {prediction:.2f} minutes")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """Simplified prediction endpoint with minimal required fields."""
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            return jsonify({
                "error": "Model or preprocessor not loaded"
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Minimal required features
        required_features = [
            'restaurant_lat', 'restaurant_lng', 'customer_lat', 'customer_lng',
            'order_hour', 'day_of_week', 'weather_condition', 'traffic_density'
        ]
        
        # Validate and fill with defaults
        simplified_data = {}
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required field: {feature}"}), 400
            simplified_data[feature] = data[feature]
        
        # Add sensible defaults for all other features
        defaults = {
            'is_weekend': simplified_data.get('day_of_week', 0) >= 5,
            'temperature': 25.0,
            'restaurant_rating': 4.0,
            'food_category': 'Indian',
            'order_value': 300.0,
            'num_items': 2,
            'preparation_time': 15.0,
            'restaurant_type': 'Casual Dining',
            'city': 'bangalore'
        }
        
        simplified_data.update(defaults)
        
        # Calculate distance
        simplified_data['distance_km'] = calculate_distance(
            simplified_data['restaurant_lat'], simplified_data['restaurant_lng'],
            simplified_data['customer_lat'], simplified_data['customer_lng']
        )
        
        # Create DataFrame and predict
        df = pd.DataFrame([simplified_data])
        df_processed = preprocessor.prepare_data(df, fit=False)
        X = df_processed[preprocessor.get_feature_names()]
        prediction = model.predict(X)[0]
        
        # Simple response
        response = {
            "predicted_delivery_time": round(prediction, 1),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs"
    }), 500

def create_sample_model():
    """Create a simple sample model for testing if no model is found."""
    logger.warning("No trained model found. Creating a sample model for testing...")
    
    # This is just for testing - create a simple mock model
    class MockModel:
        def predict(self, X):
            # Simple prediction based on distance and traffic
            if hasattr(X, 'iloc'):
                # If it's a DataFrame
                distance = X.iloc[0, 0] if len(X.columns) > 0 else 5.0  # Assume first column is distance
                traffic = X.iloc[0, 5] if len(X.columns) > 5 else 0.5   # Assume traffic is around 6th column
            else:
                # If it's an array
                distance = X[0][0] if len(X[0]) > 0 else 5.0
                traffic = X[0][5] if len(X[0]) > 5 else 0.5
            
            # Simple formula: base time + distance factor + traffic factor
            base_time = 20
            distance_factor = distance * 3
            traffic_factor = traffic * 15
            
            prediction = base_time + distance_factor + traffic_factor
            return [max(15, prediction)]  # Minimum 15 minutes
    
    global model, preprocessor
    model = MockModel()
    
    # Create a mock preprocessor
    class MockPreprocessor:
        def prepare_data(self, df, fit=False):
            return df
        
        def get_feature_names(self):
            return ['distance_km', 'order_hour', 'day_of_week', 'is_weekend', 
                   'temperature', 'traffic_density', 'restaurant_rating']
    
    preprocessor = MockPreprocessor()
    logger.info("Sample model created for testing")

if __name__ == '__main__':
    logger.info("Starting Flask API server...")
    
    # Try to load model and preprocessor
    if not load_model_and_preprocessor():
        logger.warning("Failed to load trained model. Creating sample model for testing...")
        create_sample_model()
    
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)