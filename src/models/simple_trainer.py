"""
Simple model trainer to create models for API testing.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import DataPreprocessor
from src.utils.config import SYNTHETIC_DATA_DIR, MODELS_DIR

def create_simple_models():
    """Create simple models for API testing."""
    print("Creating simple models for API testing...")
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    data_file = SYNTHETIC_DATA_DIR / "swiggy_delivery_data.csv"
    if not data_file.exists():
        print("No dataset found. Please run data_generator.py first.")
        return False
    
    # Load data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(str(data_file))
    
    # Prepare data
    df_processed = preprocessor.prepare_data(df, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    # Create and train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Calculate performance
    rf_score = rf_model.score(X_test, y_test)
    print(f"Random Forest R² Score: {rf_score:.3f}")
    
    # Save Random Forest model
    rf_model_data = {
        'model': rf_model,
        'model_name': 'Random Forest',
        'feature_names': list(X_train.columns),
        'training_metrics': {'r2': rf_score},
        'validation_metrics': {'r2': rf_score},
        'is_trained': True
    }
    
    rf_path = MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(rf_model_data, rf_path)
    print(f"Random Forest model saved to: {rf_path}")
    
    # Save as best model
    best_model_path = MODELS_DIR / "best_model_random_forest.joblib"
    joblib.dump(rf_model_data, best_model_path)
    print(f"Best model saved to: {best_model_path}")
    
    # Save preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    preprocessor.save_preprocessor(str(preprocessor_path))
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    return True

if __name__ == "__main__":
    success = create_simple_models()
    if success:
        print("✅ Simple models created successfully!")
        print("You can now run the API servers.")
    else:
        print("❌ Failed to create models.")