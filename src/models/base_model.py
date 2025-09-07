"""
Base model class for Swiggy delivery time prediction.
Provides common interface for all machine learning models.
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseModel(ABC):
    """Base class for all prediction models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
    @abstractmethod
    def create_model(self, **params):
        """Create the underlying model with given parameters."""
        pass
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info(f"Training {self.model_name} model...")
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train the model
        self.model.fit(X_train, y_train, **kwargs)
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.predict(X_train)
        self.training_metrics = self.calculate_metrics(y_train, y_train_pred)
        
        logger.info(f"Training completed for {self.model_name}")
        logger.info(f"Training MAE: {self.training_metrics['mae']:.3f}")
        logger.info(f"Training RMSE: {self.training_metrics['rmse']:.3f}")
        logger.info(f"Training R²: {self.training_metrics['r2']:.3f}")
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            self.validation_metrics = self.calculate_metrics(y_val, y_val_pred)
            
            logger.info(f"Validation MAE: {self.validation_metrics['mae']:.3f}")
            logger.info(f"Validation RMSE: {self.validation_metrics['rmse']:.3f}")
            logger.info(f"Validation R²: {self.validation_metrics['r2']:.3f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Feature columns do not match training features")
        
        return self.model.predict(X)
    
    def predict_single(self, features: dict) -> float:
        """
        Make a single prediction from feature dictionary.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            float: Single prediction
        """
        if not self.feature_names:
            raise ValueError("Feature names not available. Train model first.")
        
        # Create DataFrame with single row
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        prediction = self.predict(df)
        return float(prediction[0])
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            dict: Performance metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance if available.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"{self.model_name} does not support feature importance")
            return None
    
    def get_training_metrics(self) -> dict:
        """Get training metrics."""
        return self.training_metrics
    
    def get_validation_metrics(self) -> dict:
        """Get validation metrics."""
        return self.validation_metrics
    
    def save_model(self, file_path: str):
        """
        Save the trained model.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'is_trained': self.is_trained
        }
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """
        Load a trained model.
        
        Args:
            file_path: Path to the saved model
        """
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.validation_metrics = model_data['validation_metrics']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from: {file_path}")
    
    def summary(self) -> dict:
        """
        Get model summary.
        
        Returns:
            dict: Model summary information
        """
        summary_info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }
        
        if hasattr(self.model, 'n_estimators'):
            summary_info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            summary_info['max_depth'] = self.model.max_depth
        
        return summary_info
    
    def __str__(self):
        """String representation of the model."""
        if self.is_trained:
            val_mae = self.validation_metrics.get('mae', 'N/A')
            val_r2 = self.validation_metrics.get('r2', 'N/A')
            return f"{self.model_name} (MAE: {val_mae}, R²: {val_r2})"
        else:
            return f"{self.model_name} (Not trained)"
    
    def __repr__(self):
        """Representation of the model."""
        return self.__str__()