"""
Tree-based models for delivery time prediction.
Includes Random Forest, XGBoost, and LightGBM implementations.
"""
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.base_model import BaseModel

class RandomForestDeliveryModel(BaseModel):
    """Random Forest model for delivery time prediction."""
    
    def __init__(self):
        super().__init__("Random Forest")
    
    def create_model(self, n_estimators=100, max_depth=10, min_samples_split=5,
                     min_samples_leaf=2, random_state=42, **params):
        """
        Create Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
            **params: Additional parameters
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **params
        )
        
        return self.model

class XGBoostDeliveryModel(BaseModel):
    """XGBoost model for delivery time prediction."""
    
    def __init__(self):
        super().__init__("XGBoost")
        self._xgb_available = False
        self._check_xgb_availability()
    
    def _check_xgb_availability(self):
        """Check if XGBoost is available."""
        try:
            import xgboost as xgb
            self._xgb_available = True
            self._xgb = xgb
        except ImportError:
            print("XGBoost not installed. Please install with: pip install xgboost")
            self._xgb_available = False
    
    def create_model(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, **params):
        """
        Create XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column sampling ratio
            random_state: Random seed
            **params: Additional parameters
        """
        if not self._xgb_available:
            raise ImportError("XGBoost not available. Please install xgboost.")
        
        self.model = self._xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **params
        )
        
        return self.model

class LightGBMDeliveryModel(BaseModel):
    """LightGBM model for delivery time prediction."""
    
    def __init__(self):
        super().__init__("LightGBM")
        self._lgb_available = False
        self._check_lgb_availability()
    
    def _check_lgb_availability(self):
        """Check if LightGBM is available."""
        try:
            import lightgbm as lgb
            self._lgb_available = True
            self._lgb = lgb
        except ImportError:
            print("LightGBM not installed. Please install with: pip install lightgbm")
            self._lgb_available = False
    
    def create_model(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, 
                     verbose=-1, **params):
        """
        Create LightGBM model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column sampling ratio
            random_state: Random seed
            verbose: Verbosity level
            **params: Additional parameters
        """
        if not self._lgb_available:
            raise ImportError("LightGBM not available. Please install lightgbm.")
        
        self.model = self._lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=verbose,
            **params
        )
        
        return self.model

# Model factory function
def create_model(model_type: str, **params):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('linear', 'random_forest', 'xgboost', 'lightgbm')
        **params: Model parameters
    
    Returns:
        BaseModel: Instantiated model
    """
    model_map = {
        'linear': lambda: LinearDeliveryModel(),
        'random_forest': lambda: RandomForestDeliveryModel(),
        'xgboost': lambda: XGBoostDeliveryModel(),
        'lightgbm': lambda: LightGBMDeliveryModel()
    }
    
    if model_type.lower() not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    
    model = model_map[model_type.lower()]()
    model.create_model(**params)
    
    return model

if __name__ == "__main__":
    # Test model creation
    from src.models.linear_model import LinearDeliveryModel
    
    print("Testing model creation...")
    
    # Test Linear Regression
    linear_model = LinearDeliveryModel()
    linear_model.create_model()
    print(f"Created: {linear_model}")
    
    # Test Random Forest
    rf_model = RandomForestDeliveryModel()
    rf_model.create_model()
    print(f"Created: {rf_model}")
    
    # Test XGBoost (if available)
    try:
        xgb_model = XGBoostDeliveryModel()
        xgb_model.create_model()
        print(f"Created: {xgb_model}")
    except ImportError as e:
        print(f"XGBoost not available: {e}")
    
    # Test LightGBM (if available)
    try:
        lgb_model = LightGBMDeliveryModel()
        lgb_model.create_model()
        print(f"Created: {lgb_model}")
    except ImportError as e:
        print(f"LightGBM not available: {e}")
    
    print("Model creation tests completed!")