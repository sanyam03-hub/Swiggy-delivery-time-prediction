"""
Linear regression model for delivery time prediction.
"""
from sklearn.linear_model import LinearRegression
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.base_model import BaseModel

class LinearDeliveryModel(BaseModel):
    """Linear regression model for delivery time prediction."""
    
    def __init__(self):
        super().__init__("Linear Regression")
    
    def create_model(self, fit_intercept=True, normalize=False, **params):
        """
        Create linear regression model.
        
        Args:
            fit_intercept: Whether to calculate the intercept
            normalize: Whether to normalize features (deprecated in newer sklearn)
            **params: Additional parameters for LinearRegression
        """
        # Handle normalize parameter for newer sklearn versions
        if 'normalize' in params:
            params.pop('normalize')  # Remove deprecated parameter
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            **params
        )
        
        return self.model
    
    def get_coefficients(self):
        """
        Get model coefficients.
        
        Returns:
            dict: Feature coefficients and intercept
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        coefficients = {
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(self.feature_names, self.model.coef_))
        }
        
        return coefficients