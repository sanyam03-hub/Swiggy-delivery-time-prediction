"""
Model evaluation utilities for comparing different models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger
from src.utils.config import MODELS_DIR

logger = setup_logger(__name__)

class ModelEvaluator:
    """Evaluate and compare multiple models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def add_model(self, name: str, model):
        """
        Add a model for evaluation.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all added models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Training all models...")
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train, X_val, y_val)
                logger.info(f"✓ {name} training completed")
            except Exception as e:
                logger.error(f"✗ Error training {name}: {str(e)}")
        
        logger.info("All models training completed")
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate all models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            pd.DataFrame: Comparison of all models
        """
        logger.info("Evaluating all models on test data...")
        
        results = []
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    metrics = self.evaluate_model(model, X_test, y_test, name)
                    
                    result = {
                        'Model': name,
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'R²': metrics['r2'],
                        'MAPE': metrics['mape']
                    }
                    
                    results.append(result)
                    logger.info(f"✓ {name} evaluated - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
                    
                except Exception as e:
                    logger.error(f"✗ Error evaluating {name}: {str(e)}")
            else:
                logger.warning(f"Model {name} is not trained, skipping evaluation")
        
        results_df = pd.DataFrame(results)
        
        # Sort by MAE (lower is better)
        results_df = results_df.sort_values('MAE').reset_index(drop=True)
        
        # Identify best model
        if not results_df.empty:
            best_model_name = results_df.iloc[0]['Model']
            self.best_model = self.models[best_model_name]
            logger.info(f"Best model: {best_model_name} (MAE: {results_df.iloc[0]['MAE']:.3f})")
        
        return results_df
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='neg_mean_absolute_error'):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring)
        
        cv_results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
        
        logger.info(f"CV Score: {cv_results['mean_score']:.3f} (+/- {cv_results['std_score']:.3f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_class, param_grid, X_train, y_train, 
                            cv=3, scoring='neg_mean_absolute_error'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_class: Model class to tune
            param_grid: Parameter grid for tuning
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            dict: Best parameters and score
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Create base model
        base_model = model_class()
        base_model.create_model()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        logger.info(f"Best parameters: {tuning_results['best_params']}")
        logger.info(f"Best score: {tuning_results['best_score']:.3f}")
        
        return tuning_results
    
    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot model comparison results.
        
        Args:
            results_df: DataFrame with model results
            save_path: Path to save the plot
        """
        if results_df.empty:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # MAE comparison
        axes[0, 0].bar(results_df['Model'], results_df['MAE'], color='skyblue')
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE (minutes)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(results_df['Model'], results_df['RMSE'], color='lightcoral')
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE (minutes)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(results_df['Model'], results_df['R²'], color='lightgreen')
        axes[1, 0].set_title('R² Score')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(results_df['Model'], results_df['MAPE'], color='orange')
        axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, model, X_test, y_test, model_name: str, save_path: str = None):
        """
        Plot predictions vs actual values.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            save_path: Path to save the plot
        """
        predictions = model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Delivery Time (minutes)')
        plt.ylabel('Predicted Delivery Time (minutes)')
        plt.title(f'{model_name}: Predictions vs Actual')
        
        # Add R² score to plot
        r2 = r2_score(y_test, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def feature_importance_comparison(self, save_path: str = None):
        """
        Compare feature importance across models.
        
        Args:
            save_path: Path to save the plot
        """
        importance_data = []
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    importance_df = model.get_feature_importance()
                    if importance_df is not None:
                        importance_df['model'] = name
                        importance_data.append(importance_df)
                except Exception as e:
                    logger.warning(f"Could not get feature importance for {name}: {str(e)}")
        
        if not importance_data:
            logger.warning("No feature importance data available")
            return
        
        # Combine all importance data
        combined_importance = pd.concat(importance_data, ignore_index=True)
        
        # Get top 10 features
        top_features = combined_importance.groupby('feature')['importance'].mean().nlargest(10).index
        plot_data = combined_importance[combined_importance['feature'].isin(top_features)]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=plot_data, x='importance', y='feature', hue='model')
        plt.title('Feature Importance Comparison (Top 10 Features)')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def save_best_model(self, save_dir: str = None):
        """
        Save the best performing model.
        
        Args:
            save_dir: Directory to save the model
        """
        if self.best_model is None:
            logger.warning("No best model identified. Run evaluate_all_models first.")
            return
        
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_path = Path(save_dir) / f"best_model_{self.best_model.model_name.lower().replace(' ', '_')}.joblib"
        self.best_model.save_model(str(save_path))
        
        logger.info(f"Best model ({self.best_model.model_name}) saved to: {save_path}")
        
        return str(save_path)
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a text report of model performance.
        
        Args:
            results_df: DataFrame with model results
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("="*60)
        report.append("MODEL PERFORMANCE REPORT")
        report.append("="*60)
        report.append("")
        
        if results_df.empty:
            report.append("No models evaluated.")
            return "\n".join(report)
        
        # Best model summary
        best_model = results_df.iloc[0]
        report.append(f"BEST PERFORMING MODEL: {best_model['Model']}")
        report.append(f"MAE: {best_model['MAE']:.3f} minutes")
        report.append(f"RMSE: {best_model['RMSE']:.3f} minutes")
        report.append(f"R²: {best_model['R²']:.3f}")
        report.append(f"MAPE: {best_model['MAPE']:.2f}%")
        report.append("")
        
        # All models comparison
        report.append("ALL MODELS COMPARISON:")
        report.append("-" * 40)
        for _, row in results_df.iterrows():
            report.append(f"{row['Model']:<15} | MAE: {row['MAE']:.3f} | R²: {row['R²']:.3f}")
        
        report.append("")
        report.append("Metrics explanation:")
        report.append("- MAE: Mean Absolute Error (lower is better)")
        report.append("- RMSE: Root Mean Squared Error (lower is better)")
        report.append("- R²: Coefficient of determination (higher is better)")
        report.append("- MAPE: Mean Absolute Percentage Error (lower is better)")
        
        return "\n".join(report)