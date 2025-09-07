"""
Advanced hyperparameter tuning for Swiggy delivery time prediction models.
Includes GridSearchCV, RandomizedSearchCV, and Bayesian optimization.
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import os
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import DataPreprocessor
from src.models.linear_model import LinearDeliveryModel
from src.models.tree_models import RandomForestDeliveryModel, XGBoostDeliveryModel, LightGBMDeliveryModel
from src.models.model_evaluator import ModelEvaluator
from src.utils.config import SYNTHETIC_DATA_DIR, MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class HyperparameterTuner:
    """Advanced hyperparameter tuning for delivery time prediction models."""
    
    def __init__(self):
        self.best_models = {}
        self.tuning_results = {}
        
    def get_parameter_grids(self):
        """Define parameter grids for each model type."""
        parameter_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9, -1],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2],
                'num_leaves': [31, 50, 100]
            }
        }
        
        return parameter_grids
    
    def get_reduced_parameter_grids(self):
        """Get reduced parameter grids for faster tuning."""
        reduced_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
        
        return reduced_grids
    
    def tune_random_forest(self, X_train, y_train, cv=3, method='grid', n_iter=20):
        """Tune Random Forest hyperparameters."""
        logger.info("Tuning Random Forest hyperparameters...")
        
        # Create base model
        rf_model = RandomForestDeliveryModel()
        rf_model.create_model(random_state=42)
        
        # Get parameter grid
        param_grids = self.get_parameter_grids() if method == 'grid' else self.get_reduced_parameter_grids()
        param_grid = param_grids['Random Forest']
        
        # Setup scorer
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        
        start_time = time.time()
        
        if method == 'grid':
            # Use GridSearchCV
            search = GridSearchCV(
                rf_model.model,
                param_grid,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
        else:
            # Use RandomizedSearchCV
            search = RandomizedSearchCV(
                rf_model.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        search.fit(X_train, y_train)
        
        end_time = time.time()
        tuning_time = end_time - start_time
        
        # Create optimized model
        optimized_model = RandomForestDeliveryModel()
        optimized_model.create_model(**search.best_params_)
        optimized_model.model = search.best_estimator_
        optimized_model.is_trained = True
        optimized_model.feature_names = list(X_train.columns)
        
        self.best_models['Random Forest'] = optimized_model
        self.tuning_results['Random Forest'] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'tuning_time': tuning_time,
            'method': method
        }
        
        logger.info(f"Random Forest tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return optimized_model
    
    def tune_xgboost(self, X_train, y_train, cv=3, method='random', n_iter=20):
        """Tune XGBoost hyperparameters."""
        try:
            logger.info("Tuning XGBoost hyperparameters...")
            
            # Create base model
            xgb_model = XGBoostDeliveryModel()
            xgb_model.create_model(random_state=42)
            
            # Get parameter grid
            param_grids = self.get_parameter_grids() if method == 'grid' else self.get_reduced_parameter_grids()
            param_grid = param_grids['XGBoost']
            
            # Setup scorer
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            
            start_time = time.time()
            
            if method == 'grid':
                # Use GridSearchCV
                search = GridSearchCV(
                    xgb_model.model,
                    param_grid,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    verbose=1
                )
            else:
                # Use RandomizedSearchCV
                search = RandomizedSearchCV(
                    xgb_model.model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )
            
            search.fit(X_train, y_train)
            
            end_time = time.time()
            tuning_time = end_time - start_time
            
            # Create optimized model
            optimized_model = XGBoostDeliveryModel()
            optimized_model.create_model(**search.best_params_)
            optimized_model.model = search.best_estimator_
            optimized_model.is_trained = True
            optimized_model.feature_names = list(X_train.columns)
            
            self.best_models['XGBoost'] = optimized_model
            self.tuning_results['XGBoost'] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'tuning_time': tuning_time,
                'method': method
            }
            
            logger.info(f"XGBoost tuning completed in {tuning_time:.2f} seconds")
            logger.info(f"Best params: {search.best_params_}")
            logger.info(f"Best score: {search.best_score_:.4f}")
            
            return optimized_model
            
        except ImportError:
            logger.warning("XGBoost not available, skipping tuning")
            return None
    
    def tune_lightgbm(self, X_train, y_train, cv=3, method='random', n_iter=20):
        """Tune LightGBM hyperparameters."""
        try:
            logger.info("Tuning LightGBM hyperparameters...")
            
            # Create base model
            lgb_model = LightGBMDeliveryModel()
            lgb_model.create_model(random_state=42)
            
            # Get parameter grid
            param_grids = self.get_parameter_grids() if method == 'grid' else self.get_reduced_parameter_grids()
            param_grid = param_grids['LightGBM']
            
            # Setup scorer
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            
            start_time = time.time()
            
            if method == 'grid':
                # Use GridSearchCV
                search = GridSearchCV(
                    lgb_model.model,
                    param_grid,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    verbose=1
                )
            else:
                # Use RandomizedSearchCV
                search = RandomizedSearchCV(
                    lgb_model.model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )
            
            search.fit(X_train, y_train)
            
            end_time = time.time()
            tuning_time = end_time - start_time
            
            # Create optimized model
            optimized_model = LightGBMDeliveryModel()
            optimized_model.create_model(**search.best_params_)
            optimized_model.model = search.best_estimator_
            optimized_model.is_trained = True
            optimized_model.feature_names = list(X_train.columns)
            
            self.best_models['LightGBM'] = optimized_model
            self.tuning_results['LightGBM'] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'tuning_time': tuning_time,
                'method': method
            }
            
            logger.info(f"LightGBM tuning completed in {tuning_time:.2f} seconds")
            logger.info(f"Best params: {search.best_params_}")
            logger.info(f"Best score: {search.best_score_:.4f}")
            
            return optimized_model
            
        except ImportError:
            logger.warning("LightGBM not available, skipping tuning")
            return None
    
    def tune_all_models(self, X_train, y_train, cv=3, method='random', n_iter=20):
        """Tune hyperparameters for all available models."""
        logger.info("Starting comprehensive hyperparameter tuning...")
        
        # Tune Random Forest
        self.tune_random_forest(X_train, y_train, cv, method, n_iter)
        
        # Tune XGBoost
        self.tune_xgboost(X_train, y_train, cv, method, n_iter)
        
        # Tune LightGBM
        self.tune_lightgbm(X_train, y_train, cv, method, n_iter)
        
        logger.info("Hyperparameter tuning completed for all models")
    
    def evaluate_tuned_models(self, X_test, y_test):
        """Evaluate all tuned models."""
        logger.info("Evaluating tuned models...")
        
        evaluator = ModelEvaluator()
        
        # Add tuned models to evaluator
        for name, model in self.best_models.items():
            evaluator.add_model(f"{name} (Tuned)", model)
        
        # Evaluate models
        results_df = evaluator.evaluate_all_models(X_test, y_test)
        
        return evaluator, results_df
    
    def save_tuned_models(self):
        """Save all tuned models."""
        logger.info("Saving tuned models...")
        
        saved_models = {}
        
        for name, model in self.best_models.items():
            # Create safe filename
            safe_name = name.lower().replace(' ', '_')
            model_path = MODELS_DIR / f"{safe_name}_tuned_model.joblib"
            
            try:
                # Create model data with tuning information
                model_data = {
                    'model': model.model,
                    'model_name': f"{name} (Tuned)",
                    'feature_names': model.feature_names,
                    'is_trained': model.is_trained,
                    'best_params': self.tuning_results[name]['best_params'],
                    'best_score': self.tuning_results[name]['best_score'],
                    'tuning_method': self.tuning_results[name]['method'],
                    'tuning_time': self.tuning_results[name]['tuning_time']
                }
                
                joblib.dump(model_data, model_path)
                saved_models[name] = str(model_path)
                logger.info(f"✓ Saved tuned {name} to {model_path}")
                
            except Exception as e:
                logger.error(f"✗ Error saving tuned {name}: {str(e)}")
        
        return saved_models
    
    def generate_tuning_report(self):
        """Generate a comprehensive tuning report."""
        report = []
        report.append("="*70)
        report.append("HYPERPARAMETER TUNING REPORT")
        report.append("="*70)
        report.append("")
        
        if not self.tuning_results:
            report.append("No tuning results available.")
            return "\n".join(report)
        
        for model_name, results in self.tuning_results.items():
            report.append(f"MODEL: {model_name}")
            report.append("-" * 40)
            report.append(f"Tuning Method: {results['method'].upper()}")
            report.append(f"Best CV Score: {results['best_score']:.4f}")
            report.append(f"Tuning Time: {results['tuning_time']:.2f} seconds")
            report.append("Best Parameters:")
            
            for param, value in results['best_params'].items():
                report.append(f"  {param}: {value}")
            
            report.append("")
        
        return "\n".join(report)

def load_and_prepare_data():
    """Load and prepare the dataset."""
    logger.info("Loading and preparing data for hyperparameter tuning...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    data_file = SYNTHETIC_DATA_DIR / "swiggy_delivery_data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")
    
    df = preprocessor.load_data(str(data_file))
    
    # Prepare data
    df_processed = preprocessor.prepare_data(df, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def main():
    """Main hyperparameter tuning pipeline."""
    logger.info("Starting advanced hyperparameter tuning...")
    
    try:
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_prepare_data()
        
        # Create tuner
        tuner = HyperparameterTuner()
        
        # Perform hyperparameter tuning
        # Using RandomizedSearchCV for faster tuning
        tuner.tune_all_models(X_train, y_train, cv=3, method='random', n_iter=30)
        
        # Evaluate tuned models
        evaluator, results_df = tuner.evaluate_tuned_models(X_test, y_test)
        
        # Save tuned models
        saved_models = tuner.save_tuned_models()
        
        # Save best tuned model
        if evaluator.best_model:
            best_tuned_path = evaluator.save_best_model()
            logger.info(f"Best tuned model saved to: {best_tuned_path}")
        
        # Generate reports
        tuning_report = tuner.generate_tuning_report()
        evaluation_report = evaluator.generate_report(results_df)
        
        # Save reports
        tuning_report_path = MODELS_DIR / "hyperparameter_tuning_report.txt"
        with open(tuning_report_path, 'w') as f:
            f.write(tuning_report)
        
        evaluation_report_path = MODELS_DIR / "tuned_models_evaluation_report.txt"
        with open(evaluation_report_path, 'w') as f:
            f.write(evaluation_report)
        
        # Display results
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING COMPLETED!")
        print("="*80)
        print(tuning_report)
        print("\n" + "="*80)
        print("TUNED MODELS EVALUATION:")
        print("="*80)
        print(evaluation_report)
        print("\n" + "="*80)
        print("SAVED FILES:")
        print("-" * 40)
        print(f"Tuning Report: {tuning_report_path}")
        print(f"Evaluation Report: {evaluation_report_path}")
        print("\nTuned models:")
        for name, path in saved_models.items():
            print(f"  {name}: {path}")
        
        return results_df, tuner, evaluator
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    results_df, tuner, evaluator = main()