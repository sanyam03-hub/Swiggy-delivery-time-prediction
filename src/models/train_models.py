"""
Main training script for all delivery time prediction models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import DataPreprocessor
from src.models.linear_model import LinearDeliveryModel
from src.models.tree_models import RandomForestDeliveryModel, XGBoostDeliveryModel, LightGBMDeliveryModel
from src.models.model_evaluator import ModelEvaluator
from src.utils.config import SYNTHETIC_DATA_DIR, MODELS_DIR, MODEL_PARAMS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def install_missing_packages():
    """Install missing packages if needed."""
    try:
        import xgboost
        logger.info("XGBoost is available")
    except ImportError:
        logger.warning("XGBoost not found. Installing...")
        os.system("pip install xgboost")
    
    try:
        import lightgbm
        logger.info("LightGBM is available")
    except ImportError:
        logger.warning("LightGBM not found. Installing...")
        os.system("pip install lightgbm")
    
    try:
        import matplotlib
        import seaborn
        logger.info("Plotting libraries are available")
    except ImportError:
        logger.warning("Plotting libraries not found. Installing...")
        os.system("pip install matplotlib seaborn")

def load_and_prepare_data():
    """Load and prepare the dataset."""
    logger.info("Loading and preparing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    data_file = SYNTHETIC_DATA_DIR / "swiggy_delivery_data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}. Please run data_generator.py first.")
    
    df = preprocessor.load_data(str(data_file))
    
    # Prepare data
    df_processed = preprocessor.prepare_data(df, fit=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    # Save preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    preprocessor.save_preprocessor(str(preprocessor_path))
    
    logger.info("Data preparation completed")
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def create_all_models():
    """Create all models with their configurations."""
    logger.info("Creating all models...")
    
    models = {}
    
    # Linear Regression
    linear_model = LinearDeliveryModel()
    linear_model.create_model(**MODEL_PARAMS.get('linear', {}))
    models['Linear Regression'] = linear_model
    
    # Random Forest
    rf_model = RandomForestDeliveryModel()
    rf_model.create_model(**MODEL_PARAMS.get('random_forest', {}))
    models['Random Forest'] = rf_model
    
    # XGBoost (if available)
    try:
        xgb_model = XGBoostDeliveryModel()
        xgb_model.create_model(**MODEL_PARAMS.get('xgboost', {}))
        models['XGBoost'] = xgb_model
        logger.info("XGBoost model created")
    except ImportError as e:
        logger.warning(f"XGBoost not available: {e}")
    
    # LightGBM (if available)
    try:
        lgb_model = LightGBMDeliveryModel()
        lgb_model.create_model(**MODEL_PARAMS.get('lightgbm', {}))
        models['LightGBM'] = lgb_model
        logger.info("LightGBM model created")
    except ImportError as e:
        logger.warning(f"LightGBM not available: {e}")
    
    logger.info(f"Created {len(models)} models: {list(models.keys())}")
    return models

def train_and_evaluate_models(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate all models."""
    logger.info("Training and evaluating all models...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Add models to evaluator
    for name, model in models.items():
        evaluator.add_model(name, model)
    
    # Train all models
    evaluator.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate all models
    results_df = evaluator.evaluate_all_models(X_test, y_test)
    
    return evaluator, results_df

def save_individual_models(models):
    """Save all trained models individually."""
    logger.info("Saving individual models...")
    
    saved_models = {}
    for name, model in models.items():
        if model.is_trained:
            # Create safe filename
            safe_name = name.lower().replace(' ', '_')
            model_path = MODELS_DIR / f"{safe_name}_model.joblib"
            
            try:
                model.save_model(str(model_path))
                saved_models[name] = str(model_path)
                logger.info(f"✓ Saved {name} to {model_path}")
            except Exception as e:
                logger.error(f"✗ Error saving {name}: {str(e)}")
    
    return saved_models

def generate_plots(evaluator, results_df, X_test, y_test):
    """Generate evaluation plots."""
    logger.info("Generating evaluation plots...")
    
    try:
        # Ensure plots directory exists
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Model comparison plot
        evaluator.plot_model_comparison(results_df, save_path=str(plots_dir / "model_comparison.png"))
        
        # Feature importance comparison
        evaluator.feature_importance_comparison(save_path=str(plots_dir / "feature_importance.png"))
        
        # Predictions vs actual for best model
        if evaluator.best_model:
            evaluator.plot_predictions_vs_actual(
                evaluator.best_model, X_test, y_test, 
                evaluator.best_model.model_name,
                save_path=str(plots_dir / "predictions_vs_actual.png")
            )
        
        logger.info("Plots generated successfully")
        
    except Exception as e:
        logger.warning(f"Error generating plots: {str(e)}")

def perform_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Perform hyperparameter tuning on selected models."""
    try:
        from src.models.hyperparameter_tuning import HyperparameterTuner
        
        logger.info("Starting hyperparameter tuning...")
        
        # Create tuner
        tuner = HyperparameterTuner()
        
        # Perform tuning (using RandomizedSearchCV for speed)
        tuner.tune_all_models(X_train, y_train, cv=3, method='random', n_iter=20)
        
        # Evaluate tuned models
        tuned_evaluator, tuned_results_df = tuner.evaluate_tuned_models(X_test, y_test)
        
        # Save tuned models
        saved_tuned_models = tuner.save_tuned_models()
        
        # Generate tuning report
        tuning_report = tuner.generate_tuning_report()
        tuning_report_path = MODELS_DIR / "hyperparameter_tuning_report.txt"
        with open(tuning_report_path, 'w') as f:
            f.write(tuning_report)
        
        logger.info("Hyperparameter tuning completed successfully")
        
        return tuner, tuned_evaluator, tuned_results_df, saved_tuned_models
        
    except ImportError as e:
        logger.warning(f"Hyperparameter tuning module not available: {e}")
        return None, None, None, {}
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {str(e)}")
        return None, None, None, {}

def main(enable_tuning=True):
    """Main training pipeline with optional hyperparameter tuning."""
    logger.info("Starting Swiggy delivery time prediction model training...")
    
    try:
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Install missing packages
        install_missing_packages()
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_prepare_data()
        
        # Create models
        models = create_all_models()
        
        # Train and evaluate baseline models
        evaluator, results_df = train_and_evaluate_models(
            models, X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Save individual models
        saved_models = save_individual_models(models)
        
        # Save best baseline model
        best_model_path = evaluator.save_best_model()
        
        # Generate plots for baseline models
        generate_plots(evaluator, results_df, X_test, y_test)
        
        # Generate baseline report
        baseline_report = evaluator.generate_report(results_df)
        
        # Save baseline report
        baseline_report_path = MODELS_DIR / "baseline_training_report.txt"
        with open(baseline_report_path, 'w') as f:
            f.write(baseline_report)
        
        # Hyperparameter tuning (optional)
        tuner = None
        tuned_evaluator = None
        tuned_results_df = None
        saved_tuned_models = {}
        
        if enable_tuning:
            logger.info("\n" + "="*60)
            logger.info("STARTING HYPERPARAMETER TUNING")
            logger.info("="*60)
            
            tuner, tuned_evaluator, tuned_results_df, saved_tuned_models = perform_hyperparameter_tuning(
                X_train, y_train, X_test, y_test
            )
            
            if tuned_evaluator and tuned_results_df is not None:
                # Save best tuned model
                best_tuned_path = tuned_evaluator.save_best_model()
                
                # Generate tuned models report
                tuned_report = tuned_evaluator.generate_report(tuned_results_df)
                tuned_report_path = MODELS_DIR / "tuned_models_evaluation_report.txt"
                with open(tuned_report_path, 'w') as f:
                    f.write(tuned_report)
        
        # Display results
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nBASELINE MODELS PERFORMANCE:")
        print("-" * 50)
        print(baseline_report)
        
        if tuned_results_df is not None:
            print("\n" + "="*80)
            print("HYPERPARAMETER TUNING RESULTS:")
            print("="*80)
            tuning_report = tuner.generate_tuning_report()
            print(tuning_report)
            
            print("\nTUNED MODELS PERFORMANCE:")
            print("-" * 50)
            tuned_report = tuned_evaluator.generate_report(tuned_results_df)
            print(tuned_report)
        
        print("\n" + "="*80)
        print("SAVED FILES:")
        print("-" * 40)
        print(f"Preprocessor: {MODELS_DIR / 'preprocessor.joblib'}")
        print(f"Best baseline model: {best_model_path}")
        print(f"Baseline report: {baseline_report_path}")
        
        if saved_tuned_models:
            print(f"\nHyperparameter tuning report: {MODELS_DIR / 'hyperparameter_tuning_report.txt'}")
            print(f"Tuned models report: {MODELS_DIR / 'tuned_models_evaluation_report.txt'}")
            print("\nTuned models:")
            for name, path in saved_tuned_models.items():
                print(f"  {name}: {path}")
        
        print("\nBaseline models:")
        for name, path in saved_models.items():
            print(f"  {name}: {path}")
        
        # Return both baseline and tuned results
        return {
            'baseline_results': results_df,
            'baseline_evaluator': evaluator,
            'tuned_results': tuned_results_df,
            'tuned_evaluator': tuned_evaluator,
            'tuner': tuner
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command line arguments for tuning option
    import argparse
    
    parser = argparse.ArgumentParser(description='Train delivery time prediction models')
    parser.add_argument('--no-tuning', action='store_true', 
                       help='Skip hyperparameter tuning (faster training)')
    parser.add_argument('--tuning-only', action='store_true',
                       help='Run only hyperparameter tuning (requires existing baseline models)')
    
    args = parser.parse_args()
    
    if args.tuning_only:
        # Run only hyperparameter tuning
        try:
            from src.models.hyperparameter_tuning import main as run_tuning
            run_tuning()
        except ImportError:
            logger.error("Hyperparameter tuning module not found")
    else:
        # Run full training pipeline
        enable_tuning = not args.no_tuning
        results = main(enable_tuning=enable_tuning)