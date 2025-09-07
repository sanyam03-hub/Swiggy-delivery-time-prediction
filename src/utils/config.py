"""
Configuration settings for the Swiggy delivery prediction project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 5000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "xgboost")
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR))
RETRAIN_INTERVAL = int(os.getenv("RETRAIN_INTERVAL", 24))

# Data Configuration
DATASET_SIZE = int(os.getenv("DATASET_SIZE", 10000))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))

# External APIs
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Model hyperparameters
MODEL_PARAMS = {
    "linear": {
        "fit_intercept": True,
        "normalize": False
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1
    }
}

# Feature columns
FEATURE_COLUMNS = [
    "distance_km", "order_hour", "day_of_week", "is_weekend",
    "temperature", "traffic_density", "restaurant_rating",
    "order_value", "weather_condition_encoded", "food_category_encoded"
]

TARGET_COLUMN = "delivery_time_minutes"

# Cities configuration for synthetic data
CITIES_CONFIG = {
    "bangalore": {
        "lat_range": (12.8, 13.1),
        "lng_range": (77.4, 77.8),
        "traffic_multiplier": 1.2
    },
    "mumbai": {
        "lat_range": (19.0, 19.3),
        "lng_range": (72.7, 73.0),
        "traffic_multiplier": 1.5
    },
    "delhi": {
        "lat_range": (28.4, 28.8),
        "lng_range": (76.8, 77.3),
        "traffic_multiplier": 1.3
    }
}