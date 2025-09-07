"""
Data preprocessing module for Swiggy delivery time prediction.
Handles data cleaning, feature engineering, and transformation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED
from src.utils.helpers import calculate_distance, encode_categorical_features
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """Handle all data preprocessing tasks."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = TARGET_COLUMN
        self.is_fitted = False
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        logger.info(f"Missing values per column:\n{missing_values[missing_values > 0]}")
        
        # Handle missing values
        if missing_values.any():
            # For numeric columns, fill with median
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_columns = df_clean.select_dtypes(include=[object]).columns
            for col in categorical_columns:
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Remove outliers for delivery time (beyond 3 standard deviations)
        if self.target_column in df_clean.columns:
            mean_time = df_clean[self.target_column].mean()
            std_time = df_clean[self.target_column].std()
            lower_bound = mean_time - 3 * std_time
            upper_bound = mean_time + 3 * std_time
            
            outliers_mask = (df_clean[self.target_column] < lower_bound) | (df_clean[self.target_column] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                logger.info(f"Removing {outliers_count} outliers in delivery time")
                df_clean = df_clean[~outliers_mask].reset_index(drop=True)
        
        # Remove extreme distance outliers (beyond 20km)
        if 'distance_km' in df_clean.columns:
            extreme_distance = df_clean['distance_km'] > 20
            extreme_count = extreme_distance.sum()
            if extreme_count > 0:
                logger.info(f"Removing {extreme_count} records with extreme distances (>20km)")
                df_clean = df_clean[~extreme_distance].reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features from existing data.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Starting feature engineering...")
        df_features = df.copy()
        
        # Recalculate distance if coordinates exist
        if all(col in df_features.columns for col in ['restaurant_lat', 'restaurant_lng', 'customer_lat', 'customer_lng']):
            logger.info("Recalculating distances from coordinates...")
            df_features['calculated_distance_km'] = df_features.apply(
                lambda row: calculate_distance(
                    row['restaurant_lat'], row['restaurant_lng'],
                    row['customer_lat'], row['customer_lng']
                ), axis=1
            )
            
            # Use calculated distance if original is missing
            if 'distance_km' not in df_features.columns:
                df_features['distance_km'] = df_features['calculated_distance_km']
            else:
                # Fill missing distances with calculated ones
                mask = df_features['distance_km'].isnull()
                df_features.loc[mask, 'distance_km'] = df_features.loc[mask, 'calculated_distance_km']
        
        # Time-based features
        if 'order_timestamp' in df_features.columns:
            df_features['order_timestamp'] = pd.to_datetime(df_features['order_timestamp'])
            
            # Extract additional time features from timestamp
            df_features['month'] = df_features['order_timestamp'].dt.month
            df_features['quarter'] = df_features['order_timestamp'].dt.quarter
        
        # Create trigonometric time features (always create if order_hour and day_of_week exist)
        if 'order_hour' in df_features.columns:
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['order_hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['order_hour'] / 24)
            
            # Add month and quarter if not already present (for prediction scenarios)
            if 'month' not in df_features.columns:
                # Use current month as default for prediction
                from datetime import datetime
                current_date = datetime.now()
                df_features['month'] = current_date.month
                df_features['quarter'] = (current_date.month - 1) // 3 + 1
        
        if 'day_of_week' in df_features.columns:
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Distance bins
        if 'distance_km' in df_features.columns:
            df_features['distance_bin'] = pd.cut(
                df_features['distance_km'], 
                bins=[0, 2, 5, 10, float('inf')], 
                labels=['Very Close', 'Close', 'Medium', 'Far']
            ).astype(str)
        
        # Order value bins
        if 'order_value' in df_features.columns:
            df_features['order_value_bin'] = pd.cut(
                df_features['order_value'], 
                bins=[0, 200, 500, 1000, float('inf')], 
                labels=['Low', 'Medium', 'High', 'Very High']
            ).astype(str)
        
        # Traffic and weather interaction
        if 'traffic_density' in df_features.columns and 'weather_condition' in df_features.columns:
            # Heavy traffic + bad weather = significant delay
            bad_weather_mask = df_features['weather_condition'].isin(['Heavy Rain', 'Thunderstorm', 'Fog'])
            high_traffic_mask = df_features['traffic_density'] > 0.7
            df_features['traffic_weather_impact'] = (
                bad_weather_mask.astype(int) * high_traffic_mask.astype(int)
            )
        elif 'traffic_density' in df_features.columns:
            # If no weather condition, just check traffic
            df_features['traffic_weather_impact'] = (
                df_features['traffic_density'] > 0.7
            ).astype(int)
        else:
            # Default to 0 if neither available
            df_features['traffic_weather_impact'] = 0
        
        # Restaurant efficiency score
        if 'restaurant_rating' in df_features.columns:
            df_features['restaurant_efficiency'] = (df_features['restaurant_rating'] - 1) / 4  # 0-1 scale
        
        # Peak hour intensity
        if 'order_hour' in df_features.columns:
            peak_hours = [12, 13, 19, 20, 21]  # Lunch and dinner peaks
            df_features['peak_intensity'] = df_features['order_hour'].apply(
                lambda x: 1.0 if x in peak_hours else 0.5 if x in [11, 14, 18, 22] else 0.0
            )
        
        logger.info("Feature engineering completed")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        logger.info("Encoding categorical features...")
        df_encoded = df.copy()
        
        categorical_columns = ['weather_condition', 'food_category', 'restaurant_type', 'city']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    # Fit new encoder
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                    logger.info(f"Fitted encoder for {col}: {len(self.label_encoders[col].classes_)} categories")
                else:
                    # Use existing encoder
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_categories = set(self.label_encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].astype(str)
                        
                        # Replace unseen categories with most frequent category
                        unseen_mask = ~df_encoded[col].isin(known_categories)
                        if unseen_mask.any():
                            most_frequent = df_encoded[col].mode().iloc[0] if not df_encoded[col].mode().empty else self.label_encoders[col].classes_[0]
                            df_encoded.loc[unseen_mask, col] = most_frequent
                            logger.warning(f"Found unseen categories in {col}, replaced with {most_frequent}")
                        
                        df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
                    else:
                        logger.warning(f"No encoder found for {col}, skipping...")
        
        return df_encoded
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for modeling.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        # Define feature columns
        feature_columns = [
            'distance_km', 'order_hour', 'day_of_week', 'is_weekend',
            'temperature', 'traffic_density', 'restaurant_rating',
            'order_value', 'num_items', 'preparation_time',
            'weather_condition_encoded', 'food_category_encoded',
            'restaurant_type_encoded', 'city_encoded',
            'restaurant_efficiency', 'peak_intensity'
        ]
        
        # Add engineered features if they exist
        optional_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'traffic_weather_impact', 'month', 'quarter'
        ]
        
        # Only include features that exist in the dataframe
        available_features = [col for col in feature_columns + optional_features if col in df.columns]
        
        self.feature_columns = available_features
        logger.info(f"Selected {len(available_features)} features: {available_features}")
        
        return df[available_features + [self.target_column] if self.target_column in df.columns else available_features]
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        logger.info("Scaling features...")
        df_scaled = df.copy()
        
        # Features to scale (exclude encoded categorical features and target)
        features_to_scale = [col for col in df_scaled.columns if col != self.target_column and not col.endswith('_encoded')]
        
        if fit:
            scaled_values = self.scaler.fit_transform(df_scaled[features_to_scale])
            logger.info(f"Fitted scaler for {len(features_to_scale)} features")
        else:
            scaled_values = self.scaler.transform(df_scaled[features_to_scale])
        
        # Replace original values with scaled values
        df_scaled[features_to_scale] = scaled_values
        
        return df_scaled
    
    def prepare_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Input dataframe
            fit: Whether to fit preprocessors or use existing ones
            
        Returns:
            pd.DataFrame: Processed dataframe ready for modeling
        """
        logger.info("Starting complete data preparation pipeline...")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Engineer features
        df_features = self.engineer_features(df_clean)
        
        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, fit=fit)
        
        # Step 4: Select features
        df_selected = self.select_features(df_encoded)
        
        # Step 5: Scale features
        df_final = self.scale_features(df_selected, fit=fit)
        
        if fit:
            self.is_fitted = True
        
        logger.info(f"Data preparation completed. Final shape: {df_final.shape}")
        logger.info(f"Features: {self.feature_columns}")
        
        return df_final
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data (test: {test_size}, val: {val_size})")
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED, stratify=None
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"Train: {X_train.shape[0]} samples")
        logger.info(f"Validation: {X_val.shape[0]} samples")
        logger.info(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_names(self) -> list:
        """Get the list of feature column names."""
        return self.feature_columns if self.feature_columns else []
    
    def save_preprocessor(self, file_path: str):
        """Save the fitted preprocessor."""
        import joblib
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to: {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """Load a fitted preprocessor."""
        import joblib
        
        preprocessor_data = joblib.load(file_path)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data['target_column']
        self.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from: {file_path}")

# Feature engineering utilities
def create_feature_engineering_pipeline():
    """Create a comprehensive feature engineering pipeline."""
    preprocessor = DataPreprocessor()
    return preprocessor

if __name__ == "__main__":
    # Example usage
    from src.utils.config import SYNTHETIC_DATA_DIR
    
    # Load data
    preprocessor = DataPreprocessor()
    data_file = SYNTHETIC_DATA_DIR / "swiggy_delivery_data.csv"
    
    if data_file.exists():
        df = preprocessor.load_data(str(data_file))
        df_processed = preprocessor.prepare_data(df, fit=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
        
        print("\nPreprocessing completed successfully!")
        print(f"Features: {len(preprocessor.get_feature_names())}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
    else:
        print(f"Data file not found: {data_file}")
        print("Please run data_generator.py first to create the dataset.")