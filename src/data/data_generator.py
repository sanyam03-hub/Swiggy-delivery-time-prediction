"""
Synthetic data generator for Swiggy delivery time prediction.
This module generates realistic delivery data with various features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import SYNTHETIC_DATA_DIR, DATASET_SIZE, RANDOM_SEED, CITIES_CONFIG
from src.utils.helpers import calculate_distance, get_time_features
from src.utils.logger import setup_logger

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

logger = setup_logger(__name__)

class DeliveryDataGenerator:
    """Generate synthetic delivery data with realistic patterns."""
    
    def __init__(self, num_samples: int = DATASET_SIZE):
        self.num_samples = num_samples
        self.food_categories = [
            "Indian", "Chinese", "Italian", "Mexican", "Thai", 
            "Japanese", "Mediterranean", "American", "Continental", "South Indian"
        ]
        self.weather_conditions = [
            "Clear", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorm", "Fog"
        ]
        self.restaurant_types = ["Fast Food", "Casual Dining", "Fine Dining", "Cloud Kitchen"]
        
    def generate_location_data(self) -> pd.DataFrame:
        """Generate restaurant and customer locations."""
        data = []
        
        for _ in range(self.num_samples):
            # Randomly select a city
            city = random.choice(list(CITIES_CONFIG.keys()))
            city_config = CITIES_CONFIG[city]
            
            # Restaurant location
            restaurant_lat = np.random.uniform(
                city_config["lat_range"][0], 
                city_config["lat_range"][1]
            )
            restaurant_lng = np.random.uniform(
                city_config["lng_range"][0], 
                city_config["lng_range"][1]
            )
            
            # Customer location (within delivery radius of restaurant)
            # Most customers are within 5km radius
            distance_km = np.random.exponential(2.5)  # Exponential distribution, most within 2-3 km
            distance_km = min(distance_km, 15)  # Cap at 15km
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Convert to lat/lng offset (approximate)
            lat_offset = (distance_km / 111.32) * np.cos(angle)  # 111.32 km per degree latitude
            lng_offset = (distance_km / (111.32 * np.cos(np.radians(restaurant_lat)))) * np.sin(angle)
            
            customer_lat = restaurant_lat + lat_offset
            customer_lng = restaurant_lng + lng_offset
            
            data.append({
                "city": city,
                "restaurant_lat": round(restaurant_lat, 6),
                "restaurant_lng": round(restaurant_lng, 6),
                "customer_lat": round(customer_lat, 6),
                "customer_lng": round(customer_lng, 6),
                "distance_km": round(distance_km, 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_time_data(self) -> pd.DataFrame:
        """Generate order time data."""
        data = []
        
        # Generate random timestamps over the last 6 months
        start_date = datetime.now() - timedelta(days=180)
        
        for _ in range(self.num_samples):
            # Random timestamp
            random_days = np.random.randint(0, 180)
            random_hours = np.random.randint(0, 24)
            random_minutes = np.random.randint(0, 60)
            
            order_time = start_date + timedelta(
                days=random_days, 
                hours=random_hours, 
                minutes=random_minutes
            )
            
            # Extract time features
            time_features = get_time_features(order_time)
            
            data.append({
                "order_timestamp": order_time,
                "order_hour": time_features["hour"],
                "day_of_week": time_features["day_of_week"],
                "is_weekend": time_features["is_weekend"],
                "is_peak_hour": time_features["is_peak_hour"],
                "is_lunch": time_features["is_lunch"],
                "is_dinner": time_features["is_dinner"]
            })
        
        return pd.DataFrame(data)
    
    def generate_restaurant_data(self) -> pd.DataFrame:
        """Generate restaurant-related features."""
        data = []
        
        for _ in range(self.num_samples):
            food_category = random.choice(self.food_categories)
            restaurant_type = random.choice(self.restaurant_types)
            
            # Restaurant rating (biased towards higher ratings)
            rating = np.random.beta(2, 0.5) * 4 + 1  # Beta distribution skewed towards higher values
            rating = min(5.0, max(1.0, rating))
            
            # Preparation time based on restaurant type and food category
            base_prep_time = {
                "Fast Food": np.random.normal(8, 2),
                "Casual Dining": np.random.normal(15, 3),
                "Fine Dining": np.random.normal(25, 5),
                "Cloud Kitchen": np.random.normal(12, 3)
            }[restaurant_type]
            
            prep_time = max(5, base_prep_time)  # Minimum 5 minutes
            
            data.append({
                "food_category": food_category,
                "restaurant_type": restaurant_type,
                "restaurant_rating": round(rating, 1),
                "preparation_time": round(prep_time, 1)
            })
        
        return pd.DataFrame(data)
    
    def generate_order_data(self) -> pd.DataFrame:
        """Generate order-specific features."""
        data = []
        
        for _ in range(self.num_samples):
            # Order value (log-normal distribution)
            order_value = np.random.lognormal(mean=5.5, sigma=0.8)  # Mean around 250-300
            order_value = max(50, min(2000, order_value))  # Between 50 and 2000
            
            # Number of items
            num_items = np.random.poisson(2) + 1  # Poisson with mean 2, minimum 1
            num_items = min(10, num_items)  # Maximum 10 items
            
            data.append({
                "order_value": round(order_value, 2),
                "num_items": num_items
            })
        
        return pd.DataFrame(data)
    
    def generate_external_factors(self) -> pd.DataFrame:
        """Generate weather and traffic data."""
        data = []
        
        for _ in range(self.num_samples):
            # Weather condition
            weather = random.choice(self.weather_conditions)
            
            # Temperature (realistic for Indian cities)
            base_temp = np.random.normal(28, 8)  # Mean 28°C, std 8°C
            temperature = max(10, min(45, base_temp))
            
            # Traffic density (0 = no traffic, 1 = heavy traffic)
            # Higher during peak hours
            base_traffic = np.random.beta(2, 3)  # Slightly skewed towards lower traffic
            traffic_density = max(0.1, min(1.0, base_traffic))
            
            data.append({
                "weather_condition": weather,
                "temperature": round(temperature, 1),
                "traffic_density": round(traffic_density, 2)
            })
        
        return pd.DataFrame(data)
    
    def calculate_delivery_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate realistic delivery time based on features."""
        df = df.copy()
        
        # Base delivery time formula
        base_time = (
            5 +  # Base time
            df['distance_km'] * 2.5 +  # 2.5 minutes per km
            df['preparation_time'] +
            df['traffic_density'] * 15 +  # Traffic impact
            (df['weather_condition'] == 'Heavy Rain').astype(int) * 10 +  # Rain delay
            (df['weather_condition'] == 'Thunderstorm').astype(int) * 15 +  # Storm delay
            (df['is_peak_hour']).astype(int) * 8 +  # Peak hour delay
            (df['num_items'] - 1) * 2  # Additional items delay
        )
        
        # Restaurant efficiency based on rating
        efficiency_factor = (df['restaurant_rating'] - 1) / 4  # 0 to 1 scale
        base_time = base_time * (1.3 - 0.3 * efficiency_factor)  # Better restaurants are faster
        
        # Add some random noise
        noise = np.random.normal(0, 3, len(df))
        delivery_time = base_time + noise
        
        # Ensure minimum delivery time
        delivery_time = np.maximum(delivery_time, 15)  # Minimum 15 minutes
        
        df['delivery_time_minutes'] = delivery_time.round(1)
        
        return df
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete synthetic dataset."""
        logger.info(f"Generating synthetic dataset with {self.num_samples} samples...")
        
        # Generate all components
        location_df = self.generate_location_data()
        time_df = self.generate_time_data()
        restaurant_df = self.generate_restaurant_data()
        order_df = self.generate_order_data()
        external_df = self.generate_external_factors()
        
        # Combine all dataframes
        df = pd.concat([location_df, time_df, restaurant_df, order_df, external_df], axis=1)
        
        # Calculate delivery time
        df = self.calculate_delivery_time(df)
        
        # Add order ID
        df['order_id'] = [f"ORD_{i:06d}" for i in range(len(df))]
        
        # Reorder columns
        column_order = [
            'order_id', 'city', 'order_timestamp', 
            'restaurant_lat', 'restaurant_lng', 'customer_lat', 'customer_lng',
            'distance_km', 'order_hour', 'day_of_week', 'is_weekend',
            'is_peak_hour', 'is_lunch', 'is_dinner',
            'food_category', 'restaurant_type', 'restaurant_rating',
            'preparation_time', 'order_value', 'num_items',
            'weather_condition', 'temperature', 'traffic_density',
            'delivery_time_minutes'
        ]
        
        df = df[column_order]
        
        logger.info("Dataset generation completed!")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Average delivery time: {df['delivery_time_minutes'].mean():.1f} minutes")
        logger.info(f"Delivery time range: {df['delivery_time_minutes'].min():.1f} - {df['delivery_time_minutes'].max():.1f} minutes")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "swiggy_delivery_data.csv") -> str:
        """Save dataset to CSV file."""
        # Ensure directory exists
        SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        file_path = SYNTHETIC_DATA_DIR / filename
        df.to_csv(file_path, index=False)
        
        logger.info(f"Dataset saved to: {file_path}")
        return str(file_path)

def main():
    """Main function to generate and save synthetic data."""
    generator = DeliveryDataGenerator(num_samples=DATASET_SIZE)
    
    # Generate dataset
    df = generator.generate_complete_dataset()
    
    # Save dataset
    file_path = generator.save_dataset(df)
    
    # Display basic statistics
    print("\n" + "="*50)
    print("SYNTHETIC DATASET SUMMARY")
    print("="*50)
    print(f"Total records: {len(df):,}")
    print(f"File saved: {file_path}")
    print(f"\nDelivery Time Statistics:")
    print(f"Mean: {df['delivery_time_minutes'].mean():.1f} minutes")
    print(f"Median: {df['delivery_time_minutes'].median():.1f} minutes")
    print(f"Std Dev: {df['delivery_time_minutes'].std():.1f} minutes")
    print(f"Min: {df['delivery_time_minutes'].min():.1f} minutes")
    print(f"Max: {df['delivery_time_minutes'].max():.1f} minutes")
    
    print(f"\nCities distribution:")
    print(df['city'].value_counts())
    
    print(f"\nFood categories distribution:")
    print(df['food_category'].value_counts())
    
    return df

if __name__ == "__main__":
    df = main()