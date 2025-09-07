"""
Streamlit dashboard for Swiggy delivery time prediction.
Interactive web interface for making predictions and exploring data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import requests
import json
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import DataPreprocessor
from src.utils.config import MODELS_DIR, SYNTHETIC_DATA_DIR
from src.utils.helpers import calculate_distance
from src.utils.logger import setup_logger

# Configure Streamlit page
st.set_page_config(
    page_title="🚚 Swiggy Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #FF6B35, #FF8E53);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF6B35;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background-color: #e55a2b;
    }
    
    /* Enhanced scroll handling */
    .main .block-container {
        max-height: 100vh;
        overflow-y: auto;
        scroll-behavior: smooth;
    }
    
    /* Better slider container styling */
    .stSlider {
        margin: 10px 0;
        padding: 5px;
    }
    
    /* Improved selectbox styling */
    .stSelectbox {
        margin: 10px 0;
        padding: 5px;
    }
    
    /* Container improvements */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="element-container"] {
        margin-bottom: 1rem;
    }
    
    /* Enhanced form container */
    .stContainer {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    
    /* Fix for slider rendering issues */
    .stSlider > div > div {
        background-color: transparent !important;
    }
    
    /* Mobile responsiveness for sliders */
    @media (max-width: 768px) {
        .stSlider {
            padding: 10px 0;
        }
    }
    
    /* Force re-render sliders to fix scroll issues */
    .stSlider [data-baseweb="slider"] {
        touch-action: pan-x !important;
    }
</style>

<script>
// JavaScript to fix slider scroll issues
setTimeout(() => {
    // Force re-render of sliders
    const sliders = document.querySelectorAll('.stSlider');
    sliders.forEach(slider => {
        slider.style.transform = 'translateZ(0)';
        slider.style.willChange = 'transform';
    });
    
    // Ensure sliders are properly interactive
    const sliderInputs = document.querySelectorAll('.stSlider input');
    sliderInputs.forEach(input => {
        input.style.pointerEvents = 'auto';
        input.addEventListener('touchstart', function(e) {
            e.stopPropagation();
        });
        input.addEventListener('touchmove', function(e) {
            e.stopPropagation();
        });
    });
}, 1000);
</script>
""", unsafe_allow_html=True)

# Global variables
@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor with caching."""
    try:
        # Load preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(str(preprocessor_path))
        else:
            st.error(f"Preprocessor not found at {preprocessor_path}")
            return None, None
        
        # Try to load the best model first
        model_files = [
            "best_model_lightgbm.joblib",
            "best_model_random_forest.joblib",
            "random_forest_model.joblib",
            "lightgbm_model.joblib",
            "xgboost_model.joblib",
            "linear_regression_model.joblib"
        ]
        
        for model_file in model_files:
            model_path = MODELS_DIR / model_file
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    model = model_data['model']
                    
                    # Verify that model and preprocessor are compatible
                    feature_names = preprocessor.get_feature_names()
                    if hasattr(model, 'feature_importances_') and len(model.feature_importances_) > 0:
                        if len(feature_names) != len(model.feature_importances_):
                            st.warning(f"Feature count mismatch in {model_file}: {len(feature_names)} vs {len(model.feature_importances_)}")
                    
                    return model, preprocessor
                except Exception as e:
                    st.warning(f"Error loading {model_file}: {str(e)}")
                    continue
        
        st.error("No valid model found")
        return None, None
        
    except Exception as e:
        st.error(f"Error loading model and preprocessor: {str(e)}")
        return None, None

@st.cache_data
def load_sample_data():
    """Load sample data for visualization."""
    try:
        data_file = SYNTHETIC_DATA_DIR / "swiggy_delivery_data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            return df.sample(1000)  # Sample for performance
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def predict_delivery_time(model, preprocessor, features):
    """Make prediction using loaded model."""
    try:
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        df_processed = preprocessor.prepare_data(df, fit=False)
        
        # Get feature names and check if they exist in processed data
        expected_features = preprocessor.get_feature_names()
        available_features = [col for col in expected_features if col in df_processed.columns]
        
        if len(available_features) != len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            st.warning(f"Missing features in processed data: {missing_features}")
            
            # Try to create missing time features if possible
            if 'order_hour' in df_processed.columns:
                if 'hour_sin' not in df_processed.columns:
                    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['order_hour'] / 24)
                if 'hour_cos' not in df_processed.columns:
                    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['order_hour'] / 24)
                    
            if 'day_of_week' in df_processed.columns:
                if 'day_sin' not in df_processed.columns:
                    df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
                if 'day_cos' not in df_processed.columns:
                    df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
            
            # Add month and quarter if missing
            if 'month' not in df_processed.columns:
                from datetime import datetime
                df_processed['month'] = datetime.now().month
            if 'quarter' not in df_processed.columns:
                from datetime import datetime
                df_processed['quarter'] = (datetime.now().month - 1) // 3 + 1
                
            # Add traffic_weather_impact if missing
            if 'traffic_weather_impact' not in df_processed.columns:
                if 'traffic_density' in df_processed.columns:
                    df_processed['traffic_weather_impact'] = (
                        df_processed['traffic_density'] > 0.7
                    ).astype(int)
                else:
                    df_processed['traffic_weather_impact'] = 0
            
            # Update available features
            available_features = [col for col in expected_features if col in df_processed.columns]
        
        # Select features for prediction
        if len(available_features) == len(expected_features):
            X = df_processed[expected_features]
        else:
            # Use only available features and warn user
            X = df_processed[available_features]
            st.error(f"Still missing features: {set(expected_features) - set(available_features)}")
            return None
        
        # Predict
        prediction = model.predict(X)[0]
        
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚚 Swiggy Delivery Time Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict food delivery times using machine learning")
    
    # Load model and data
    model, preprocessor = load_model_and_preprocessor()
    sample_data = load_sample_data()
    
    if model is None or preprocessor is None:
        st.error("❌ Model not loaded. Please check if models are trained and saved.")
        st.info("Run the model training script first: `python src/models/simple_trainer.py`")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "🎯 Prediction", 
        "📊 Data Exploration", 
        "📈 Model Performance",
        "🔍 Batch Prediction"
    ])
    
    if page == "🎯 Prediction":
        prediction_page(model, preprocessor)
    elif page == "📊 Data Exploration":
        data_exploration_page(sample_data)
    elif page == "📈 Model Performance":
        model_performance_page(sample_data, model, preprocessor)
    elif page == "🔍 Batch Prediction":
        batch_prediction_page(model, preprocessor)

def prediction_page(model, preprocessor):
    """Single prediction page."""
    st.header("🎯 Make a Delivery Time Prediction")
    
    # Create a scrollable form container
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            # Location section in an expandable container
            with st.expander("📍 Location Details", expanded=True):
                
                # Restaurant location
                st.write("**Restaurant Location**")
                restaurant_lat = st.number_input("Restaurant Latitude", value=12.9716, step=0.0001, format="%.4f", key="rest_lat")
                restaurant_lng = st.number_input("Restaurant Longitude", value=77.5946, step=0.0001, format="%.4f", key="rest_lng")
                
                # Customer location
                st.write("**Customer Location**")
                customer_lat = st.number_input("Customer Latitude", value=12.9500, step=0.0001, format="%.4f", key="cust_lat")
                customer_lng = st.number_input("Customer Longitude", value=77.6000, step=0.0001, format="%.4f", key="cust_lng")
                
                # Calculate and display distance
                distance = calculate_distance(restaurant_lat, restaurant_lng, customer_lat, customer_lng)
                st.metric("📏 Distance", f"{distance:.2f} km")
            
            # Time & Weather section in an expandable container
            with st.expander("🕐 Time & Weather", expanded=True):
                
                # Use containers for better scroll handling
                with st.container():
                    order_hour = st.slider("Order Hour", 0, 23, 14, key="order_hour_slider", help="Select the hour when the order is placed (0-23)")
                
                with st.container():
                    day_of_week = st.selectbox("Day of Week", 
                                              options=[0, 1, 2, 3, 4, 5, 6],
                                              format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                                              key="day_of_week_select")
                
                with st.container():
                    weather_condition = st.selectbox("Weather Condition", 
                                                   ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorm", "Fog"],
                                                   key="weather_select")
                
                with st.container():
                    temperature = st.slider("Temperature (°C)", 10, 45, 25, key="temperature_slider", help="Current temperature in Celsius")
                
                with st.container():
                    traffic_density = st.slider("Traffic Density", 0.0, 1.0, 0.5, 0.1, key="traffic_slider", help="Traffic density from 0 (no traffic) to 1 (heavy traffic)")
        
        with col2:
            # Restaurant & Order Details section in an expandable container
            with st.expander("🍽️ Restaurant & Order Details", expanded=True):
                
                # Use containers for better scroll handling
                with st.container():
                    restaurant_rating = st.slider("Restaurant Rating", 1.0, 5.0, 4.0, 0.1, key="rating_slider", help="Restaurant rating from 1 to 5 stars")
                
                with st.container():
                    restaurant_type = st.selectbox("Restaurant Type", 
                                                 ["Fast Food", "Casual Dining", "Fine Dining", "Cloud Kitchen"],
                                                 key="restaurant_type_select")
                
                with st.container():
                    food_category = st.selectbox("Food Category", 
                                               ["Indian", "Chinese", "Italian", "Mexican", "Thai", "Japanese", 
                                                "Mediterranean", "American", "Continental", "South Indian"],
                                               key="food_category_select")
                
                with st.container():
                    order_value = st.number_input("Order Value (₹)", min_value=50, max_value=2000, value=300, step=50, key="order_value_input")
                
                with st.container():
                    num_items = st.number_input("Number of Items", min_value=1, max_value=10, value=2, key="num_items_input")
                
                with st.container():
                    preparation_time = st.number_input("Preparation Time (minutes)", min_value=5, max_value=60, value=15, key="prep_time_input")
                
                with st.container():
                    city = st.selectbox("City", ["bangalore", "mumbai", "delhi"], key="city_select")
            
            # Prediction button in a separate container
            with st.container():
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                predict_button = st.button("🚀 Predict Delivery Time", type="primary", key="predict_btn")
        
        # Handle prediction when button is clicked
        if predict_button:
            # Prepare features
            features = {
                'restaurant_lat': restaurant_lat,
                'restaurant_lng': restaurant_lng,
                'customer_lat': customer_lat,
                'customer_lng': customer_lng,
                'distance_km': distance,
                'order_hour': order_hour,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5,
                'weather_condition': weather_condition,
                'temperature': temperature,
                'traffic_density': traffic_density,
                'restaurant_rating': restaurant_rating,
                'restaurant_type': restaurant_type,
                'food_category': food_category,
                'order_value': order_value,
                'num_items': num_items,
                'preparation_time': preparation_time,
                'city': city
            }
            
            # Make prediction
            prediction = predict_delivery_time(model, preprocessor, features)
            
            if prediction is not None:
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Predicted Delivery Time</h2>
                    <h1>{prediction:.1f} minutes</h1>
                    <p>Estimated arrival: {(datetime.now() + timedelta(minutes=prediction)).strftime('%H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    if prediction <= 30:
                        st.success("🟢 Fast delivery expected")
                    elif prediction <= 45:
                        st.warning("🟡 Moderate delivery time")
                    else:
                        st.error("🔴 Longer delivery time")
                
                with col4:
                    if traffic_density > 0.7:
                        st.info("🚦 High traffic may cause delays")
                    elif weather_condition in ["Heavy Rain", "Thunderstorm"]:
                        st.info("🌧️ Weather may affect delivery")
                    else:
                        st.info("✅ Good delivery conditions")
                
                with col5:
                    if distance > 10:
                        st.info("📍 Long distance delivery")
                    elif distance < 2:
                        st.info("📍 Very close delivery")
                    else:
                        st.info("📍 Standard delivery distance")

def data_exploration_page(sample_data):
    """Data exploration page."""
    st.header("📊 Data Exploration")
    
    if sample_data is None:
        st.error("No data available for exploration")
        return
    
    # Basic statistics
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", len(sample_data))
    with col2:
        st.metric("Avg Delivery Time", f"{sample_data['delivery_time_minutes'].mean():.1f} min")
    with col3:
        st.metric("Avg Distance", f"{sample_data['distance_km'].mean():.1f} km")
    with col4:
        st.metric("Avg Order Value", f"₹{sample_data['order_value'].mean():.0f}")
    
    # Charts
    st.subheader("📊 Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Delivery Times", "Geography", "Time Patterns", "Order Analysis"])
    
    with tab1:
        # Delivery time distribution
        fig = px.histogram(sample_data, x='delivery_time_minutes', nbins=30,
                          title='Distribution of Delivery Times')
        st.plotly_chart(fig, use_container_width=True)
        
        # Delivery time by factors
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(sample_data, x='weather_condition', y='delivery_time_minutes',
                        title='Delivery Time by Weather')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(sample_data, x='city', y='delivery_time_minutes',
                        title='Delivery Time by City')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Geographic distribution
        fig = px.scatter_mapbox(sample_data.head(100), 
                               lat='restaurant_lat', lon='restaurant_lng',
                               hover_data=['delivery_time_minutes', 'city'],
                               title='Restaurant Locations',
                               mapbox_style='open-street-map',
                               height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Time patterns
        col1, col2 = st.columns(2)
        with col1:
            hourly_avg = sample_data.groupby('order_hour')['delivery_time_minutes'].mean().reset_index()
            fig = px.line(hourly_avg, x='order_hour', y='delivery_time_minutes',
                         title='Average Delivery Time by Hour')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            daily_avg = sample_data.groupby('day_of_week')['delivery_time_minutes'].mean().reset_index()
            daily_avg['day_name'] = daily_avg['day_of_week'].map({
                0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
            })
            fig = px.bar(daily_avg, x='day_name', y='delivery_time_minutes',
                        title='Average Delivery Time by Day')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Order analysis
        col1, col2 = st.columns(2)
        with col1:
            category_counts = sample_data['food_category'].value_counts().head(10)
            fig = px.pie(values=category_counts.values, names=category_counts.index,
                        title='Food Category Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(sample_data, x='order_value', y='delivery_time_minutes',
                           color='num_items', title='Order Value vs Delivery Time')
            st.plotly_chart(fig, use_container_width=True)

def model_performance_page(sample_data, model, preprocessor):
    """Model performance analysis page."""
    st.header("📈 Model Performance Analysis")
    
    if sample_data is None:
        st.error("No data available for performance analysis")
        return
    
    if model is None or preprocessor is None:
        st.error("Model or preprocessor not loaded")
        return
    
    st.subheader("🎯 Feature Importance")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = preprocessor.get_feature_names()
        feature_importances = model.feature_importances_
        
        # Check if lengths match
        if len(feature_names) == len(feature_importances):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importances
            }).sort_values('importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Feature names length ({len(feature_names)}) doesn't match feature importances length ({len(feature_importances)})")
            st.info("Displaying feature importances with generic names")
            
            # Create generic feature names
            generic_names = [f"Feature_{i+1}" for i in range(len(feature_importances))]
            importance_df = pd.DataFrame({
                'feature': generic_names,
                'importance': feature_importances
            }).sort_values('importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title='Top 10 Most Important Features (Generic Names)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type")
    
    # Model predictions vs actual (on sample data)
    st.subheader("🎯 Prediction Accuracy")
    
    # Take a smaller sample for performance
    test_sample = sample_data.sample(min(100, len(sample_data)))
    
    try:
        predictions = []
        actuals = []
        
        progress_bar = st.progress(0)
        
        for idx, row in test_sample.iterrows():
            # Prepare features
            features = {
                'restaurant_lat': row['restaurant_lat'],
                'restaurant_lng': row['restaurant_lng'],
                'customer_lat': row['customer_lat'],
                'customer_lng': row['customer_lng'],
                'distance_km': row['distance_km'],
                'order_hour': row['order_hour'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'weather_condition': row['weather_condition'],
                'temperature': row['temperature'],
                'traffic_density': row['traffic_density'],
                'restaurant_rating': row['restaurant_rating'],
                'restaurant_type': row['restaurant_type'],
                'food_category': row['food_category'],
                'order_value': row['order_value'],
                'num_items': row['num_items'],
                'preparation_time': row['preparation_time'],
                'city': row['city']
            }
            
            pred = predict_delivery_time(model, preprocessor, features)
            if pred is not None:
                predictions.append(pred)
                actuals.append(row['delivery_time_minutes'])
            
            progress_bar.progress((len(predictions)) / len(test_sample))
        
        if predictions:
            # Create scatter plot
            fig = px.scatter(x=actuals, y=predictions, 
                           title='Predicted vs Actual Delivery Times',
                           labels={'x': 'Actual (minutes)', 'y': 'Predicted (minutes)'})
            
            # Add perfect prediction line
            min_val = min(min(actuals), min(predictions))
            max_val = max(max(actuals), max(predictions))
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate metrics
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.2f} min")
            with col2:
                st.metric("Root Mean Square Error", f"{rmse:.2f} min")
            with col3:
                st.metric("Sample Size", len(predictions))
    
    except Exception as e:
        st.error(f"Error in performance analysis: {str(e)}")

def batch_prediction_page(model, preprocessor):
    """Batch prediction page."""
    st.header("🔍 Batch Prediction")
    st.write("Upload a CSV file to predict delivery times for multiple orders")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Required columns
            required_cols = [
                'restaurant_lat', 'restaurant_lng', 'customer_lat', 'customer_lng',
                'order_hour', 'day_of_week', 'weather_condition', 'traffic_density'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.write("Required columns:", required_cols)
            else:
                if st.button("🚀 Predict All"):
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, (idx, row) in enumerate(df.iterrows()):
                        # Add default values for missing optional columns with proper type handling
                        temp_val = row.get('temperature', 25.0)
                        rating_val = row.get('restaurant_rating', 4.0)
                        order_val = row.get('order_value', 300.0)
                        num_items_val = row.get('num_items', 2)
                        prep_time_val = row.get('preparation_time', 15.0)
                        
                        features = {
                            'restaurant_lat': float(row['restaurant_lat']),
                            'restaurant_lng': float(row['restaurant_lng']),
                            'customer_lat': float(row['customer_lat']),
                            'customer_lng': float(row['customer_lng']),
                            'distance_km': calculate_distance(
                                float(row['restaurant_lat']), float(row['restaurant_lng']),
                                float(row['customer_lat']), float(row['customer_lng'])
                            ),
                            'order_hour': int(row['order_hour']),
                            'day_of_week': int(row['day_of_week']),
                            'is_weekend': bool(int(row['day_of_week']) >= 5),
                            'weather_condition': str(row['weather_condition']),
                            'temperature': float(temp_val if temp_val is not None else 25.0),
                            'traffic_density': float(row['traffic_density']),
                            'restaurant_rating': float(rating_val if rating_val is not None else 4.0),
                            'restaurant_type': str(row.get('restaurant_type', 'Casual Dining')),
                            'food_category': str(row.get('food_category', 'Indian')),
                            'order_value': float(order_val if order_val is not None else 300.0),
                            'num_items': int(num_items_val if num_items_val is not None else 2),
                            'preparation_time': float(prep_time_val if prep_time_val is not None else 15.0),
                            'city': str(row.get('city', 'bangalore'))
                        }
                        
                        pred = predict_delivery_time(model, preprocessor, features)
                        predictions.append(pred if pred is not None else np.nan)
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Add predictions to dataframe with proper type handling
                    df['predicted_delivery_time'] = pd.Series(predictions, dtype='float64')
                    
                    st.write("### Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="delivery_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.write("### Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        avg_time = np.nanmean([p for p in predictions if p is not None])
                        st.metric("Average Delivery Time", f"{avg_time:.1f} min")
                    with col3:
                        max_time = np.nanmax([p for p in predictions if p is not None])
                        st.metric("Max Delivery Time", f"{max_time:.1f} min")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample format
        st.write("### Sample CSV Format")
        sample_data = {
            'restaurant_lat': [12.9716, 19.0760],
            'restaurant_lng': [77.5946, 72.8777],
            'customer_lat': [12.9500, 19.0896],
            'customer_lng': [77.6000, 72.8656],
            'order_hour': [14, 19],
            'day_of_week': [1, 5],
            'weather_condition': ['Clear', 'Cloudy'],
            'traffic_density': [0.5, 0.8],
            'temperature': [25, 28],
            'restaurant_rating': [4.2, 3.8]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

if __name__ == "__main__":
    main()