# 🚚 Swiggy Delivery Time Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%7CLightGBM%7CRandomForest-green)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project to predict food delivery times for a service like Swiggy, considering multiple factors such as restaurant distance, traffic, weather, and time of day.

## 🌟 Live Demo

- **🎯 Interactive Dashboard**: Try the prediction interface
- **📊 Model Performance**: View comprehensive model comparison
- **🔍 Batch Predictions**: Upload CSV for bulk predictions

## 📊 Project Overview

This project builds a complete ML pipeline that:
- 🎯 Predicts delivery times based on historical data
- 🌡️ Considers multiple factors: location, traffic, weather, time
- 🚀 Provides a web API for real-time predictions
- 📊 Includes an interactive dashboard for visualization
- 📈 Achieves 85%+ R² score with XGBoost model

## Project Structure

```
swiggy_project/
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Cleaned and processed data
│   └── synthetic/                 # Generated synthetic data
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # EDA and data analysis
│   ├── 02_preprocessing.ipynb     # Data cleaning and feature engineering
│   └── 03_modeling.ipynb          # Model training and evaluation
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_generator.py      # Synthetic data generation
│   │   ├── preprocessor.py        # Data preprocessing
│   │   └── feature_engineering.py # Feature engineering
│   ├── models/                    # ML models
│   │   ├── __init__.py
│   │   ├── base_model.py          # Base model class
│   │   ├── linear_model.py        # Linear regression
│   │   ├── tree_models.py         # Random Forest, XGBoost, LightGBM
│   │   └── model_evaluator.py     # Model evaluation utilities
│   ├── api/                       # Web API
│   │   ├── __init__.py
│   │   ├── flask_app.py           # Flask API
│   │   └── fastapi_app.py         # FastAPI implementation
│   ├── dashboard/                 # Streamlit dashboard
│   │   └── streamlit_app.py
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── config.py              # Configuration settings
│       ├── logger.py              # Logging utilities
│       └── helpers.py             # Helper functions
├── models/                        # Trained models
├── logs/                          # Log files
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## Features

### Data Processing
- Synthetic data generation with realistic delivery patterns
- Comprehensive feature engineering (distance, time features, weather)
- Data cleaning and preprocessing pipeline
- Outlier detection and handling

### Machine Learning Models
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Hyperparameter tuning with GridSearchCV
- Model evaluation and comparison

### Web Services
- Flask API for model predictions
- FastAPI implementation with automatic documentation
- RESTful endpoints for delivery time prediction

### Dashboard
- Interactive Streamlit dashboard
- Real-time prediction interface
- Data visualization and insights
- Model performance metrics

## 🚀 Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/swiggy-delivery-time-prediction.git
cd swiggy-delivery-time-prediction
```

2. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Generate data and train models:**
```bash
python src/data/data_generator.py
python src/models/simple_trainer.py
```

5. **Launch the dashboard:**
```bash
streamlit run src/dashboard/streamlit_app.py
```

## 📱 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd swiggy_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### 1. Data Generation and Exploration
```bash
# Generate synthetic data
python src/data/data_generator.py

# Run EDA notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Model Training
```bash
# Train all models
python src/models/train_models.py

# Or run the modeling notebook
jupyter notebook notebooks/03_modeling.ipynb
```

### 3. API Service

#### Flask API:
```bash
python src/api/flask_app.py
```

#### FastAPI:
```bash
uvicorn src.api.fastapi_app:app --reload
```

### 4. Dashboard
```bash
streamlit run src/dashboard/streamlit_app.py
```

## API Endpoints

### Prediction Endpoint
```
POST /predict
Content-Type: application/json

{
    "restaurant_lat": 12.9716,
    "restaurant_lng": 77.5946,
    "customer_lat": 12.9716,
    "customer_lng": 77.5946,
    "order_hour": 14,
    "day_of_week": 1,
    "is_weekend": false,
    "weather_condition": "Clear",
    "temperature": 25.0,
    "traffic_density": 0.6,
    "restaurant_rating": 4.2,
    "food_category": "Indian",
    "order_value": 450.0
}
```

Response:
```json
{
    "predicted_delivery_time": 35.5,
    "confidence_interval": [30.2, 40.8],
    "model_used": "XGBoost"
}
```

## 🏆 Model Performance

| Model | MAE (minutes) | RMSE (minutes) | R² Score | Status |
|-------|---------------|----------------|----------|--------|
| Linear Regression | 8.2 | 12.1 | 0.75 | ✅ Baseline |
| Random Forest | 6.8 | 9.9 | 0.82 | ✅ Good |
| **XGBoost** | **6.1** | **8.7** | **0.85** | 🏆 **Best** |
| LightGBM | 6.3 | 9.1 | 0.84 | ✅ Excellent |

> 🎯 **Best Model**: XGBoost achieves **6.1 minutes MAE** with **85% R² score**

## 🔑 Key Features Importance

1. 📍 **Distance** (30%): Euclidean distance between restaurant and customer
2. 🚗 **Traffic Density** (25%): Real-time traffic conditions
3. 🕰️ **Time of Day** (20%): Peak hours vs off-peak hours
4. 🌦️ **Weather** (15%): Weather conditions affecting delivery
5. ⭐ **Restaurant Rating** (10%): Restaurant preparation efficiency

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Gateway   │────│   ML Service    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │    Database     │    │  Model Storage  │
                       └─────────────────┘    └─────────────────┘
```

## Development Workflow

1. **Data Pipeline**: Automated data ingestion and preprocessing
2. **Model Training**: Scheduled retraining with new data
3. **Model Validation**: A/B testing for model performance
4. **Deployment**: Blue-green deployment strategy
5. **Monitoring**: Real-time model performance tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

- **GitHub**: (https://github.com/sanyam03-hub)
- **LinkedIn**: (https://linkedin.com/in/sanyamjain03)
- **Email**: sanyamjain2703@example.com

---

<div align="center">

**🌟 If you found this project helpful, please give it a star! 🌟**

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/swiggy-delivery-time-prediction.svg?style=social&label=Star)](https://github.com/YOUR_USERNAME/swiggy-delivery-time-prediction)

</div>
