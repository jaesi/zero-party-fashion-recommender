"""
Zero-Party Fashion Recommender
CART-based fashion recommendation using MBTI and zero-party data
"""

__version__ = "1.0.0"
__author__ = "Fashion Recommender Team"

from src.config import data_config, model_config, viz_config
from src.preprocessing import load_and_preprocess_data, DataLoader, DataCleaner
from src.feature_engineering import prepare_features, FeatureEngineer
from src.models import ModelTrainer, ModelComparison
from src.evaluation import ModelEvaluator, ErrorAnalysis
from src.visualization import Visualizer, create_all_visualizations

__all__ = [
    # Configuration
    "data_config",
    "model_config",
    "viz_config",
    # Preprocessing
    "load_and_preprocess_data",
    "DataLoader",
    "DataCleaner",
    # Feature Engineering
    "prepare_features",
    "FeatureEngineer",
    # Modeling
    "ModelTrainer",
    "ModelComparison",
    # Evaluation
    "ModelEvaluator",
    "ErrorAnalysis",
    # Visualization
    "Visualizer",
    "create_all_visualizations",
]
