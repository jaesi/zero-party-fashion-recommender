"""
Configuration file for Zero-Party Fashion Recommender
Centralized settings for data paths, model parameters, and processing options
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "rawdata_무채색유채색 통합.xlsx"


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""

    # Target variable
    target_column: str = "target_group"

    # Feature categories
    demographic_features: List[str] = None
    mbti_features: List[str] = None
    psychographic_features: List[str] = None
    sock_preference_features: List[str] = None

    # Missing value handling
    missing_value_strategy: str = "median"  # Options: "zero", "median", "mean", "drop"
    missing_threshold: float = 0.5  # Drop columns with >50% missing values

    # Test split
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        """Initialize feature lists"""
        if self.demographic_features is None:
            self.demographic_features = ["q1", "q2_1", "q3", "q5"]

        if self.mbti_features is None:
            self.mbti_features = ["I_E", "S_N", "T_F", "J_P"]

        if self.psychographic_features is None:
            self.psychographic_features = [
                "개성지향", "과시지향", "운동지향",
                "SNS활동_자기표현", "SNS활동_시간",
                "의복추구혜택_실용성", "의복추구혜택_유행스타일추구",
                "의복추구혜택_외모추구", "패션관여도"
            ]

        if self.sock_preference_features is None:
            self.sock_preference_features = [
                f"q11_{i}" for i in [1, 3, 4, 5, 6, 7, 8]
            ] + [
                f"q12_{i}" for i in [1, 2, 3, 4, 5, 6]
            ]


@dataclass
class ModelConfig:
    """Model training and hyperparameter configuration"""

    # Random state for reproducibility
    random_state: int = 42

    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "accuracy"

    # Decision Tree hyperparameter search space
    dt_param_grid: dict = None

    # Random Forest hyperparameter search space
    rf_param_grid: dict = None

    # Model selection
    use_grid_search: bool = True
    use_random_search: bool = False
    n_iter_random_search: int = 20

    # Model saving
    save_best_model: bool = True
    model_format: str = "joblib"  # Options: "joblib", "pickle"

    def __post_init__(self):
        """Initialize hyperparameter grids"""
        if self.dt_param_grid is None:
            self.dt_param_grid = {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
                "class_weight": ["balanced", None]
            }

        if self.rf_param_grid is None:
            self.rf_param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced", None],
                "bootstrap": [True, False]
            }


@dataclass
class VisualizationConfig:
    """Visualization settings"""

    # Figure settings
    figure_dpi: int = 300
    figure_format: str = "png"

    # Tree visualization
    tree_max_depth: Optional[int] = 5  # None for full tree
    tree_figsize: tuple = (20, 12)

    # Feature importance
    top_n_features: int = 15
    feature_importance_figsize: tuple = (12, 8)

    # Confusion matrix
    confusion_matrix_figsize: tuple = (10, 8)
    confusion_matrix_cmap: str = "Blues"

    # Font settings (for Korean support)
    font_family: str = "DejaVu Sans"  # Use system default
    font_size: int = 10


# Global configuration instances
data_config = DataConfig()
model_config = ModelConfig()
viz_config = VisualizationConfig()


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        METRICS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✓ All directories created successfully")


if __name__ == "__main__":
    # Test configuration
    create_directories()
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"\nData Config: {data_config}")
    print(f"\nModel Config: {model_config}")
