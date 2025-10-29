"""
Machine Learning Models Module
Handles model training, hyperparameter tuning, and evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.config import model_config, MODELS_DIR, METRICS_DIR


class ModelTrainer:
    """Train and optimize CART-based models"""

    def __init__(
        self,
        model_type: str = "random_forest",
        use_tuning: bool = True,
        tuning_method: str = "grid"
    ):
        """
        Initialize ModelTrainer

        Args:
            model_type: "decision_tree" or "random_forest"
            use_tuning: Whether to use hyperparameter tuning
            tuning_method: "grid" for GridSearchCV or "random" for RandomizedSearchCV
        """
        self.model_type = model_type
        self.use_tuning = use_tuning
        self.tuning_method = tuning_method
        self.model = None
        self.best_params = None
        self.cv_results = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train model with optional hyperparameter tuning

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional, for evaluation)
            y_test: Test target (optional, for evaluation)

        Returns:
            Dictionary with training results
        """
        print(f"\n=== Training {self.model_type.replace('_', ' ').title()} ===")

        # Initialize base model
        base_model = self._get_base_model()

        if self.use_tuning:
            # Hyperparameter tuning
            print(f"🔍 Performing {self.tuning_method} search for hyperparameters...")
            self.model = self._tune_hyperparameters(base_model, X_train, y_train)
        else:
            # Train with default parameters
            print("Training with default parameters...")
            base_model.fit(X_train, y_train)
            self.model = base_model

        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        results = {
            "model_type": self.model_type,
            "train_accuracy": train_accuracy,
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0]
        }

        print(f"  ✓ Training accuracy: {train_accuracy:.4f}")

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)
            results["test_accuracy"] = test_accuracy
            results["n_test_samples"] = X_test.shape[0]

            print(f"  ✓ Test accuracy: {test_accuracy:.4f}")

            # Check for overfitting
            if train_accuracy - test_accuracy > 0.15:
                print(f"  ⚠️  Potential overfitting detected!")
                print(f"     Train-test gap: {(train_accuracy - test_accuracy):.4f}")

        # Store best parameters if tuning was used
        if self.use_tuning and hasattr(self.model, 'best_params_'):
            self.best_params = self.model.best_params_
            results["best_params"] = self.best_params
            print(f"\n  Best parameters found:")
            for param, value in self.best_params.items():
                print(f"    • {param}: {value}")

        return results

    def _get_base_model(self):
        """Get base model instance"""
        if self.model_type == "decision_tree":
            return DecisionTreeClassifier(random_state=model_config.random_state)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(random_state=model_config.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _tune_hyperparameters(
        self,
        base_model,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """
        Perform hyperparameter tuning

        Args:
            base_model: Base model instance
            X_train: Training features
            y_train: Training target

        Returns:
            Fitted search object (GridSearchCV or RandomizedSearchCV)
        """
        # Get parameter grid
        if self.model_type == "decision_tree":
            param_grid = model_config.dt_param_grid
        elif self.model_type == "random_forest":
            param_grid = model_config.rf_param_grid
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=model_config.cv_folds,
            shuffle=True,
            random_state=model_config.random_state
        )

        # Perform search
        if self.tuning_method == "grid":
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=model_config.cv_scoring,
                n_jobs=-1,
                verbose=1
            )
        elif self.tuning_method == "random":
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=model_config.n_iter_random_search,
                cv=cv,
                scoring=model_config.cv_scoring,
                n_jobs=-1,
                random_state=model_config.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown tuning method: {self.tuning_method}")

        # Fit search
        search.fit(X_train, y_train)

        # Store results
        self.cv_results = pd.DataFrame(search.cv_results_)

        print(f"  ✓ Best CV score: {search.best_score_:.4f}")

        return search

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = None
    ) -> Dict[str, float]:
        """
        Perform cross-validation

        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        cv_folds = cv_folds or model_config.cv_folds

        print(f"\n🔄 Performing {cv_folds}-fold cross-validation...")

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=model_config.random_state
        )

        # Get model for CV (use best model if available)
        if self.model is not None and hasattr(self.model, 'best_estimator_'):
            cv_model = self.model.best_estimator_
        elif self.model is not None:
            cv_model = self.model
        else:
            cv_model = self._get_base_model()

        # Perform CV
        cv_scores = cross_val_score(
            cv_model,
            X,
            y,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        results = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist()
        }

        print(f"  ✓ CV Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")

        return results

    def get_feature_importance(
        self,
        feature_names: list,
        top_n: int = None
    ) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get the actual model (might be wrapped in GridSearchCV)
        if hasattr(self.model, 'best_estimator_'):
            model = self.model.best_estimator_
        else:
            model = self.model

        # Get feature importance
        importance = model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save_model(self, filepath: Optional[Path] = None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_type}_model.pkl"

        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")

        # Save best parameters if available
        if self.best_params:
            params_file = filepath.parent / f"{filepath.stem}_params.json"
            with open(params_file, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            print(f"✓ Best parameters saved to {params_file}")

    def load_model(self, filepath: Optional[Path] = None):
        """Load trained model"""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.model_type}_model.pkl"

        self.model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")

        # Load best parameters if available
        params_file = filepath.parent / f"{filepath.stem}_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)
            print(f"✓ Best parameters loaded from {params_file}")


class ModelComparison:
    """Compare multiple models"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name: str, trainer: ModelTrainer):
        """Add a trained model to comparison"""
        self.models[name] = trainer

    def compare(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare all models

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with comparison results
        """
        print("\n=== Model Comparison ===")

        comparison_results = []

        for name, trainer in self.models.items():
            if trainer.model is None:
                print(f"⚠️  Model '{name}' not trained, skipping...")
                continue

            # Get predictions
            y_pred = trainer.model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            comparison_results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })

        results_df = pd.DataFrame(comparison_results)
        results_df = results_df.sort_values('Accuracy', ascending=False)

        print("\n" + results_df.to_string(index=False))

        # Save comparison
        results_file = METRICS_DIR / "model_comparison.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Comparison results saved to {results_file}")

        return results_df


if __name__ == "__main__":
    print("Models module - Run train.py to train models")
