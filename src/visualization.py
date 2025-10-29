"""
Visualization Module
Generate plots and visualizations for model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

from src.config import viz_config, FIGURES_DIR


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = viz_config.figure_dpi
plt.rcParams['font.size'] = viz_config.font_size


class Visualizer:
    """Create visualizations for model analysis"""

    def __init__(self, save_figures: bool = True):
        """
        Initialize Visualizer

        Args:
            save_figures: Whether to save figures to disk
        """
        self.save_figures = save_figures

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_labels: Optional[List] = None,
        model_name: str = "model",
        normalize: bool = False
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap

        Args:
            confusion_matrix: Confusion matrix array
            class_labels: Labels for classes
            model_name: Name of model for title
            normalize: Whether to normalize values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=viz_config.confusion_matrix_figsize)

        # Normalize if requested
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'

        # Create heatmap
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=fmt,
            cmap=viz_config.confusion_matrix_cmap,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"{model_name}_confusion_matrix.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Confusion matrix saved to {filename}")

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str = "model",
        top_n: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot feature importance

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            model_name: Name of model for title
            top_n: Number of top features to show

        Returns:
            Matplotlib figure
        """
        # Select top N features
        top_n = top_n or viz_config.top_n_features
        plot_df = importance_df.head(top_n)

        # Create figure
        fig, ax = plt.subplots(figsize=viz_config.feature_importance_figsize)

        # Create horizontal bar plot
        bars = ax.barh(range(len(plot_df)), plot_df['importance'], color='skyblue')

        # Color the top 3 features differently
        for i in range(min(3, len(bars))):
            bars[i].set_color('coral')

        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['feature'])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}',
                     fontsize=14, fontweight='bold')

        # Add value labels on bars
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.text(row['importance'], i, f" {row['importance']:.4f}",
                   va='center', fontsize=9)

        # Invert y-axis to show highest importance at top
        ax.invert_yaxis()

        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"{model_name}_feature_importance.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Feature importance plot saved to {filename}")

        return fig

    def plot_decision_tree(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        model_name: str = "decision_tree",
        max_depth: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot decision tree

        Args:
            model: Trained decision tree model
            feature_names: List of feature names
            class_names: List of class names
            model_name: Name of model for saving
            max_depth: Maximum depth to visualize (None for full tree)

        Returns:
            Matplotlib figure
        """
        max_depth = max_depth or viz_config.tree_max_depth

        # Create figure
        fig, ax = plt.subplots(figsize=viz_config.tree_figsize)

        # Plot tree
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=max_depth,
            ax=ax
        )

        ax.set_title(
            f'Decision Tree Visualization - {model_name}'
            f'{f" (max_depth={max_depth})" if max_depth else ""}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"{model_name}_tree_plot.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Decision tree plot saved to {filename}")

        return fig

    def plot_target_distribution(
        self,
        y: pd.Series,
        title: str = "Target Distribution"
    ) -> plt.Figure:
        """
        Plot target variable distribution

        Args:
            y: Target variable
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Count values
        value_counts = y.value_counts().sort_index()

        # Create bar plot
        bars = ax.bar(value_counts.index, value_counts.values, color='steelblue', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(y)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Target Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(value_counts.index)

        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"target_distribution.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Target distribution plot saved to {filename}")

        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame
    ) -> plt.Figure:
        """
        Plot model comparison

        Args:
            comparison_df: DataFrame with model comparison results

        Returns:
            Matplotlib figure
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(18, 5))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Create bar plot
            bars = ax.bar(
                comparison_df['Model'],
                comparison_df[metric],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(comparison_df)))
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"model_comparison.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Model comparison plot saved to {filename}")

        return fig

    def plot_learning_curve(
        self,
        train_scores: List[float],
        test_scores: List[float],
        title: str = "Learning Curve"
    ) -> plt.Figure:
        """
        Plot learning curve (train vs test performance)

        Args:
            train_scores: Training scores
            test_scores: Test scores
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = list(range(1, len(train_scores) + 1))

        ax.plot(x, train_scores, 'o-', color='blue', label='Training Score', linewidth=2)
        ax.plot(x, test_scores, 'o-', color='red', label='Test Score', linewidth=2)

        ax.set_xlabel('Iteration / Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"learning_curve.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Learning curve saved to {filename}")

        return fig

    def plot_class_performance(
        self,
        classification_report: dict,
        model_name: str = "model"
    ) -> plt.Figure:
        """
        Plot per-class performance metrics

        Args:
            classification_report: sklearn classification report dict
            model_name: Name of model for title

        Returns:
            Matplotlib figure
        """
        # Extract per-class metrics
        classes = [k for k in classification_report.keys()
                  if k not in ['accuracy', 'macro avg', 'weighted avg']]

        metrics_data = {
            'Class': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }

        for cls in classes:
            metrics_data['Class'].append(cls)
            metrics_data['Precision'].append(classification_report[cls]['precision'])
            metrics_data['Recall'].append(classification_report[cls]['recall'])
            metrics_data['F1-Score'].append(classification_report[cls]['f1-score'])

        df = pd.DataFrame(metrics_data)

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(df['Class']))
        width = 0.25

        ax.bar(x - width, df['Precision'], width, label='Precision', color='skyblue')
        ax.bar(x, df['Recall'], width, label='Recall', color='lightcoral')
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', color='lightgreen')

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Performance - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Class'])
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if self.save_figures:
            filename = FIGURES_DIR / f"{model_name}_class_performance.{viz_config.figure_format}"
            fig.savefig(filename, dpi=viz_config.figure_dpi, bbox_inches='tight')
            print(f"  ✓ Class performance plot saved to {filename}")

        return fig


def create_all_visualizations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    confusion_mat: np.ndarray,
    importance_df: pd.DataFrame,
    classification_report: dict,
    model_name: str = "model"
):
    """
    Create all standard visualizations

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        y_pred: Predictions
        confusion_mat: Confusion matrix
        importance_df: Feature importance DataFrame
        classification_report: Classification report dict
        model_name: Model name for saving
    """
    print("\n=== Generating Visualizations ===")

    viz = Visualizer(save_figures=True)

    # 1. Target distribution
    viz.plot_target_distribution(y_train, "Training Target Distribution")

    # 2. Confusion matrix
    class_labels = sorted(y_test.unique())
    viz.plot_confusion_matrix(confusion_mat, class_labels, model_name)

    # 3. Feature importance
    viz.plot_feature_importance(importance_df, model_name)

    # 4. Class performance
    viz.plot_class_performance(classification_report, model_name)

    # 5. Decision tree (if applicable)
    if hasattr(model, 'tree_'):
        # Single decision tree
        viz.plot_decision_tree(
            model,
            list(X_train.columns),
            [str(c) for c in class_labels],
            model_name,
            max_depth=5
        )
    elif hasattr(model, 'estimators_'):
        # Random forest - plot first tree
        viz.plot_decision_tree(
            model.estimators_[0],
            list(X_train.columns),
            [str(c) for c in class_labels],
            f"{model_name}_tree_0",
            max_depth=5
        )

    print("✓ All visualizations generated")


if __name__ == "__main__":
    print("Visualization module - Run train.py to generate visualizations")
