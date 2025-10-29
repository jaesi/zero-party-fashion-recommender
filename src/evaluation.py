"""
Model Evaluation Module
Comprehensive evaluation metrics and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score
)
import json
from pathlib import Path

from src.config import METRICS_DIR


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics"""

    def __init__(self, model_name: str = "model"):
        self.model_name = model_name
        self.evaluation_results = {}

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            dataset_name: Name of dataset (e.g., "test", "train", "validation")

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n=== Evaluating {self.model_name} on {dataset_name} set ===")

        results = {
            "model_name": self.model_name,
            "dataset": dataset_name,
            "n_samples": len(y_true)
        }

        # Basic metrics
        results.update(self._calculate_basic_metrics(y_true, y_pred))

        # Per-class metrics
        results.update(self._calculate_class_metrics(y_true, y_pred))

        # Confusion matrix
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # Classification report
        results["classification_report"] = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        # ROC-AUC (if probabilities provided and multi-class)
        if y_pred_proba is not None:
            try:
                results["roc_auc_ovr"] = roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class='ovr',
                    average='weighted'
                )
                results["roc_auc_ovo"] = roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class='ovo',
                    average='weighted'
                )
            except Exception as e:
                print(f"  ⚠️  Could not calculate ROC-AUC: {e}")

        # Cohen's Kappa
        results["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

        # Store results
        self.evaluation_results[dataset_name] = results

        # Print summary
        self._print_evaluation_summary(results)

        return results

    def _calculate_basic_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, average='macro', zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average='macro', zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average='macro', zero_division=0
            )
        }

    def _calculate_class_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Dict]:
        """Calculate per-class metrics"""
        classes = np.unique(y_true)
        class_metrics = {}

        for cls in classes:
            cls_true = (y_true == cls).astype(int)
            cls_pred = (y_pred == cls).astype(int)

            class_metrics[f"class_{cls}"] = {
                "precision": precision_score(cls_true, cls_pred, zero_division=0),
                "recall": recall_score(cls_true, cls_pred, zero_division=0),
                "f1": f1_score(cls_true, cls_pred, zero_division=0),
                "support": int(cls_true.sum())
            }

        return {"per_class_metrics": class_metrics}

    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n📊 Evaluation Summary:")
        print(f"  • Accuracy:    {results['accuracy']:.4f}")
        print(f"  • Precision:   {results['precision_weighted']:.4f} (weighted)")
        print(f"  • Recall:      {results['recall_weighted']:.4f} (weighted)")
        print(f"  • F1-Score:    {results['f1_weighted']:.4f} (weighted)")
        print(f"  • Cohen Kappa: {results['cohen_kappa']:.4f}")

        if "roc_auc_ovr" in results:
            print(f"  • ROC-AUC:     {results['roc_auc_ovr']:.4f} (OvR)")

    def compare_datasets(self) -> pd.DataFrame:
        """Compare metrics across datasets (train vs test)"""
        if len(self.evaluation_results) < 2:
            print("⚠️  Need at least 2 datasets to compare")
            return None

        comparison = []

        for dataset, results in self.evaluation_results.items():
            comparison.append({
                "Dataset": dataset,
                "Accuracy": results["accuracy"],
                "Precision": results["precision_weighted"],
                "Recall": results["recall_weighted"],
                "F1-Score": results["f1_weighted"],
                "Cohen Kappa": results["cohen_kappa"]
            })

        comparison_df = pd.DataFrame(comparison)

        print("\n=== Dataset Comparison ===")
        print(comparison_df.to_string(index=False))

        # Check for overfitting
        if "train" in self.evaluation_results and "test" in self.evaluation_results:
            train_acc = self.evaluation_results["train"]["accuracy"]
            test_acc = self.evaluation_results["test"]["accuracy"]
            gap = train_acc - test_acc

            print(f"\n📈 Generalization Gap: {gap:.4f}")

            if gap > 0.15:
                print("  ⚠️  High gap detected - possible overfitting!")
            elif gap > 0.10:
                print("  ⚠️  Moderate gap - consider regularization")
            else:
                print("  ✓ Good generalization")

        return comparison_df

    def save_results(self, filepath: Optional[Path] = None):
        """Save evaluation results to JSON"""
        if filepath is None:
            filepath = METRICS_DIR / f"{self.model_name}_evaluation.json"

        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)

        print(f"✓ Evaluation results saved to {filepath}")

    def generate_classification_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """
        Generate detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for target classes

        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0
        )

        print("\n=== Classification Report ===")
        print(report)

        # Save report
        report_file = METRICS_DIR / f"{self.model_name}_classification_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✓ Classification report saved to {report_file}")

        return report


class ErrorAnalysis:
    """Analyze model errors and misclassifications"""

    def __init__(self, model_name: str = "model"):
        self.model_name = model_name

    def analyze_errors(
        self,
        X_test: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze misclassified samples

        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            DataFrame with error analysis
        """
        print(f"\n=== Error Analysis for {self.model_name} ===")

        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        n_misclassified = misclassified_mask.sum()
        total_samples = len(y_true)

        print(f"  • Total samples: {total_samples}")
        print(f"  • Misclassified: {n_misclassified} ({n_misclassified/total_samples*100:.2f}%)")

        if n_misclassified == 0:
            print("  ✓ Perfect classification!")
            return pd.DataFrame()

        # Create error DataFrame
        error_df = pd.DataFrame({
            "true_label": y_true[misclassified_mask],
            "predicted_label": y_pred[misclassified_mask]
        })

        # Analyze error patterns
        error_patterns = error_df.groupby(["true_label", "predicted_label"]).size()
        error_patterns = error_patterns.reset_index(name="count")
        error_patterns = error_patterns.sort_values("count", ascending=False)

        print("\n  Most common misclassification patterns:")
        for _, row in error_patterns.head(5).iterrows():
            print(f"    • True: {row['true_label']} → Predicted: {row['predicted_label']} "
                  f"({row['count']} times)")

        # Save error analysis
        error_file = METRICS_DIR / f"{self.model_name}_error_analysis.csv"
        error_patterns.to_csv(error_file, index=False)
        print(f"\n✓ Error analysis saved to {error_file}")

        return error_df

    def get_confusion_by_class(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Get confusion matrix as DataFrame"""
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(y_true.unique())

        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{c}" for c in classes],
            columns=[f"Pred_{c}" for c in classes]
        )

        print("\n=== Confusion Matrix ===")
        print(cm_df)

        return cm_df


if __name__ == "__main__":
    print("Evaluation module - Run train.py to evaluate models")
