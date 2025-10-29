"""
Main Training Pipeline
End-to-end pipeline for Zero-Party Fashion Recommender
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.config import create_directories, data_config, model_config
from src.preprocessing import load_and_preprocess_data
from src.feature_engineering import prepare_features
from src.models import ModelTrainer, ModelComparison
from src.evaluation import ModelEvaluator, ErrorAnalysis
from src.visualization import create_all_visualizations


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("  Zero-Party Fashion Recommender - CART Decision Tree Model")
    print("  Fashion Recommendation using MBTI & Zero-Party Data")
    print("="*70 + "\n")


def run_pipeline(
    use_tuning: bool = True,
    tuning_method: str = "grid",
    compare_models: bool = True,
    create_visualizations: bool = True
):
    """
    Run the complete training pipeline

    Args:
        use_tuning: Whether to use hyperparameter tuning
        tuning_method: "grid" or "random" search
        compare_models: Whether to compare DT and RF
        create_visualizations: Whether to create visualizations
    """
    start_time = time.time()
    print_banner()

    # Step 1: Setup
    print("📁 Step 1/6: Setting up directories...")
    create_directories()

    # Step 2: Load and preprocess data
    print("\n📊 Step 2/6: Loading and preprocessing data...")
    try:
        df = load_and_preprocess_data(cleaning_strategy="smart")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease place your data file in: data/raw/rawdata_무채색유채색 통합.xlsx")
        return

    # Step 3: Feature engineering
    print("\n🔧 Step 3/6: Feature engineering...")
    X_train, X_test, y_train, y_test, feature_engineer = prepare_features(
        df,
        scaler_type="minmax",
        create_engineered_features=True
    )

    print(f"\n✓ Features prepared:")
    print(f"  • Training set: {X_train.shape}")
    print(f"  • Test set: {X_test.shape}")

    # Step 4: Model training
    print("\n🤖 Step 4/6: Training models...")

    models = {}
    evaluators = {}

    # Train Decision Tree
    print("\n" + "-"*70)
    dt_trainer = ModelTrainer(
        model_type="decision_tree",
        use_tuning=use_tuning,
        tuning_method=tuning_method
    )

    dt_results = dt_trainer.train(X_train, y_train, X_test, y_test)
    dt_trainer.save_model()
    models['Decision Tree'] = dt_trainer

    # Train Random Forest
    print("\n" + "-"*70)
    rf_trainer = ModelTrainer(
        model_type="random_forest",
        use_tuning=use_tuning,
        tuning_method=tuning_method
    )

    rf_results = rf_trainer.train(X_train, y_train, X_test, y_test)
    rf_trainer.save_model()
    models['Random Forest'] = rf_trainer

    # Step 5: Model evaluation
    print("\n📈 Step 5/6: Evaluating models...")
    print("\n" + "-"*70)

    # Evaluate Decision Tree
    dt_evaluator = ModelEvaluator("decision_tree")

    dt_pred = dt_trainer.model.predict(X_test)
    try:
        dt_pred_proba = dt_trainer.model.predict_proba(X_test)
    except:
        dt_pred_proba = None

    dt_evaluator.evaluate(y_test, dt_pred, dt_pred_proba, "test")
    dt_evaluator.evaluate(y_train, dt_trainer.model.predict(X_train), None, "train")
    dt_evaluator.compare_datasets()
    dt_evaluator.save_results()

    evaluators['Decision Tree'] = dt_evaluator

    # Error analysis for DT
    dt_error_analysis = ErrorAnalysis("decision_tree")
    dt_error_analysis.analyze_errors(X_test, y_test, dt_pred)

    print("\n" + "-"*70)

    # Evaluate Random Forest
    rf_evaluator = ModelEvaluator("random_forest")

    rf_pred = rf_trainer.model.predict(X_test)
    try:
        rf_pred_proba = rf_trainer.model.predict_proba(X_test)
    except:
        rf_pred_proba = None

    rf_evaluator.evaluate(y_test, rf_pred, rf_pred_proba, "test")
    rf_evaluator.evaluate(y_train, rf_trainer.model.predict(X_train), None, "train")
    rf_evaluator.compare_datasets()
    rf_evaluator.save_results()

    evaluators['Random Forest'] = rf_evaluator

    # Error analysis for RF
    rf_error_analysis = ErrorAnalysis("random_forest")
    rf_error_analysis.analyze_errors(X_test, y_test, rf_pred)

    # Model comparison
    if compare_models:
        print("\n" + "="*70)
        comparison = ModelComparison()
        comparison.add_model("Decision Tree", dt_trainer)
        comparison.add_model("Random Forest", rf_trainer)
        comparison_df = comparison.compare(X_test, y_test)

    # Step 6: Visualization
    if create_visualizations:
        print("\n🎨 Step 6/6: Creating visualizations...")
        print("\n" + "-"*70)

        # Visualize Decision Tree
        print("\nGenerating Decision Tree visualizations...")
        dt_importance = dt_trainer.get_feature_importance(list(X_train.columns))

        # Get the actual model (unwrap from GridSearchCV if needed)
        if hasattr(dt_trainer.model, 'best_estimator_'):
            dt_model_viz = dt_trainer.model.best_estimator_
        else:
            dt_model_viz = dt_trainer.model

        create_all_visualizations(
            dt_model_viz,
            X_train, X_test, y_train, y_test,
            dt_pred,
            dt_evaluator.evaluation_results["test"]["confusion_matrix"],
            dt_importance,
            dt_evaluator.evaluation_results["test"]["classification_report"],
            "decision_tree"
        )

        # Visualize Random Forest
        print("\nGenerating Random Forest visualizations...")
        rf_importance = rf_trainer.get_feature_importance(list(X_train.columns))

        if hasattr(rf_trainer.model, 'best_estimator_'):
            rf_model_viz = rf_trainer.model.best_estimator_
        else:
            rf_model_viz = rf_trainer.model

        create_all_visualizations(
            rf_model_viz,
            X_train, X_test, y_train, y_test,
            rf_pred,
            rf_evaluator.evaluation_results["test"]["confusion_matrix"],
            rf_importance,
            rf_evaluator.evaluation_results["test"]["classification_report"],
            "random_forest"
        )

        # Model comparison visualization
        if compare_models:
            from src.visualization import Visualizer
            viz = Visualizer(save_figures=True)
            viz.plot_model_comparison(comparison_df)

    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\n📊 Results Summary:")
    print(f"  • Decision Tree Test Accuracy: {dt_results['test_accuracy']:.4f}")
    print(f"  • Random Forest Test Accuracy: {rf_results['test_accuracy']:.4f}")
    print(f"\n📁 Output Locations:")
    print(f"  • Models: models/")
    print(f"  • Metrics: results/metrics/")
    print(f"  • Figures: results/figures/")
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Train Zero-Party Fashion Recommender models"
    )

    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip hyperparameter tuning (use default parameters)"
    )

    parser.add_argument(
        "--tuning-method",
        choices=["grid", "random"],
        default="grid",
        help="Hyperparameter tuning method (default: grid)"
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization generation"
    )

    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip model comparison"
    )

    args = parser.parse_args()

    # Run pipeline
    run_pipeline(
        use_tuning=not args.no_tuning,
        tuning_method=args.tuning_method,
        compare_models=not args.no_comparison,
        create_visualizations=not args.no_visualization
    )


if __name__ == "__main__":
    main()
