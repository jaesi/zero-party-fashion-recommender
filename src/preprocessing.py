"""
Data Preprocessing Module
Handles data loading, cleaning, validation, and missing value imputation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.config import data_config, RAW_DATA_FILE, PROCESSED_DATA_DIR


class DataLoader:
    """Load and validate raw data from Excel files"""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or RAW_DATA_FILE

    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file with validation"""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please place your data file in {self.data_path.parent}"
            )

        print(f"Loading data from {self.data_path}...")
        df = pd.read_excel(self.data_path)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data structure and required columns"""
        required_columns = [data_config.target_column]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print("✓ Data validation passed")
        return True


class DataCleaner:
    """Clean and preprocess raw data"""

    def __init__(self, strategy: str = "smart"):
        """
        Initialize DataCleaner

        Args:
            strategy: Missing value handling strategy
                - "zero": Fill all missing values with 0 (original approach)
                - "smart": Use intelligent imputation based on feature type
                - "median": Fill with median values
                - "mean": Fill with mean values
                - "drop": Drop rows with missing values
        """
        self.strategy = strategy

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main cleaning pipeline"""
        df = df.copy()

        print("\n=== Data Cleaning Pipeline ===")

        # 1. Analyze missing values
        self._analyze_missing_values(df)

        # 2. Handle missing values
        df = self._handle_missing_values(df)

        # 3. Validate data types
        df = self._validate_data_types(df)

        # 4. Remove duplicates
        df = self._remove_duplicates(df)

        # 5. Handle outliers (optional)
        df = self._handle_outliers(df)

        print(f"\n✓ Cleaning completed: {len(df)} samples remaining")
        return df

    def _analyze_missing_values(self, df: pd.DataFrame) -> None:
        """Analyze and report missing values"""
        missing_info = df.isnull().sum()
        missing_pct = (missing_info / len(df) * 100).round(2)

        missing_df = pd.DataFrame({
            'Missing_Count': missing_info,
            'Missing_Percentage': missing_pct
        })

        # Only show columns with missing values
        missing_df = missing_df[missing_df['Missing_Count'] > 0]

        if len(missing_df) > 0:
            print("\n📊 Missing Value Analysis:")
            print(missing_df.sort_values('Missing_Percentage', ascending=False))
        else:
            print("\n✓ No missing values found")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        print(f"\n🔧 Handling missing values (strategy: {self.strategy})...")

        if self.strategy == "zero":
            df = df.fillna(0)
            print("  → Filled all missing values with 0")

        elif self.strategy == "drop":
            initial_count = len(df)
            df = df.dropna()
            dropped = initial_count - len(df)
            print(f"  → Dropped {dropped} rows with missing values")

        elif self.strategy == "smart":
            df = self._smart_imputation(df)

        elif self.strategy in ["median", "mean"]:
            # Separate numeric and non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

            if self.strategy == "median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            else:  # mean
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # Fill non-numeric with mode or 0
            for col in non_numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)

            print(f"  → Filled missing values with {self.strategy}")

        return df

    def _smart_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent imputation based on feature type"""
        # MBTI features: Binary, fill with mode (most common value)
        mbti_features = data_config.mbti_features
        for col in mbti_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                df[col] = df[col].fillna(mode_val)
                print(f"  → {col}: filled with mode ({mode_val})")

        # Demographic features: Fill with median
        demographic_features = data_config.demographic_features
        for col in demographic_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  → {col}: filled with median ({median_val:.2f})")

        # Sock preference features: Fill with 0 (means "not selected")
        # This is domain-specific knowledge
        sock_features = data_config.sock_preference_features
        for col in sock_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(0)
                print(f"  → {col}: filled with 0 (not selected)")

        # Psychographic features: Fill with median
        psycho_features = data_config.psychographic_features
        for col in psycho_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  → {col}: filled with median ({median_val:.2f})")

        # Remaining features: Fill with 0
        remaining_cols = df.columns[df.isnull().any()]
        if len(remaining_cols) > 0:
            df[remaining_cols] = df[remaining_cols].fillna(0)
            print(f"  → Remaining {len(remaining_cols)} columns: filled with 0")

        return df

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types"""
        # MBTI should be binary integers
        for col in data_config.mbti_features:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Target should be integer
        if data_config.target_column in df.columns:
            df[data_config.target_column] = df[data_config.target_column].astype(int)

        print("✓ Data types validated")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)

        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate rows")
        else:
            print("✓ No duplicates found")

        return df

    def _handle_outliers(self, df: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
        """
        Detect and optionally handle outliers using z-score method

        Args:
            df: DataFrame
            z_threshold: Z-score threshold for outlier detection
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != data_config.target_column]

        outlier_counts = {}

        for col in numeric_cols:
            if df[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > z_threshold
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    outlier_counts[col] = outlier_count

        if outlier_counts:
            print(f"\n⚠️  Outliers detected in {len(outlier_counts)} columns:")
            for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  → {col}: {count} outliers")
            print("  (Keeping outliers for now - CART models are robust to outliers)")
        else:
            print("✓ No significant outliers detected")

        return df


def load_and_preprocess_data(
    data_path: Optional[Path] = None,
    cleaning_strategy: str = "smart"
) -> pd.DataFrame:
    """
    Main function to load and preprocess data

    Args:
        data_path: Path to data file
        cleaning_strategy: Missing value handling strategy

    Returns:
        Cleaned DataFrame
    """
    # Load data
    loader = DataLoader(data_path)
    df = loader.load_data()
    loader.validate_data(df)

    # Clean data
    cleaner = DataCleaner(strategy=cleaning_strategy)
    df_clean = cleaner.clean_data(df)

    # Save processed data
    output_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Processed data saved to {output_path}")

    return df_clean


if __name__ == "__main__":
    # Test preprocessing
    try:
        df = load_and_preprocess_data(cleaning_strategy="smart")
        print("\n" + "="*50)
        print("Final Dataset Summary:")
        print(f"  • Shape: {df.shape}")
        print(f"  • Features: {df.shape[1]}")
        print(f"  • Samples: {df.shape[0]}")
        print(f"  • Missing values: {df.isnull().sum().sum()}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo run preprocessing, place your data file in:")
        print(f"  {RAW_DATA_FILE}")
