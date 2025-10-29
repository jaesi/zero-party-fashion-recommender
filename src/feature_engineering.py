"""
Feature Engineering Module
Handles feature transformation, scaling, and engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

from src.config import data_config, PROCESSED_DATA_DIR, MODELS_DIR


class FeatureEngineer:
    """Feature transformation and engineering"""

    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize FeatureEngineer

        Args:
            scaler_type: Type of scaler to use
                - "minmax": MinMaxScaler (0-1 range) - original approach
                - "standard": StandardScaler (z-score normalization)
                - "robust": RobustScaler (robust to outliers)
                - "none": No scaling
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self._initialize_scaler()

    def _initialize_scaler(self):
        """Initialize the appropriate scaler"""
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        print("\n=== Feature Engineering ===")

        # 1. MBTI combination features
        if all(col in df.columns for col in data_config.mbti_features):
            df = self._create_mbti_features(df)

        # 2. Interaction features for psychographics
        if all(col in df.columns for col in data_config.psychographic_features):
            df = self._create_psychographic_interactions(df)

        # 3. Sock preference aggregations
        df = self._create_sock_preference_features(df)

        # 4. Age group features
        if "q2_1" in df.columns:
            df = self._create_age_groups(df)

        print(f"✓ Feature engineering completed: {len(df.columns)} total features")
        return df

    def _create_mbti_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create MBTI combination features"""
        # Create MBTI type string (e.g., "ESTJ")
        mbti_map = {
            'I_E': {0: 'I', 1: 'E'},
            'S_N': {0: 'S', 1: 'N'},
            'T_F': {0: 'T', 1: 'F'},
            'J_P': {0: 'J', 1: 'P'}
        }

        # Create combinations
        df['mbti_extraversion_score'] = df['I_E']  # Already binary
        df['mbti_intuition_score'] = df['S_N']
        df['mbti_thinking_score'] = df['T_F']
        df['mbti_perceiving_score'] = df['J_P']

        # Total MBTI diversity score (sum of all dimensions)
        df['mbti_total_score'] = df[data_config.mbti_features].sum(axis=1)

        print("  → Created 5 MBTI combination features")
        return df

    def _create_psychographic_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for psychographic variables"""
        psycho_cols = [col for col in data_config.psychographic_features if col in df.columns]

        if len(psycho_cols) >= 2:
            # Fashion involvement score (패션관여도)
            if '패션관여도' in df.columns:
                # Interaction with other psychographics
                if '개성지향' in df.columns:
                    df['fashion_individuality'] = df['패션관여도'] * df['개성지향']

                if '과시지향' in df.columns:
                    df['fashion_ostentation'] = df['패션관여도'] * df['과시지향']

            # SNS activity composite
            if 'SNS활동_자기표현' in df.columns and 'SNS활동_시간' in df.columns:
                df['sns_composite'] = (df['SNS활동_자기표현'] + df['SNS활동_시간']) / 2

            # Clothing benefit composite
            benefit_cols = [col for col in psycho_cols if '의복추구혜택' in col]
            if len(benefit_cols) > 0:
                df['clothing_benefit_avg'] = df[benefit_cols].mean(axis=1)

            print(f"  → Created psychographic interaction features")

        return df

    def _create_sock_preference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated sock preference features"""
        # Black & White sock preferences (q11_X)
        bw_cols = [col for col in df.columns if col.startswith('q11_')]
        if len(bw_cols) > 0:
            df['bw_sock_preference_count'] = df[bw_cols].apply(lambda x: (x > 0).sum(), axis=1)
            df['bw_sock_preference_max_rank'] = df[bw_cols].max(axis=1)
            df['bw_sock_preference_sum'] = df[bw_cols].sum(axis=1)

        # Color sock preferences (q12_X)
        color_cols = [col for col in df.columns if col.startswith('q12_')]
        if len(color_cols) > 0:
            df['color_sock_preference_count'] = df[color_cols].apply(lambda x: (x > 0).sum(), axis=1)
            df['color_sock_preference_max_rank'] = df[color_cols].max(axis=1)
            df['color_sock_preference_sum'] = df[color_cols].sum(axis=1)

        # Total sock engagement
        if len(bw_cols) > 0 and len(color_cols) > 0:
            df['total_sock_engagement'] = (
                df['bw_sock_preference_count'] + df['color_sock_preference_count']
            )

        print("  → Created sock preference aggregation features")
        return df

    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age group categorical features"""
        # Assuming q2_1 is age
        df['age_group'] = pd.cut(
            df['q2_1'],
            bins=[0, 25, 35, 45, 100],
            labels=[1, 2, 3, 4]  # Young, Young Adult, Middle Age, Senior
        )
        df['age_group'] = df['age_group'].astype(int)

        print("  → Created age group features")
        return df

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit scaler and transform features

        Args:
            X: Feature DataFrame
            y: Target (not used, for compatibility)

        Returns:
            Scaled DataFrame
        """
        if self.scaler is None:
            print("  → No scaling applied")
            return X

        print(f"  → Applying {self.scaler_type} scaling...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler

        Args:
            X: Feature DataFrame

        Returns:
            Scaled DataFrame
        """
        if self.scaler is None:
            return X

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_scaled

    def save_scaler(self, filepath: Optional[str] = None):
        """Save fitted scaler"""
        if self.scaler is None:
            print("No scaler to save")
            return

        if filepath is None:
            filepath = MODELS_DIR / "scaler.pkl"

        joblib.dump(self.scaler, filepath)
        print(f"✓ Scaler saved to {filepath}")

    def load_scaler(self, filepath: Optional[str] = None):
        """Load fitted scaler"""
        if filepath is None:
            filepath = MODELS_DIR / "scaler.pkl"

        self.scaler = joblib.load(filepath)
        print(f"✓ Scaler loaded from {filepath}")


def prepare_features(
    df: pd.DataFrame,
    target_col: str = None,
    test_size: float = None,
    random_state: int = None,
    scaler_type: str = "minmax",
    create_engineered_features: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FeatureEngineer]:
    """
    Complete feature preparation pipeline

    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Test set size (fraction)
        random_state: Random state for reproducibility
        scaler_type: Type of scaler to use
        create_engineered_features: Whether to create engineered features

    Returns:
        X_train, X_test, y_train, y_test, feature_engineer
    """
    # Use defaults from config
    target_col = target_col or data_config.target_column
    test_size = test_size or data_config.test_size
    random_state = random_state or data_config.random_state

    print("\n=== Feature Preparation Pipeline ===")

    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    y = df[target_col]
    X = df.drop([target_col], axis=1)

    print(f"✓ Separated features and target")
    print(f"  • Features: {X.shape[1]}")
    print(f"  • Samples: {X.shape[0]}")
    print(f"  • Target classes: {y.nunique()}")

    # Create engineered features
    if create_engineered_features:
        engineer = FeatureEngineer(scaler_type=scaler_type)
        X = engineer.create_features(X)
    else:
        engineer = FeatureEngineer(scaler_type=scaler_type)

    # Split data BEFORE scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )

    print(f"\n✓ Train/test split: {len(X_train)} train, {len(X_test)} test")

    # Scale features
    X_train = engineer.fit_transform(X_train)
    X_test = engineer.transform(X_test)

    # Save scaler
    engineer.save_scaler()

    return X_train, X_test, y_train, y_test, engineer


if __name__ == "__main__":
    # Test feature engineering
    try:
        # Load processed data
        processed_data = PROCESSED_DATA_DIR / "cleaned_data.csv"
        if not processed_data.exists():
            print(f"❌ Processed data not found: {processed_data}")
            print("Run preprocessing.py first")
        else:
            df = pd.read_csv(processed_data)
            print(f"Loaded data: {df.shape}")

            # Prepare features
            X_train, X_test, y_train, y_test, engineer = prepare_features(
                df,
                scaler_type="minmax",
                create_engineered_features=True
            )

            print("\n" + "="*50)
            print("Feature Preparation Summary:")
            print(f"  • Train features shape: {X_train.shape}")
            print(f"  • Test features shape: {X_test.shape}")
            print(f"  • Train target shape: {y_train.shape}")
            print(f"  • Test target shape: {y_test.shape}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
