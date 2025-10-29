"""
Generate Sample Fashion Survey Data

This script generates synthetic survey data for the zero-party fashion recommender system.
The data simulates responses from approximately 400 participants answering questions about
their fashion preferences, MBTI personality types, and sock color preferences.

Column Descriptions:
- gender: 1 (Male) or 2 (Female)
- age: Age of participant (18-60)
- mbti_* : MBTI personality type indicators (0 or 1)
  - mbti_e_i: Extraversion (1) vs Introversion (0)
  - mbti_s_n: Sensing (1) vs Intuition (0)
  - mbti_t_f: Thinking (1) vs Feeling (0)
  - mbti_j_p: Judging (1) vs Perceiving (0)
- lifestyle_*: Lifestyle preference scores (1-5)
- fashion_*: Fashion-related behavior scores
- sns_*: Social media activity scores
- fashion_involvement: Overall fashion involvement score
- target_group: Target recommendation group (1-4)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_mbti_profile(n_samples):
    """Generate MBTI personality profiles."""
    return {
        'mbti_e_i': np.random.binomial(1, 0.5, n_samples),  # E vs I
        'mbti_s_n': np.random.binomial(1, 0.6, n_samples),  # S vs N (S slightly more common)
        'mbti_t_f': np.random.binomial(1, 0.5, n_samples),  # T vs F
        'mbti_j_p': np.random.binomial(1, 0.5, n_samples),  # J vs P
    }


def generate_demographics(n_samples):
    """Generate demographic information."""
    return {
        'gender': np.random.choice([1, 2], n_samples, p=[0.35, 0.65]),  # 35% male, 65% female
        'age': np.random.randint(18, 61, n_samples),
        'height': np.random.randint(220, 281, n_samples),  # Shoe size in mm
    }


def generate_lifestyle_preferences(n_samples):
    """Generate lifestyle preference scores."""
    return {
        'lifestyle_q7': np.random.randint(1, 6, n_samples),
        'lifestyle_q8': np.random.randint(1, 6, n_samples),
        'lifestyle_q9': np.random.randint(1, 6, n_samples),
        'lifestyle_q10': np.random.randint(1, 5, n_samples),
    }


def generate_foot_characteristics(n_samples):
    """Generate foot characteristics (binary responses)."""
    return {
        'foot_bunion': np.random.binomial(1, 0.15, n_samples),
        'foot_fungus': np.random.binomial(1, 0.20, n_samples),
        'foot_ingrown': np.random.binomial(1, 0.18, n_samples),
        'foot_long_toe': np.random.binomial(1, 0.12, n_samples),
        'foot_high_arch': np.random.binomial(1, 0.25, n_samples),
        'foot_wide': np.random.binomial(1, 0.35, n_samples),
        'foot_heel_flat': np.random.binomial(1, 0.20, n_samples),
        'foot_flat': np.random.binomial(1, 0.30, n_samples),
        'foot_large': np.random.binomial(1, 0.15, n_samples),
        'foot_small': np.random.binomial(1, 0.10, n_samples),
        'foot_no_issue': np.random.binomial(1, 0.25, n_samples),
    }


def generate_fashion_attitudes(n_samples):
    """Generate fashion attitude scores."""
    # These are on a 1-5 scale, using continuous values
    individuality = np.random.normal(3.0, 0.8, n_samples)
    ostentation = np.random.normal(2.5, 0.7, n_samples)
    sports_orientation = np.random.normal(2.8, 0.9, n_samples)

    # Clothing pursuit benefits
    practicality = np.random.normal(3.5, 0.8, n_samples)
    trend_pursuit = np.random.normal(2.5, 0.7, n_samples)
    appearance_pursuit = np.random.normal(3.2, 0.8, n_samples)

    # SNS activity
    sns_personal = np.random.normal(2.5, 1.0, n_samples)
    sns_time = np.random.normal(3.0, 1.0, n_samples)

    # Fashion involvement
    fashion_involvement = np.random.normal(3.0, 0.9, n_samples)

    # Clip values to valid range [1, 5]
    return {
        'individuality_orientation': np.clip(np.round(individuality, 2), 1, 5),
        'ostentation_orientation': np.clip(np.round(ostentation, 2), 1, 5),
        'sports_orientation': np.clip(np.round(sports_orientation, 2), 1, 5),
        'clothing_practicality': np.clip(np.round(practicality, 2), 1, 5),
        'clothing_trend_pursuit': np.clip(np.round(trend_pursuit, 2), 1, 5),
        'clothing_appearance': np.clip(np.round(appearance_pursuit, 2), 1, 5),
        'sns_personal_expression': np.clip(np.round(sns_personal, 2), 1, 5),
        'sns_time_spent': np.clip(np.round(sns_time, 2), 1, 5),
        'fashion_involvement': np.clip(np.round(fashion_involvement, 2), 1, 5),
    }


def generate_sock_preferences_achromatic(n_samples):
    """Generate achromatic (black & white) sock preference rankings."""
    # 6 options for achromatic socks
    sock_prefs = {}
    for i in range(3, 9):  # q11_3 to q11_8
        # Not all participants rank all options, some will be 0 (not ranked)
        sock_prefs[f'achromatic_sock_{i}'] = np.where(
            np.random.random(n_samples) < 0.3,  # 30% chance of ranking each option
            np.random.randint(1, 9, n_samples),
            0
        )
    return sock_prefs


def generate_sock_preferences_color(n_samples):
    """Generate chromatic (colored) sock preference rankings."""
    # 6 options for colored socks
    sock_prefs = {}
    for i in [2, 3, 4, 5, 6, 8]:  # q12_2, q12_3, q12_4, q12_5, q12_6, q12_8
        # Not all participants rank all options
        sock_prefs[f'color_sock_{i}'] = np.where(
            np.random.random(n_samples) < 0.3,  # 30% chance of ranking each option
            np.random.randint(1, 9, n_samples),
            0
        )
    return sock_prefs


def generate_target_groups(df):
    """
    Generate target groups based on sock preferences.

    Target groups represent different recommendation categories:
    - Group 1: Bold colors preferred
    - Group 2: Pastel colors preferred
    - Group 3: Achromatic colors preferred
    - Group 4: Mixed/neutral preferences
    """
    n_samples = len(df)

    # Calculate preference scores
    achromatic_score = df[[c for c in df.columns if 'achromatic_sock_' in c]].sum(axis=1)
    color_score = df[[c for c in df.columns if 'color_sock_' in c]].sum(axis=1)

    # Assign target groups based on preferences
    target_group = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if achromatic_score.iloc[i] > color_score.iloc[i] * 1.5:
            target_group[i] = 3  # Achromatic preference
        elif color_score.iloc[i] > achromatic_score.iloc[i] * 1.5:
            # Split color preference into bold vs pastel
            if df['fashion_involvement'].iloc[i] > 3.5:
                target_group[i] = 1  # Bold colors
            else:
                target_group[i] = 2  # Pastel colors
        else:
            target_group[i] = 4  # Mixed/neutral

    return target_group


def generate_integrated_dataset(n_samples=450):
    """
    Generate integrated dataset with both achromatic and chromatic preferences.

    This dataset is used for the main model that considers all color preferences.
    """
    set_random_seed(42)

    data = {}
    data.update(generate_demographics(n_samples))
    data.update(generate_mbti_profile(n_samples))
    data.update(generate_lifestyle_preferences(n_samples))
    data.update(generate_foot_characteristics(n_samples))
    data.update(generate_fashion_attitudes(n_samples))
    data.update(generate_sock_preferences_achromatic(n_samples))
    data.update(generate_sock_preferences_color(n_samples))

    df = pd.DataFrame(data)
    df['target_group'] = generate_target_groups(df)

    return df


def generate_achromatic_dataset(n_samples=300):
    """
    Generate dataset focused on achromatic sock preferences.

    This dataset is used for models specifically analyzing black & white sock preferences.
    """
    set_random_seed(43)

    data = {}
    data.update(generate_demographics(n_samples))
    data.update(generate_mbti_profile(n_samples))
    data.update(generate_lifestyle_preferences(n_samples))
    data.update(generate_fashion_attitudes(n_samples))
    data.update(generate_sock_preferences_achromatic(n_samples))

    df = pd.DataFrame(data)

    return df


def generate_color_dataset(n_samples=300):
    """
    Generate dataset focused on chromatic sock preferences.

    This dataset is used for models specifically analyzing colored sock preferences.
    """
    set_random_seed(44)

    data = {}
    data.update(generate_demographics(n_samples))
    data.update(generate_mbti_profile(n_samples))
    data.update(generate_lifestyle_preferences(n_samples))
    data.update(generate_fashion_attitudes(n_samples))
    data.update(generate_sock_preferences_color(n_samples))

    df = pd.DataFrame(data)

    return df


def save_datasets(output_dir='data/raw'):
    """Generate and save all datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating integrated dataset...")
    df_integrated = generate_integrated_dataset(450)
    df_integrated.to_csv(output_path / 'fashion_survey_integrated.csv', index=False)
    print(f"✓ Saved: {output_path / 'fashion_survey_integrated.csv'} ({len(df_integrated)} samples)")

    print("\nGenerating achromatic dataset...")
    df_achromatic = generate_achromatic_dataset(300)
    df_achromatic.to_csv(output_path / 'fashion_survey_achromatic.csv', index=False)
    print(f"✓ Saved: {output_path / 'fashion_survey_achromatic.csv'} ({len(df_achromatic)} samples)")

    print("\nGenerating color dataset...")
    df_color = generate_color_dataset(300)
    df_color.to_csv(output_path / 'fashion_survey_color.csv', index=False)
    print(f"✓ Saved: {output_path / 'fashion_survey_color.csv'} ({len(df_color)} samples)")

    print("\n" + "="*50)
    print("Sample data generation complete!")
    print("="*50)


if __name__ == '__main__':
    save_datasets()
