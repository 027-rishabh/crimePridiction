import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
INPUT_FILE = Path("crime_data_cleaned.csv")
OUTPUT_DIR = Path("../features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load cleaned dataset"""
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df):,} records")
    return df


def create_temporal_features(df):
    """Create time-based features"""
    print("\nCreating temporal features...")

    df['year_normalized'] = (
        (df['year'] - df['year'].min()) /
        (df['year'].max() - df['year'].min())
    )

    df['years_since_2017'] = df['year'] - 2017

    df['year_position'] = (
        df.groupby(['state_name', 'district_name', 'protected_group'])
          .cumcount()
    )

    print("  ✓ Added temporal features")
    return df


def create_lag_features(df, periods=[1, 2, 3]):
    """Create lagged features for time series"""
    print(f"\nCreating lag features (periods: {periods})...")

    df = df.sort_values(
        ['state_name', 'district_name', 'protected_group', 'year']
    )

    crime_types = [
        'violent_crimes',
        'sexual_crimes',
        'property_crimes',
        'kidnapping_crimes',
        'total_crimes'
    ]

    for crime_type in crime_types:
        for period in periods:
            df[f'{crime_type}_lag_{period}'] = (
                df.groupby(
                    ['state_name', 'district_name', 'protected_group']
                )[crime_type]
                .shift(period)
            )

    print(f"  ✓ Added lag features for {len(crime_types)} crime types")
    return df


def create_rolling_features(df, windows=[2, 3]):
    """Create rolling window statistics"""
    print(f"\nCreating rolling features (windows: {windows})...")

    df = df.sort_values(
        ['state_name', 'district_name', 'protected_group', 'year']
    )

    for window in windows:
        df[f'total_crimes_rolling_mean_{window}y'] = (
            df.groupby(
                ['state_name', 'district_name', 'protected_group']
            )['total_crimes']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        df[f'total_crimes_rolling_std_{window}y'] = (
            df.groupby(
                ['state_name', 'district_name', 'protected_group']
            )['total_crimes']
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    print("  ✓ Added rolling statistics")
    return df


def create_group_encoding(df):
    """Encode protected groups"""
    print("\nEncoding protected groups...")

    group_dummies = pd.get_dummies(df['protected_group'], prefix='group')
    df = pd.concat([df, group_dummies], axis=1)

    group_map = {'ST': 0, 'SC': 1, 'Women': 2, 'Children': 3}
    df['group_encoded'] = df['protected_group'].map(group_map)

    print("  ✓ Added group encodings")
    return df


def create_geographic_features(df):
    """Create state/district level aggregations"""
    print("\nCreating geographic features...")

    df['state_total_crimes'] = (
        df.groupby(['state_name', 'year'])['total_crimes']
          .transform('sum')
    )

    df['district_total_crimes'] = (
        df.groupby(
            ['state_name', 'district_name', 'year']
        )['total_crimes']
        .transform('sum')
    )

    df['group_proportion'] = (
        df['total_crimes'] / (df['district_total_crimes'] + 1)
    )

    df['state_crime_rank'] = (
        df.groupby('year')['state_total_crimes']
          .rank(ascending=False, method='dense')
    )

    print("  ✓ Added geographic features")
    return df


def create_trend_features(df):
    """Create trend indicators"""
    print("\nCreating trend features...")

    df = df.sort_values(
        ['state_name', 'district_name', 'protected_group', 'year']
    )

    def calculate_slope(series):
        n = len(series)

        # Not enough data → flat trend
        if n < 2:
            return pd.Series(np.zeros(n), index=series.index)

        x = np.arange(n)
        y = series.values

        # Fit linear trend
        slope, _ = np.polyfit(x, y, 1)

        return pd.Series(
            np.repeat(slope, n),
            index=series.index
        )

    df['crime_trend'] = (
        df.groupby(
            ['state_name', 'district_name', 'protected_group']
        )['total_crimes']
        .transform(calculate_slope)
    )

    df['trend_increasing'] = (df['crime_trend'] > 0).astype(int)

    print("  ✓ Added trend features")
    return df


def save_feature_data(df):
    """Save dataset with features"""
    output_path = OUTPUT_DIR / "crime_data_features.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✓ Feature data saved to: {output_path}")
    print(f"  Shape: {df.shape}")

    feature_cols = [
        col for col in df.columns
        if col not in [
            'id', 'state_name', 'state_code',
            'district_name', 'district_code',
            'registration_circles', 'protected_group'
        ]
    ]

    feature_list_path = OUTPUT_DIR / "feature_list.txt"
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"✓ Feature list saved to: {feature_list_path}")
    print(f"  Total features: {len(feature_cols)}")


def main():
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    df = load_data()
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_group_encoding(df)
    df = create_geographic_features(df)
    df = create_trend_features(df)

    save_feature_data(df)

    print("\n✓ Feature engineering complete!")


if __name__ == "__main__":
    main()

