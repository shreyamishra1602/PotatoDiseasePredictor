#!/usr/bin/env python3
"""
Weather Data Preprocessing Script
Prepares weather data for ML model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


def load_weather_data():
    """Load weather CSV data from Kaggle or local fallback"""
    print("\nLoading weather data...")

    # Try multiple possible locations (Kaggle download names vary)
    possible_paths = [
        Path("data/raw/Potato_Leaf_Disease_Dataset.csv"),
        Path("data/raw/potato_disease_weather.csv"),
        Path("data/raw/weather_data.csv"),
        Path("data/raw/potato-leaf-disease-based-on-weather-details.csv"),
        Path("potato_blight_data.csv"),  # Local fallback
    ]

    # Also search for any CSV in data/raw/
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        for csv in raw_dir.glob("*.csv"):
            if csv not in possible_paths:
                possible_paths.insert(0, csv)

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded from: {path}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            return df

    raise FileNotFoundError("No weather CSV found! Run download script first.")


def normalize_columns(df):
    """Normalize column names from various Kaggle dataset formats"""
    col_map = {}
    used_names = set()

    for col in df.columns:
        lower = col.strip().lower().replace(" ", "_")

        if lower == "wind_speed":
            name = "wind_speed"
        elif lower == "wind_bearing":
            name = "wind_bearing"
        elif "temp" in lower:
            name = "temperature"
        elif "humid" in lower:
            name = "humidity"
        elif "rain" in lower or "precip" in lower:
            name = "rainfall"
        elif "wind" in lower and "bearing" in lower:
            name = "wind_bearing"
        elif "wind" in lower:
            name = "wind_speed"
        elif "dew" in lower:
            name = "dew_point"
        elif lower == "disease_in_number" or lower == "disease in number":
            name = "disease_number"  # keep separate from text label
        elif "disease" in lower or "blight" in lower or "risk" in lower:
            name = "disease_label"
        elif "date" in lower:
            name = "date"
        elif "pressure" in lower:
            name = "pressure"
        elif "visib" in lower:
            name = "visibility"
        else:
            name = lower

        # Avoid duplicate column names
        if name in used_names:
            name = f"{name}_2"
        used_names.add(name)
        col_map[col] = name

    df = df.rename(columns=col_map)
    print(f"   Normalized columns: {list(df.columns)}")
    return df


def engineer_features(df):
    """Create additional features for better prediction"""
    print("\nEngineering features...")

    df = df.copy()

    # Handle date if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    # Temperature-Humidity interaction (critical for blight)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        df['is_blight_favorable'] = (
            (df['temperature'] >= 15) & (df['temperature'] <= 25) &
            (df['humidity'] >= 85)
        ).astype(int)

    # Dew point features
    if 'dew_point' in df.columns and 'temperature' in df.columns:
        df['temp_dewpoint_diff'] = df['temperature'] - df['dew_point']

    # Rainfall categories
    if 'rainfall' in df.columns:
        df['rainfall_category'] = pd.cut(
            df['rainfall'],
            bins=[-np.inf, 1, 5, 10, np.inf],
            labels=['no_rain', 'light', 'moderate', 'heavy']
        )
        df['rainfall_category'] = df['rainfall_category'].astype(str)

    # Wind features
    if 'wind_speed' in df.columns:
        df['wind_category'] = pd.cut(
            df['wind_speed'],
            bins=[0, 5, 10, 15, np.inf],
            labels=['calm', 'light', 'moderate', 'strong']
        )
        df['wind_category'] = df['wind_category'].astype(str)

    # Pressure features
    if 'pressure' in df.columns:
        df['low_pressure'] = (df['pressure'] < 1010).astype(int)

    print(f"Features engineered. New shape: {df.shape}")
    return df


def prepare_training_data(df):
    """Prepare features and labels"""
    print("\nPreparing training data...")

    # Identify target column
    target_cols = ['disease_label', 'blight_risk', 'disease', 'Disease', 'risk']
    target_col = None
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError("No target column found! Looking for: disease_label, blight_risk, disease, etc.")

    print(f"   Target column: {target_col}")
    print(f"   Classes: {df[target_col].unique()}")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Clean target labels (strip whitespace to merge "Late Blight " -> "Late Blight")
    df[target_col] = df[target_col].str.strip()
    print(f"   Classes (cleaned): {df[target_col].unique()}")

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    print(f"   Encoded classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Select features (exclude target, target-derived, and non-numeric identifiers)
    exclude_cols = [target_col, 'date', 'Disease in number', 'disease_number']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Handle categorical columns
    df_features = df[feature_cols].copy()
    categorical_cols = df_features.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        print(f"   Encoding categorical: {list(categorical_cols)}")
        df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)

    # Drop any remaining non-numeric
    df_features = df_features.select_dtypes(include=[np.number])

    X = df_features.values

    print(f"   Features shape: {X.shape}")
    print(f"   Feature names: {list(df_features.columns)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler, list(df_features.columns)


def split_and_save(X, y, le, scaler, feature_names):
    """Split data and save"""
    print("\nSplitting and saving data...")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    # Save encoders and feature names
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print(f"Data saved to: {output_dir}")

    return X_train, X_test, y_train, y_test


def main():
    print("=" * 60)
    print("WEATHER DATA PREPROCESSING")
    print("=" * 60)

    try:
        # Load data
        df = load_weather_data()

        # Normalize column names
        df = normalize_columns(df)

        # Engineer features
        df_engineered = engineer_features(df)

        # Prepare training data
        X, y, le, scaler, feature_names = prepare_training_data(df_engineered)

        # Split and save
        X_train, X_test, y_train, y_test = split_and_save(X, y, le, scaler, feature_names)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {list(le.classes_)}")
        print(f"Next: Run python scripts/3_train_weather_model.py")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
