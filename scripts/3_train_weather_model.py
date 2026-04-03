#!/usr/bin/env python3
"""
Weather Model Training Script - Mac Optimized
Trains Random Forest and XGBoost models for blight prediction
"""

import numpy as np
import pickle
import platform
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import time


def load_processed_data():
    """Load preprocessed data"""
    print("\nLoading preprocessed data...")

    data_dir = Path("data/processed")

    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")

    with open(data_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    print(f"Data loaded!")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Classes: {le.classes_}")

    return X_train, X_test, y_train, y_test, le


def get_n_jobs():
    """Get safe n_jobs for macOS"""
    if platform.system() == "Darwin":
        # macOS can have multiprocessing issues; use half cores or 2
        import os
        return min(os.cpu_count() // 2 or 1, 4)
    return -1


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest...")

    n_jobs = get_n_jobs()
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=n_jobs,
        verbose=1
    )

    rf.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Random Forest trained in {elapsed:.2f}s")
    return rf


def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    print("\nTraining XGBoost...")

    n_jobs = get_n_jobs()
    num_classes = len(np.unique(y_train))

    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        n_jobs=n_jobs,
        verbosity=1,
        num_class=num_classes if num_classes > 2 else None,
    )

    xgb_model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"XGBoost trained in {elapsed:.2f}s")
    return xgb_model


def evaluate_model(model, X_test, y_test, le, model_name):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} Results:")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return accuracy, y_pred


def save_model(model, model_name):
    """Save trained model"""
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}_weather_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved: {model_path}")
    return model_path


def main():
    print("=" * 60)
    print("WEATHER MODEL TRAINING - MAC OPTIMIZED")
    print("=" * 60)

    try:
        # Load data
        X_train, X_test, y_train, y_test, le = load_processed_data()

        # Train Random Forest
        rf_model = train_random_forest(X_train, y_train)
        rf_acc, _ = evaluate_model(rf_model, X_test, y_test, le, "Random Forest")
        save_model(rf_model, "random_forest")

        # Train XGBoost
        xgb_model = train_xgboost(X_train, y_train)
        xgb_acc, _ = evaluate_model(xgb_model, X_test, y_test, le, "XGBoost")
        save_model(xgb_model, "xgboost")

        # Compare models
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"Random Forest: {rf_acc*100:.2f}%")
        print(f"XGBoost:       {xgb_acc*100:.2f}%")

        if xgb_acc > rf_acc:
            print("\nBest Model: XGBoost")
            best_model = xgb_model
        else:
            print("\nBest Model: Random Forest")
            best_model = rf_model

        # Save best model separately
        save_model(best_model, "best")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Models saved to: data/models/")
        print(f"Next: Run python scripts/5_train_vit_model.py")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
