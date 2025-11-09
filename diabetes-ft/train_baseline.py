"""
XGBoost Baseline Training Script

This script trains an XGBoost classifier as a baseline model for diabetes prediction.
Includes class weighting for imbalanced data and feature importance analysis.
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pickle


def train_xgboost_baseline(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    use_gpu=True,
    verbose=True
):
    """
    Train XGBoost baseline model with class balancing.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        use_gpu: Whether to use GPU acceleration
        verbose: Print training progress

    Returns:
        Trained XGBoost model
    """

    print("="*80)
    print("XGBOOST BASELINE TRAINING")
    print("="*80)

    # Load preprocessed data
    if not os.path.exists('data/train.csv'):
        print("\n❌ Error: Training data not found!")
        print("Please run: .venv/bin/python data.py")
        return None

    print("\n[1/5] Loading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    print(f"   Training samples: {len(train_df):,}")
    print(f"   Validation samples: {len(val_df):,}")

    # Separate features and target
    print("\n[2/5] Preparing features and target...")
    X_train = train_df.drop('Diabetes_012', axis=1)
    y_train = train_df['Diabetes_012']
    X_val = val_df.drop('Diabetes_012', axis=1)
    y_val = val_df['Diabetes_012']

    print(f"   Features: {X_train.shape[1]}")

    # Show class distribution
    print(f"\n   Training class distribution:")
    for class_label, count in y_train.value_counts().sort_index().items():
        percentage = (count / len(y_train)) * 100
        print(f"     Class {class_label}: {count:,} ({percentage:.2f}%)")

    # Calculate class weights
    print("\n[3/5] Calculating class weights for imbalanced data...")
    class_weights = compute_sample_weight('balanced', y_train)
    print(f"   Class weights applied: balanced")

    # Configure XGBoost
    print(f"\n[4/5] Configuring XGBoost...")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    print(f"   learning_rate: {learning_rate}")
    print(f"   GPU acceleration: {'CUDA' if use_gpu else 'CPU'}")

    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss',
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        early_stopping_rounds=20,
        device='cuda' if use_gpu else 'cpu',
    )

    # Train the model
    print(f"\n[5/5] Training XGBoost...")
    print("="*80)

    xgb_model.fit(
        X_train, y_train,
        sample_weight=class_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20 if verbose else 0
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Validation predictions
    print("\n[Validation Performance]")
    y_val_pred = xgb_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    f1_macro = f1_score(y_val, y_val_pred, average='macro')
    f1_weighted = f1_score(y_val, y_val_pred, average='weighted')

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (macro): {f1_macro:.4f}")
    print(f"   F1-Score (weighted): {f1_weighted:.4f}")

    print("\n   Classification Report:")
    print(classification_report(y_val, y_val_pred,
                                target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))

    # Feature importance
    print("\n[Feature Importance]")
    feature_importance = xgb_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\n   Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"     {row['feature']:<20} {row['importance']:.4f}")

    # Save the model
    print("\n[Saving Model]")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_baseline.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)

    print(f"   ✓ Model saved to: {model_path}")

    # Save feature importance
    importance_path = 'models/xgboost_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✓ Feature importance saved to: {importance_path}")

    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel location: {model_path}")
    print("Next step: Run 'python evaluate.py' to evaluate on test set")

    return xgb_model


if __name__ == "__main__":
    import sys

    # Check if data exists
    if not os.path.exists('data/train.csv'):
        print("\n❌ Training data not found!")
        print("\nPlease run data preparation first:")
        print("   .venv/bin/python data.py")
        print("\nThis will download and preprocess the diabetes dataset.")
        sys.exit(1)

    # Train baseline model
    model = train_xgboost_baseline(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        use_gpu=True,
        verbose=True
    )

    if model is not None:
        print("\n✅ XGBoost baseline training successful!")
