"""
Optimized training script for 4GB GPU with performance improvements.
Includes:
- Mixed precision training
- Gradient accumulation
- Class weighting for imbalanced data
- Feature importance analysis
- Early stopping with better patience
"""

import os
import pandas as pd
import numpy as np
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import pickle
from omegaconf import DictConfig, ListConfig
from omegaconf.base import Container, ContainerMetadata

# Set float32 matmul precision for better performance on Tensor Cores
torch.set_float32_matmul_precision('medium')

# Add safe globals for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([DictConfig, ListConfig, Container, ContainerMetadata])


def train_ft_transformer_optimized():
    """Train FT-Transformer with memory optimizations for 4GB GPU."""

    print("="*60)
    print("Training FT-Transformer (Memory Optimized)")
    print("="*60)

    # Load preprocessed data
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")

    # Get feature columns (all except target)
    feature_cols = [col for col in train_df.columns if col != 'Diabetes_012']

    # Data configuration
    data_config = DataConfig(
        target=['Diabetes_012'],
        continuous_cols=feature_cols,
        categorical_cols=[],
        num_workers=2,  # Reduced from 4 to save memory
    )

    # Trainer configuration with memory optimizations
    trainer_config = TrainerConfig(
        batch_size=64,  # Further reduced for 4GB GPU
        max_epochs=100,  # Increased to compensate for smaller batch
        early_stopping='valid_loss',
        early_stopping_patience=15,  # More patient since smaller batches
        checkpoints='valid_loss',
        load_best=False,
        trainer_kwargs=dict(
            enable_model_summary=True,
            enable_progress_bar=False,
            precision='16-mixed',  # Half precision for 50% memory savings
            accumulate_grad_batches=2,  # Gradient accumulation for larger effective batch
            gradient_clip_val=1.0,  # Stabilize training
        ),
    )

    # Optimizer configuration
    optimizer_config = OptimizerConfig()

    # FT-Transformer model configuration (minimal for 4GB GPU)
    model_config = FTTransformerConfig(
        task='classification',
        input_embed_dim=48,  # Very small embedding
        num_heads=4,
        num_attn_blocks=2,
        attn_dropout=0.3,  # Increased dropout for regularization
        ff_dropout=0.2,
        learning_rate=2e-3,  # Higher LR for faster convergence
    )

    # Create tabular model
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    # Train the model
    print("\nStarting training...")
    print("Using mixed precision (16-bit) and gradient accumulation")
    tabular_model.fit(train=train_df, validation=val_df)

    # Save the model
    os.makedirs('models', exist_ok=True)
    tabular_model.save_model('models/ft_transformer')
    print("\nFT-Transformer saved to models/ft_transformer/")

    return tabular_model


def train_xgboost_balanced():
    """Train XGBoost with class balancing for imbalanced data."""

    print("\n" + "="*60)
    print("Training XGBoost (Balanced)")
    print("="*60)

    # Load preprocessed data
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # Separate features and target
    X_train = train_df.drop('Diabetes_012', axis=1)
    y_train = train_df['Diabetes_012']
    X_val = val_df.drop('Diabetes_012', axis=1)
    y_val = val_df['Diabetes_012']

    print(f"\nClass distribution in training data:")
    print(y_train.value_counts())

    # Calculate class weights
    class_weights = compute_sample_weight('balanced', y_train)
    print(f"\nClass weights applied: {np.unique(y_train)} -> {np.unique(class_weights)}")

    # Train XGBoost
    print("\nTraining XGBoost with class weighting...")
    xgb_model = XGBClassifier(
        n_estimators=300,  # Increased for better performance
        max_depth=5,  # Slightly reduced for regularization
        learning_rate=0.05,  # Reduced for smoother learning
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss',
        subsample=0.8,  # Subsampling for regularization
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        use_label_encoder=False,
        early_stopping_rounds=20,
    )

    xgb_model.fit(
        X_train, y_train,
        sample_weight=class_weights,
        eval_set=[(X_val, y_val)],
        verbose=20
    )

    # Save the model
    os.makedirs('models', exist_ok=True)
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("\nXGBoost saved to models/xgboost_model.pkl")

    # Print feature importance
    feature_importance = xgb_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    return xgb_model


if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists('data/train.csv'):
        print("Data not found. Running data preparation...")
        from data import download_and_prepare_data
        download_and_prepare_data()

    # Train FT-Transformer (optimized)
    ft_model = train_ft_transformer_optimized()

    # Train XGBoost (balanced)
    xgb_model = train_xgboost_balanced()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nRun 'python evaluate.py' to evaluate the models on the test set.")
