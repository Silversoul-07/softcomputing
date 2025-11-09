import os
import pandas as pd
import numpy as np
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import pickle
from omegaconf import DictConfig, ListConfig
from omegaconf.base import Container, ContainerMetadata

# Set float32 matmul precision for better performance on Tensor Cores
torch.set_float32_matmul_precision('medium')

# Add safe globals for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([DictConfig, ListConfig, Container, ContainerMetadata])


def train_ft_transformer():
    """Train FT-Transformer using pytorch_tabular."""

    print("="*60)
    print("Training FT-Transformer")
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
        num_workers=4,
    )

    # Trainer configuration
    trainer_config = TrainerConfig(
        batch_size=128,  # Reduced from 1024 for 4GB GPU
        max_epochs=50,
        early_stopping='valid_loss',
        early_stopping_patience=10,
        checkpoints='valid_loss',
        load_best=False,  # Don't load best to avoid checkpoint loading issues
        trainer_kwargs=dict(
            enable_model_summary=True,
            enable_progress_bar=False,  # Disable to avoid rich console live stack issues
            precision='16-mixed',  # Mixed precision training to reduce memory
        ),
    )

    # Optimizer configuration
    optimizer_config = OptimizerConfig()

    # FT-Transformer model configuration
    model_config = FTTransformerConfig(
        task='classification',
        input_embed_dim=64,  # Reduced from 192 (3x smaller)
        num_heads=4,  # Reduced from 8 (2x smaller)
        num_attn_blocks=2,  # Reduced from 3
        attn_dropout=0.2,
        ff_dropout=0.1,
        learning_rate=1e-3,  # Slightly higher LR for faster convergence
        # ff_dim_multiplier=2,  # Reduced from default 4
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
    tabular_model.fit(train=train_df, validation=val_df)

    # Save the model
    os.makedirs('models', exist_ok=True)
    tabular_model.save_model('models/ft_transformer')
    print("\nFT-Transformer saved to models/ft_transformer/")

    return tabular_model


def train_xgboost():
    """Train XGBoost baseline."""

    print("\n" + "="*60)
    print("Training XGBoost Baseline")
    print("="*60)

    # Load preprocessed data
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    # Separate features and target
    X_train = train_df.drop('Diabetes_012', axis=1)
    y_train = train_df['Diabetes_012']
    X_val = val_df.drop('Diabetes_012', axis=1)
    y_val = val_df['Diabetes_012']

    # Train XGBoost
    print("\nTraining XGBoost...")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_sample_weight('balanced', y_train)
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss',
        scale_pos_weight=1,  # Handle multiclass imbalance
        use_label_encoder=False,
        device='cuda',  # Use GPU if available (XGBoost 3.1+)
    )

    xgb_model.fit(
        X_train, y_train,
        sample_weight=class_weights,  # Apply class weights
        eval_set=[(X_val, y_val)],
        verbose=10
    )

    # Save the model
    os.makedirs('models', exist_ok=True)
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("\nXGBoost saved to models/xgboost_model.pkl")

    return xgb_model


if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists('data/train.csv'):
        print("Data not found. Running data preparation...")
        from data import download_and_prepare_data
        download_and_prepare_data()

    # Train FT-Transformer
    ft_model = train_ft_transformer()

    # Train XGBoost
    xgb_model = train_xgboost()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nRun 'python evaluate.py' to evaluate the models on the test set.")
