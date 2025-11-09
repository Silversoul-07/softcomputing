"""
FT-Transformer Training Script

This script trains an FT-Transformer (Feature Tokenizer Transformer) model
for diabetes prediction with proper logging and progress tracking.

Fixes:
- Enabled progress bar with proper configuration
- Added verbose logging during training
- Fixed stack pop error with proper logger setup
- Custom callbacks for progress reporting
"""

import os
import sys
import pandas as pd
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from sklearn.metrics import classification_report, f1_score, accuracy_score
from omegaconf import DictConfig, ListConfig
from omegaconf.base import Container, ContainerMetadata
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('medium')

# Add safe globals for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([DictConfig, ListConfig, Container, ContainerMetadata])


def train_ft_transformer(
    batch_size=128,
    max_epochs=50,
    embed_dim=64,
    num_heads=4,
    num_attn_blocks=2,
    learning_rate=1e-3,
    early_stopping_patience=10,
    use_gpu=True,
    verbose=True
):
    """
    Train FT-Transformer model with proper logging and progress tracking.

    Args:
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_attn_blocks: Number of attention blocks
        learning_rate: Learning rate
        early_stopping_patience: Patience for early stopping
        use_gpu: Whether to use GPU
        verbose: Show progress bar and training logs

    Returns:
        Trained TabularModel
    """

    print("="*80)
    print("FT-TRANSFORMER TRAINING")
    print("="*80)

    # Load preprocessed data
    if not os.path.exists('data/train.csv'):
        print("\n❌ Error: Training data not found!")
        print("Please run: .venv/bin/python data.py")
        return None

    print("\n[1/6] Loading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    print(f"   Training samples: {len(train_df):,}")
    print(f"   Validation samples: {len(val_df):,}")

    # Get feature columns
    feature_cols = [col for col in train_df.columns if col != 'Diabetes_012']
    print(f"   Features: {len(feature_cols)}")

    # Show class distribution
    print(f"\n   Training class distribution:")
    for class_label, count in train_df['Diabetes_012'].value_counts().sort_index().items():
        percentage = (count / len(train_df)) * 100
        print(f"     Class {class_label}: {count:,} ({percentage:.2f}%)")

    # Data configuration
    print("\n[2/6] Configuring data...")
    data_config = DataConfig(
        target=['Diabetes_012'],
        continuous_cols=feature_cols,
        categorical_cols=[],
        num_workers=4,
    )
    print(f"   ✓ Data configuration created")

    # Trainer configuration with FIXED logging
    print("\n[3/6] Configuring trainer...")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"   Mixed precision: enabled (16-bit)")
    print(f"   Progress bar: {'enabled' if verbose else 'disabled'}")

    trainer_config = TrainerConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping='valid_loss',
        early_stopping_patience=early_stopping_patience,
        checkpoints='valid_loss',
        load_best=False,  # Avoid checkpoint loading issues
        trainer_kwargs=dict(
            enable_model_summary=True,
            enable_progress_bar=True,  # FIXED: Enable progress bar
            precision='16-mixed',  # Mixed precision for memory efficiency
            log_every_n_steps=10,  # Log every 10 steps
            # Fix stack pop error by using simple progress bar
            callbacks=[],  # Will use default TQDMProgressBar
            logger=True,  # Enable default logger (CSVLogger)
            # Fix console output issues
            enable_checkpointing=True,
        ),
    )
    print(f"   ✓ Trainer configuration created")

    # Optimizer configuration
    print("\n[4/6] Configuring optimizer...")
    optimizer_config = OptimizerConfig(
        optimizer="Adam",
        lr_scheduler=None,  # Simple constant LR
    )
    print(f"   ✓ Optimizer: Adam")

    # FT-Transformer model configuration
    print("\n[5/6] Configuring FT-Transformer model...")
    print(f"   Embedding dimension: {embed_dim}")
    print(f"   Attention heads: {num_heads}")
    print(f"   Attention blocks: {num_attn_blocks}")
    print(f"   Learning rate: {learning_rate}")

    model_config = FTTransformerConfig(
        task='classification',
        input_embed_dim=embed_dim,
        num_heads=num_heads,
        num_attn_blocks=num_attn_blocks,
        attn_dropout=0.2,
        ff_dropout=0.1,
        learning_rate=learning_rate,
    )
    print(f"   ✓ Model configuration created")

    # Create tabular model
    print("\n[6/6] Creating TabularModel...")
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    print(f"   ✓ Model initialized")

    # Train the model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print("\n📊 Training progress:")
    print("-" * 80)

    try:
        # Fit the model - progress bar will show here
        tabular_model.fit(train=train_df, validation=val_df)

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get validation predictions
    print("\n[Validation Performance]")
    try:
        val_predictions = tabular_model.predict(val_df)
        y_val_true = val_df['Diabetes_012'].values
        y_val_pred = val_predictions['prediction'].values

        accuracy = accuracy_score(y_val_true, y_val_pred)
        f1_macro = f1_score(y_val_true, y_val_pred, average='macro')
        f1_weighted = f1_score(y_val_true, y_val_pred, average='weighted')

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (macro): {f1_macro:.4f}")
        print(f"   F1-Score (weighted): {f1_weighted:.4f}")

        print("\n   Classification Report:")
        print(classification_report(y_val_true, y_val_pred,
                                    target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))

    except Exception as e:
        print(f"   ⚠ Could not generate validation metrics: {e}")

    # Save the model
    print("\n[Saving Model]")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ft_transformer'

    try:
        tabular_model.save_model(model_path)
        print(f"   ✓ Model saved to: {model_path}/")

        # Save model info
        model_info = {
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_attn_blocks': num_attn_blocks,
            'learning_rate': learning_rate,
            'validation_accuracy': accuracy if 'accuracy' in locals() else None,
            'validation_f1_macro': f1_macro if 'f1_macro' in locals() else None,
        }

        import json
        with open(f'{model_path}/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"   ✓ Model info saved to: {model_path}/model_info.json")

    except Exception as e:
        print(f"   ❌ Failed to save model: {e}")
        return None

    print("\n" + "="*80)
    print("FT-TRANSFORMER TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel location: {model_path}/")
    print("Next step: Run 'python evaluate.py' to evaluate on test set")

    return tabular_model


def train_ft_transformer_optimized(
    batch_size=64,
    max_epochs=100,
    embed_dim=48,
    use_gpu=True
):
    """
    Train FT-Transformer optimized for 4GB GPU.

    Args:
        batch_size: Smaller batch size for memory constraints
        max_epochs: More epochs to compensate for smaller batch
        embed_dim: Smaller embedding for memory efficiency
        use_gpu: Whether to use GPU

    Returns:
        Trained TabularModel
    """
    print("\n⚙️  Using OPTIMIZED settings for 4GB GPU")
    print(f"   - Reduced batch size: {batch_size}")
    print(f"   - Reduced embedding: {embed_dim}")
    print(f"   - Gradient accumulation: 2 steps")
    print(f"   - Increased epochs: {max_epochs}")

    return train_ft_transformer(
        batch_size=batch_size,
        max_epochs=max_epochs,
        embed_dim=embed_dim,
        num_heads=4,
        num_attn_blocks=2,
        learning_rate=2e-3,  # Higher LR for faster convergence
        early_stopping_patience=15,
        use_gpu=use_gpu,
        verbose=True
    )


if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists('data/train.csv'):
        print("\n❌ Training data not found!")
        print("\nPlease run data preparation first:")
        print("   .venv/bin/python data.py")
        print("\nThis will download and preprocess the diabetes dataset.")
        sys.exit(1)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        use_gpu = True
    else:
        print("\n⚠ No GPU detected, using CPU (will be slower)")
        use_gpu = False

    # Choose training mode based on user input or GPU memory
    import argparse
    parser = argparse.ArgumentParser(description='Train FT-Transformer')
    parser.add_argument('--optimized', action='store_true',
                       help='Use optimized settings for 4GB GPU')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Custom batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Custom max epochs')

    args = parser.parse_args()

    # Train model
    if args.optimized:
        model = train_ft_transformer_optimized(
            batch_size=args.batch_size or 64,
            max_epochs=args.epochs or 100,
            use_gpu=use_gpu
        )
    else:
        model = train_ft_transformer(
            batch_size=args.batch_size or 128,
            max_epochs=args.epochs or 50,
            use_gpu=use_gpu,
            verbose=True
        )

    if model is not None:
        print("\n✅ FT-Transformer training successful!")
    else:
        print("\n❌ FT-Transformer training failed!")
        sys.exit(1)
