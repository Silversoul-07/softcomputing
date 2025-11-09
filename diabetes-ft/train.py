"""
Main Training Script

This script trains both FT-Transformer and XGBoost models.

NEW STRUCTURE:
- Baseline (XGBoost) is now in: train_baseline.py
- FT-Transformer is now in: train_ft_transformer.py

This file now serves as a convenient wrapper to train both models.

Usage:
    # Train both models
    python train.py

    # Train only baseline
    python train_baseline.py

    # Train only FT-Transformer
    python train_ft_transformer.py

    # Train FT-Transformer with optimized settings for 4GB GPU
    python train_ft_transformer.py --optimized
"""

import os
import sys


def main():
    """Train both baseline and FT-Transformer models."""

    print("\n" + "="*80)
    print("DIABETES PREDICTION MODEL TRAINING")
    print("="*80)
    print("\nThis script will train both models:")
    print("  1. XGBoost Baseline (train_baseline.py)")
    print("  2. FT-Transformer (train_ft_transformer.py)")
    print("\nNote: You can also train them separately using the individual scripts.")
    print("="*80)

    # Check if data exists
    if not os.path.exists('data/train.csv'):
        print("\n❌ Training data not found!")
        print("\nPlease run data preparation first:")
        print("   .venv/bin/python data.py")
        print("\nThis will download and preprocess the diabetes dataset.")
        sys.exit(1)

    # Import training functions
    try:
        from train_baseline import train_xgboost_baseline
        from train_ft_transformer import train_ft_transformer
    except ImportError as e:
        print(f"\n❌ Failed to import training modules: {e}")
        sys.exit(1)

    # Train XGBoost baseline
    print("\n\n" + "="*80)
    print("STEP 1/2: TRAINING XGBOOST BASELINE")
    print("="*80)

    xgb_model = train_xgboost_baseline(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        use_gpu=True,
        verbose=True
    )

    if xgb_model is None:
        print("\n❌ XGBoost training failed!")
        sys.exit(1)

    # Train FT-Transformer
    print("\n\n" + "="*80)
    print("STEP 2/2: TRAINING FT-TRANSFORMER")
    print("="*80)

    ft_model = train_ft_transformer(
        batch_size=128,
        max_epochs=50,
        embed_dim=64,
        num_heads=4,
        num_attn_blocks=2,
        learning_rate=1e-3,
        early_stopping_patience=10,
        use_gpu=True,
        verbose=True
    )

    if ft_model is None:
        print("\n❌ FT-Transformer training failed!")
        sys.exit(1)

    # Final summary
    print("\n\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print("\n✅ XGBoost baseline trained successfully")
    print("\n✅ FT-Transformer trained successfully")
    print("\nModels saved to:")
    print("  - models/xgboost_baseline.pkl")
    print("  - models/ft_transformer/")
    print("\nNext step: Run 'python evaluate.py' to evaluate on test set")
    print("="*80)


if __name__ == "__main__":
    main()
