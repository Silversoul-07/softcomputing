"""
Optimized Training Script for 4GB GPU

This script trains models optimized for memory-constrained environments.

NEW STRUCTURE:
- Baseline (XGBoost) is in: train_baseline.py
- FT-Transformer is in: train_ft_transformer.py --optimized

This file now serves as a convenient wrapper for optimized training.

Usage:
    # Train both models (optimized for 4GB GPU)
    python train_optimized.py

    # Train only FT-Transformer (optimized)
    python train_ft_transformer.py --optimized

    # Train only baseline
    python train_baseline.py
"""

import os
import sys


def main():
    """Train both models with optimized settings for 4GB GPU."""

    print("\n" + "="*80)
    print("OPTIMIZED TRAINING FOR 4GB GPU")
    print("="*80)
    print("\nThis script uses memory-optimized settings:")
    print("  - Smaller batch sizes")
    print("  - Reduced model dimensions")
    print("  - Gradient accumulation")
    print("  - Mixed precision training")
    print("="*80)

    # Check if data exists
    if not os.path.exists('data/train.csv'):
        print("\n❌ Training data not found!")
        print("\nPlease run data preparation first:")
        print("   .venv/bin/python data.py")
        sys.exit(1)

    # Import training functions
    try:
        from train_baseline import train_xgboost_baseline
        from train_ft_transformer import train_ft_transformer_optimized
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

    # Train FT-Transformer (optimized)
    print("\n\n" + "="*80)
    print("STEP 2/2: TRAINING FT-TRANSFORMER (OPTIMIZED)")
    print("="*80)

    ft_model = train_ft_transformer_optimized(
        batch_size=64,
        max_epochs=100,
        embed_dim=48,
        use_gpu=True
    )

    if ft_model is None:
        print("\n❌ FT-Transformer training failed!")
        sys.exit(1)

    # Final summary
    print("\n\n" + "="*80)
    print("OPTIMIZED TRAINING COMPLETE!")
    print("="*80)
    print("\n✅ XGBoost baseline trained successfully")
    print("\n✅ FT-Transformer (optimized) trained successfully")
    print("\nModels saved to:")
    print("  - models/xgboost_baseline.pkl")
    print("  - models/ft_transformer/")
    print("\nMemory optimizations used:")
    print("  - Batch size: 64 (vs 128 standard)")
    print("  - Embedding dim: 48 (vs 64 standard)")
    print("  - Gradient accumulation: 2 steps")
    print("  - Mixed precision: 16-bit")
    print("\nNext step: Run 'python evaluate.py' to evaluate on test set")
    print("="*80)


if __name__ == "__main__":
    main()
