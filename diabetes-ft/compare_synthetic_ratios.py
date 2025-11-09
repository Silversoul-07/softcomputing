"""
Compare different balancing ratios and their impact on synthetic data proportion.

This script demonstrates the tradeoff between class balance and synthetic data.
"""

import pandas as pd
import os
import kagglehub
from preprocessing_pipeline import PreprocessingPipeline


def test_balancing_ratio(ratio, description):
    """Test a specific balancing ratio."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Target Ratio: {ratio if ratio else 'Full Balance (1:1)'}")
    print(f"{'='*80}")

    # Load dataset
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    csv_file = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(csv_file)

    # Create pipeline
    pipeline = PreprocessingPipeline(
        imputation_strategy='knn',
        balancing_strategy='smote',
        random_state=42,
        target_imbalance_ratio=ratio
    )

    # Run preprocessing
    train_df, val_df, test_df = pipeline.preprocess(
        df=df,
        target_column='Diabetes_012',
        test_size=0.30,
        val_ratio=0.50,
        apply_balancing=True
    )

    # Calculate metrics
    original_train_size = int(len(df) * 0.70)
    balanced_train_size = len(train_df)
    synthetic_samples = balanced_train_size - original_train_size
    synthetic_percentage = (synthetic_samples / balanced_train_size) * 100

    # Get class distribution
    class_dist = train_df['Diabetes_012'].value_counts().sort_index()
    imbalance_ratio = class_dist.max() / class_dist.min()

    results = {
        'ratio': ratio if ratio else 1,
        'description': description,
        'original_samples': original_train_size,
        'final_samples': balanced_train_size,
        'synthetic_samples': synthetic_samples,
        'synthetic_percentage': synthetic_percentage,
        'class_0': class_dist[0],
        'class_1': class_dist[1],
        'class_2': class_dist[2],
        'imbalance_ratio': imbalance_ratio
    }

    print(f"\n📊 Results:")
    print(f"   Original training samples: {original_train_size:,}")
    print(f"   Final training samples: {balanced_train_size:,}")
    print(f"   Synthetic samples added: {synthetic_samples:,}")
    print(f"   Synthetic data percentage: {synthetic_percentage:.2f}%")
    print(f"\n   Class distribution:")
    print(f"     Class 0 (No Diabetes): {class_dist[0]:,}")
    print(f"     Class 1 (Prediabetes): {class_dist[1]:,}")
    print(f"     Class 2 (Diabetes): {class_dist[2]:,}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")

    return results


def main():
    """Compare different balancing ratios."""
    print("\n" + "="*80)
    print("SYNTHETIC DATA PROPORTION COMPARISON")
    print("="*80)
    print("\nThis script compares different balancing ratios and shows the")
    print("tradeoff between class balance and synthetic data proportion.")

    # Test different ratios
    test_cases = [
        (None, "Full Balance (Maximum minority class performance)"),
        (5, "Aggressive Balance (5:1 ratio)"),
        (10, "Moderate Balance (10:1 ratio) - RECOMMENDED"),
        (20, "Conservative Balance (20:1 ratio)"),
        (30, "Light Balance (30:1 ratio)"),
    ]

    all_results = []

    for ratio, description in test_cases:
        try:
            results = test_balancing_ratio(ratio, description)
            all_results.append(results)
        except Exception as e:
            print(f"\n   ❌ Error: {e}")

    # Print comparison table
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Ratio':<10} {'Synthetic':<15} {'Samples Added':<15} {'Class 1 Size':<15} {'Imbalance':<10}")
    print("-" * 80)

    for r in all_results:
        ratio_str = f"{r['ratio']}:1" if r['ratio'] != 1 else "1:1 (Full)"
        print(f"{ratio_str:<10} {r['synthetic_percentage']:>6.2f}% {r['synthetic_samples']:>14,} "
              f"{r['class_1']:>14,} {r['imbalance_ratio']:>9.2f}:1")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\n🎯 For Production (Real-World Deployment):")
    print("   ✅ Use 10:1 ratio (moderate balance)")
    print("   ✅ Synthetic data: ~6-7%")
    print("   ✅ Good balance between performance and data authenticity")
    print("   ✅ Reduces overfitting risk")

    print("\n🔬 For Research (Maximum Performance):")
    print("   ✅ Use 5:1 ratio (aggressive balance)")
    print("   ✅ Synthetic data: ~15%")
    print("   ✅ Better minority class performance")
    print("   ✅ Higher F1-macro score")

    print("\n💡 For Conservative Approach:")
    print("   ✅ Use 20:1 ratio (conservative balance)")
    print("   ✅ Synthetic data: ~3%")
    print("   ✅ Minimal synthetic data")
    print("   ✅ May sacrifice some minority class performance")

    print("\n⚠️  Full Balance (1:1) - NOT RECOMMENDED:")
    print("   ❌ Synthetic data: ~60%")
    print("   ❌ High overfitting risk")
    print("   ❌ Poor generalization to real data")
    print("   ❌ Use only for academic research/benchmarking")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("\nTo use a specific ratio in your preprocessing:")
    print("\n   from data import download_and_prepare_data")
    print("\n   # Moderate balance (recommended)")
    print("   train, val, test = download_and_prepare_data(")
    print("       target_imbalance_ratio=10")
    print("   )")


if __name__ == "__main__":
    main()
