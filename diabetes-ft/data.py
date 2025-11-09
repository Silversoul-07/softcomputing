import os
import pandas as pd
import kagglehub
from preprocessing_pipeline import PreprocessingPipeline, load_and_combine_datasets
from download_multi_datasets import download_and_combine_multiple_datasets


def download_and_prepare_data(
    imputation_strategy: str = 'knn',
    balancing_strategy: str = 'smote',
    apply_balancing: bool = True,
    combine_datasets_list: list = None,
    use_multi_datasets: bool = False,
    target_imbalance_ratio: float = None
):
    """
    Download diabetes dataset from Kaggle and prepare train/val/test splits
    using advanced preprocessing pipeline.

    Args:
        imputation_strategy: Method for null value imputation
            Options: 'knn', 'iterative', 'mean', 'median', 'most_frequent'
        balancing_strategy: Method for class imbalance handling
            Options: 'smote', 'adasyn', 'borderline', 'svmsmote', 'random',
                     'smote_tomek', 'smote_enn'
        apply_balancing: Whether to apply class balancing
        combine_datasets_list: Optional list of additional CSV paths to combine
        use_multi_datasets: If True, downloads and combines multiple datasets
                           from Kaggle (BRFSS 2015 + 2023) to reduce synthetic data
        target_imbalance_ratio: Target max/min class ratio after balancing
                               None = full balance (~60% synthetic)
                               10 = moderate balance (~7% synthetic)
                               20 = light balance (~3% synthetic)

    Returns:
        train_df, val_df, test_df
    """

    print("\n" + "=" * 80)
    print("DIABETES DATASET PREPARATION")
    print("=" * 80)

    # Download and combine multiple datasets if requested
    if use_multi_datasets:
        print("\n[Step 1-2] Downloading and combining multiple datasets...")
        print("   This will significantly reduce synthetic data requirements!")
        try:
            df = download_and_combine_multiple_datasets()
            print(f"\n   ✓ Combined dataset loaded: {len(df):,} samples")
        except Exception as e:
            print(f"\n   ⚠ Failed to download multiple datasets: {e}")
            print(f"   ⚠ Falling back to single dataset...")
            use_multi_datasets = False

    # Fallback to single dataset if multi-dataset download failed
    if not use_multi_datasets:
        print("\n[Step 1] Downloading single BRFSS 2015 dataset...")
        path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
        csv_file = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
        print(f"[Step 2] Loading dataset from {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"   - Dataset shape: {df.shape}")
        print(f"   - Features: {df.shape[1] - 1}")
        print(f"   - Samples: {df.shape[0]}")

    # Combine with additional CSV files if provided
    if combine_datasets_list:
        print(f"\n[Step 3] Combining with additional CSV files...")
        datasets = [df] + [pd.read_csv(p) for p in combine_datasets_list if os.path.exists(p)]
        pipeline_temp = PreprocessingPipeline()
        df = pipeline_temp.combine_datasets(datasets, remove_duplicates=True)
        print(f"   - Combined dataset size: {len(df)}")
    else:
        print(f"\n[Step 3] No additional CSV files to combine (skipping)")

    # Initialize preprocessing pipeline
    print(f"\n[Step 4] Initializing preprocessing pipeline...")
    print(f"   - Imputation strategy: {imputation_strategy}")
    print(f"   - Balancing strategy: {balancing_strategy}")
    print(f"   - Apply balancing: {apply_balancing}")

    if target_imbalance_ratio is not None:
        print(f"   - Target imbalance ratio: {target_imbalance_ratio}:1 (partial balancing)")
        print(f"   - Expected synthetic data: ~{(1 - 1/target_imbalance_ratio) * 10:.1f}%")
    else:
        print(f"   - Target imbalance ratio: 1:1 (full balancing)")
        print(f"   - Expected synthetic data: ~60%")

    pipeline = PreprocessingPipeline(
        imputation_strategy=imputation_strategy,
        balancing_strategy=balancing_strategy,
        random_state=42,
        target_imbalance_ratio=target_imbalance_ratio
    )

    # Run preprocessing
    print(f"\n[Step 5] Running preprocessing pipeline...")
    train_df, val_df, test_df = pipeline.preprocess(
        df=df,
        target_column='Diabetes_012',
        test_size=0.30,
        val_ratio=0.50,
        apply_balancing=apply_balancing,
        balance_train_only=True
    )

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Save datasets
    print(f"\n[Step 6] Saving processed datasets...")
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print(f"   ✓ data/train.csv ({len(train_df)} samples)")
    print(f"   ✓ data/val.csv ({len(val_df)} samples)")
    print(f"   ✓ data/test.csv ({len(test_df)} samples)")

    # Save preprocessing report
    pipeline.save_report('data/preprocessing_report.json')

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)

    return train_df, val_df, test_df


if __name__ == "__main__":
    # You can customize the preprocessing here

    # Option 1: Partial balancing (RECOMMENDED - only ~7% synthetic data)
    download_and_prepare_data(
        imputation_strategy='knn',
        balancing_strategy='smote',
        apply_balancing=True,
        target_imbalance_ratio=10       # 10:1 ratio = ~7% synthetic data
    )

    # Option 2: Full balancing (~60% synthetic data)
    # download_and_prepare_data(
    #     imputation_strategy='knn',
    #     balancing_strategy='smote',
    #     apply_balancing=True,
    #     target_imbalance_ratio=None     # Full balance = ~60% synthetic
    # )

    # Option 3: No balancing (0% synthetic data)
    # download_and_prepare_data(
    #     imputation_strategy='knn',
    #     balancing_strategy='smote',
    #     apply_balancing=False,
    #     target_imbalance_ratio=None     # No balancing = 0% synthetic
    # )
