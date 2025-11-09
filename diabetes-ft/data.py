import os
import pandas as pd
import kagglehub
from preprocessing_pipeline import PreprocessingPipeline, load_and_combine_datasets


def download_and_prepare_data(
    imputation_strategy: str = 'knn',
    balancing_strategy: str = 'smote',
    apply_balancing: bool = True,
    combine_datasets_list: list = None
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

    Returns:
        train_df, val_df, test_df
    """

    print("\n" + "=" * 80)
    print("DIABETES DATASET PREPARATION")
    print("=" * 80)

    # Download dataset
    print("\n[Step 1] Downloading diabetes dataset from Kaggle...")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

    # Load the main dataset
    csv_file = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
    print(f"[Step 2] Loading dataset from {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"   - Dataset shape: {df.shape}")
    print(f"   - Features: {df.shape[1] - 1}")
    print(f"   - Samples: {df.shape[0]}")

    # Combine with additional datasets if provided
    if combine_datasets_list:
        print(f"\n[Step 3] Combining with additional datasets...")
        datasets = [df] + [pd.read_csv(p) for p in combine_datasets_list if os.path.exists(p)]
        pipeline_temp = PreprocessingPipeline()
        df = pipeline_temp.combine_datasets(datasets, remove_duplicates=True)
        print(f"   - Combined dataset size: {len(df)}")
    else:
        print(f"\n[Step 3] No additional datasets to combine (skipping)")

    # Initialize preprocessing pipeline
    print(f"\n[Step 4] Initializing preprocessing pipeline...")
    print(f"   - Imputation strategy: {imputation_strategy}")
    print(f"   - Balancing strategy: {balancing_strategy}")
    print(f"   - Apply balancing: {apply_balancing}")

    pipeline = PreprocessingPipeline(
        imputation_strategy=imputation_strategy,
        balancing_strategy=balancing_strategy,
        random_state=42
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
    download_and_prepare_data(
        imputation_strategy='knn',      # knn, iterative, mean, median, most_frequent
        balancing_strategy='smote',     # smote, adasyn, borderline, svmsmote, etc.
        apply_balancing=True,           # Set to True to handle class imbalance
        combine_datasets_list=None      # Add list of CSV paths to combine datasets
    )
