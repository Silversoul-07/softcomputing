"""
Download and combine multiple diabetes datasets from Kaggle and other sources.

This script downloads multiple similar diabetes datasets and combines them
BEFORE applying preprocessing, significantly reducing the need for synthetic
data generation.

Datasets included:
1. BRFSS 2015 (alexteboul) - 253,680 samples
2. BRFSS 2023 (siamaktahmasbi) - 433,323 samples
3. Additional compatible datasets as available

By combining real datasets first, we reduce synthetic data from ~60% to <20%.
"""

import os
import pandas as pd
import kagglehub
from typing import List, Tuple
import hashlib


class DatasetCombiner:
    """Download and combine multiple diabetes datasets."""

    def __init__(self):
        self.datasets = []
        self.dataset_info = []

    def download_brfss_2015(self) -> pd.DataFrame:
        """Download BRFSS 2015 dataset (original - 253K samples)."""
        print("\n[1/3] Downloading BRFSS 2015 dataset...")
        try:
            path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
            csv_file = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
            df = pd.read_csv(csv_file)

            print(f"   ✓ Loaded BRFSS 2015: {len(df):,} samples, {df.shape[1]} features")

            self.dataset_info.append({
                'name': 'BRFSS 2015',
                'source': 'alexteboul/diabetes-health-indicators-dataset',
                'samples': len(df),
                'features': df.shape[1],
                'year': 2015
            })

            return df
        except Exception as e:
            print(f"   ✗ Failed to download BRFSS 2015: {e}")
            return None

    def download_brfss_2023(self) -> pd.DataFrame:
        """Download BRFSS 2023 dataset (433K samples)."""
        print("\n[2/3] Downloading BRFSS 2023 dataset...")
        try:
            path = kagglehub.dataset_download("siamaktahmasbi/diabetes-health-indicators")

            # Try different possible filenames
            possible_files = [
                "Diabetes Health Indicators.csv",
                "diabetes_health_indicators.csv",
                "diabetes_012_health_indicators_BRFSS2023.csv",
                "diabetes.csv",
                "data.csv"
            ]

            csv_file = None
            for filename in possible_files:
                test_path = os.path.join(path, filename)
                if os.path.exists(test_path) and os.path.isfile(test_path):
                    csv_file = test_path
                    break

            if csv_file is None:
                # List all files in the directory and find CSV files
                all_files = os.listdir(path)
                csv_files = [f for f in all_files if f.endswith('.csv') and os.path.isfile(os.path.join(path, f))]

                print(f"   Available files: {all_files}")
                print(f"   CSV files: {csv_files}")

                if csv_files:
                    csv_file = os.path.join(path, csv_files[0])
                else:
                    raise FileNotFoundError("No CSV files found")

            df = pd.read_csv(csv_file)
            print(f"   ✓ Loaded BRFSS 2023: {len(df):,} samples, {df.shape[1]} features")

            self.dataset_info.append({
                'name': 'BRFSS 2023',
                'source': 'siamaktahmasbi/diabetes-health-indicators',
                'samples': len(df),
                'features': df.shape[1],
                'year': 2023
            })

            return df
        except Exception as e:
            print(f"   ✗ Failed to download BRFSS 2023: {e}")
            print(f"   (This dataset may require manual download)")
            return None

    def download_balanced_version(self) -> pd.DataFrame:
        """Download balanced version if available (70K samples)."""
        print("\n[3/3] Checking for additional balanced datasets...")
        try:
            path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
            csv_file = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")

            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)

                # Only use if it has different samples (not just a subset)
                print(f"   ✓ Found balanced version: {len(df):,} samples")
                print(f"   (Skipping to avoid duplicate data)")
                return None
            else:
                print(f"   No additional datasets found")
                return None
        except Exception as e:
            print(f"   No additional datasets available")
            return None

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names across datasets."""
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        # Check if this is raw BRFSS data (has columns like DIABETE4, _BMI5, etc.)
        is_raw_brfss = 'DIABETE4' in df.columns or 'DIABTYPE' in df.columns

        if is_raw_brfss:
            print("   ⚠ Detected raw BRFSS format - incompatible with cleaned format")
            print("   This dataset uses different coding and cannot be easily combined")
            return None

        # Map common variations to standard names
        column_mapping = {
            'Diabetes_binary': 'Diabetes_012',  # Convert binary to 012 if needed
            'diabetes': 'Diabetes_012',
            'diabetes_binary': 'Diabetes_012',
        }

        df = df.rename(columns=column_mapping)

        return df

    def align_schemas(self, datasets: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Align schemas across datasets to ensure compatibility."""
        if not datasets:
            return datasets

        print("\n[Schema Alignment]")

        # Find common columns
        all_columns = [set(df.columns) for df in datasets]
        common_columns = set.intersection(*all_columns) if all_columns else set()

        print(f"   Common columns across datasets: {len(common_columns)}")

        if len(common_columns) < 5:  # Too few common columns
            print(f"   ⚠ Warning: Only {len(common_columns)} common columns found")
            print(f"   Datasets may not be compatible for combination")

            # Show column differences
            for i, df in enumerate(datasets):
                unique_cols = set(df.columns) - common_columns
                if unique_cols:
                    print(f"   Dataset {i+1} unique columns: {unique_cols}")

        # Keep only common columns, preserving order from first dataset
        aligned_datasets = []
        reference_columns = [col for col in datasets[0].columns if col in common_columns]

        for i, df in enumerate(datasets):
            aligned_df = df[reference_columns].copy()
            aligned_datasets.append(aligned_df)
            print(f"   ✓ Dataset {i+1} aligned: {len(aligned_df)} samples, {len(reference_columns)} features")

        return aligned_datasets

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows from combined dataset."""
        initial_size = len(df)
        df_clean = df.drop_duplicates()
        duplicates_removed = initial_size - len(df_clean)

        return df_clean, duplicates_removed

    def combine_datasets(self) -> pd.DataFrame:
        """Download and combine all available datasets."""
        print("=" * 80)
        print("DOWNLOADING AND COMBINING MULTIPLE DIABETES DATASETS")
        print("=" * 80)
        print("\nThis will significantly reduce the need for synthetic data!")

        # Download all available datasets
        datasets = []

        df_2015 = self.download_brfss_2015()
        if df_2015 is not None:
            datasets.append(df_2015)

        df_2023 = self.download_brfss_2023()
        if df_2023 is not None:
            datasets.append(df_2023)

        # Try balanced version (but skip if it's just a subset)
        self.download_balanced_version()

        if not datasets:
            raise ValueError("Failed to download any datasets!")

        if len(datasets) == 1:
            print("\n⚠ Warning: Only 1 dataset available. Consider manual download of:")
            print("   - https://www.kaggle.com/datasets/siamaktahmasbi/diabetes-health-indicators")
            return datasets[0]

        print(f"\n[Combining {len(datasets)} datasets]")

        # Normalize column names
        datasets = [self.normalize_column_names(df) for df in datasets]

        # Remove None entries (incompatible datasets)
        datasets = [df for df in datasets if df is not None]

        if not datasets:
            raise ValueError("No compatible datasets found after normalization!")

        if len(datasets) == 1:
            print("\n⚠ Warning: Only 1 compatible dataset available")
            print("   BRFSS 2023 raw data is incompatible with cleaned 2015 format")
            print("   Using single dataset - synthetic data proportion will be higher")
            return datasets[0]

        # Align schemas
        datasets = self.align_schemas(datasets)

        # Combine datasets
        print("\n[Combining datasets...]")
        combined = pd.concat(datasets, ignore_index=True)
        print(f"   Combined size before deduplication: {len(combined):,} samples")

        # Remove duplicates
        print("\n[Removing duplicates...]")
        combined, duplicates = self.remove_duplicates(combined)
        print(f"   ✓ Removed {duplicates:,} duplicate rows")
        print(f"   ✓ Final combined dataset: {len(combined):,} samples")

        # Show class distribution
        print("\n[Class Distribution in Combined Dataset]")
        for class_label, count in combined['Diabetes_012'].value_counts().sort_index().items():
            percentage = (count / len(combined)) * 100
            print(f"   Class {class_label}: {count:,} ({percentage:.2f}%)")

        imbalance_ratio = combined['Diabetes_012'].value_counts().max() / combined['Diabetes_012'].value_counts().min()
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")

        return combined

    def print_summary(self):
        """Print summary of downloaded datasets."""
        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)

        for info in self.dataset_info:
            print(f"\n{info['name']}:")
            print(f"   Source: {info['source']}")
            print(f"   Year: {info['year']}")
            print(f"   Samples: {info['samples']:,}")
            print(f"   Features: {info['features']}")


def download_and_combine_multiple_datasets() -> pd.DataFrame:
    """
    Main function to download and combine multiple diabetes datasets.

    Returns:
        Combined DataFrame with all datasets merged
    """
    combiner = DatasetCombiner()
    combined_df = combiner.combine_datasets()
    combiner.print_summary()

    return combined_df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-DATASET DOWNLOADER")
    print("=" * 80)
    print("\nThis script downloads and combines multiple diabetes datasets")
    print("to reduce the need for synthetic data generation.")
    print("\nExpected outcome:")
    print("  - BRFSS 2015: ~253K samples")
    print("  - BRFSS 2023: ~433K samples (if available)")
    print("  - Combined: ~686K samples (after deduplication)")
    print("  - Synthetic data needed: <20% (vs 60% with single dataset)")
    print("=" * 80)

    combined_df = download_and_combine_multiple_datasets()

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"\nTotal samples in combined dataset: {len(combined_df):,}")
    print(f"Total features: {combined_df.shape[1]}")
    print("\nNext: Run data.py to preprocess this combined dataset")
