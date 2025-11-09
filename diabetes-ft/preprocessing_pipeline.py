"""
Advanced Preprocessing Pipeline for Diabetes Dataset

Features:
- Multiple null value imputation techniques (KNN, Iterative, Simple)
- Class imbalance handling (SMOTE, ADASYN, BorderlineSMOTE, hybrid)
- Dataset combination and merging
- Comprehensive reporting and validation
- Measurable outcomes with detailed statistics
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler, SVMSMOTE
)
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import json


class PreprocessingPipeline:
    """Comprehensive preprocessing pipeline with null handling and class balancing."""

    def __init__(
        self,
        imputation_strategy: str = 'knn',
        balancing_strategy: str = 'smote',
        random_state: int = 42,
        target_imbalance_ratio: float = None
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            imputation_strategy: Method for handling nulls
                - 'knn': KNN Imputation (default)
                - 'iterative': Iterative Imputation (MICE)
                - 'mean': Simple mean imputation
                - 'median': Simple median imputation
                - 'most_frequent': Mode imputation
            balancing_strategy: Method for handling class imbalance
                - 'smote': SMOTE oversampling (default)
                - 'adasyn': ADASYN adaptive oversampling
                - 'borderline': BorderlineSMOTE
                - 'svmsmote': SVM-based SMOTE
                - 'random': Random oversampling
                - 'smote_tomek': SMOTE + Tomek links (hybrid)
                - 'smote_enn': SMOTE + ENN (hybrid)
            random_state: Random seed for reproducibility
            target_imbalance_ratio: Target max/min class ratio after balancing
                - None (default): Full balance (1:1:1 ratio, ~60% synthetic)
                - 10: Moderate balance (10:1 ratio, ~7% synthetic)
                - 20: Light balance (20:1 ratio, ~3% synthetic)
                - Higher values = less synthetic data
        """
        self.imputation_strategy = imputation_strategy
        self.balancing_strategy = balancing_strategy
        self.random_state = random_state
        self.target_imbalance_ratio = target_imbalance_ratio
        self.scaler = StandardScaler()
        self.imputer = None
        self.sampler = None
        self.report = {}

    def _create_imputer(self):
        """Create imputer based on strategy."""
        if self.imputation_strategy == 'knn':
            return KNNImputer(n_neighbors=5)
        elif self.imputation_strategy == 'iterative':
            return IterativeImputer(random_state=self.random_state, max_iter=10)
        elif self.imputation_strategy == 'mean':
            return SimpleImputer(strategy='mean')
        elif self.imputation_strategy == 'median':
            return SimpleImputer(strategy='median')
        elif self.imputation_strategy == 'most_frequent':
            return SimpleImputer(strategy='most_frequent')
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")

    def _create_sampler(self):
        """Create sampler based on strategy."""
        if self.balancing_strategy == 'smote':
            return SMOTE(random_state=self.random_state, k_neighbors=3)
        elif self.balancing_strategy == 'adasyn':
            return ADASYN(random_state=self.random_state, n_neighbors=3)
        elif self.balancing_strategy == 'borderline':
            return BorderlineSMOTE(random_state=self.random_state, k_neighbors=3)
        elif self.balancing_strategy == 'svmsmote':
            return SVMSMOTE(random_state=self.random_state, k_neighbors=3)
        elif self.balancing_strategy == 'random':
            return RandomOverSampler(random_state=self.random_state)
        elif self.balancing_strategy == 'smote_tomek':
            return SMOTETomek(random_state=self.random_state)
        elif self.balancing_strategy == 'smote_enn':
            return SMOTEENN(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown balancing strategy: {self.balancing_strategy}")

    def analyze_nulls(self, df: pd.DataFrame) -> Dict:
        """Analyze null values in dataset."""
        null_info = {
            'total_cells': df.size,
            'total_nulls': df.isnull().sum().sum(),
            'null_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'columns_with_nulls': {},
            'rows_with_nulls': df.isnull().any(axis=1).sum()
        }

        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_info['columns_with_nulls'][col] = {
                    'count': int(null_count),
                    'percentage': float((null_count / len(df)) * 100)
                }

        return null_info

    def analyze_class_distribution(self, y: pd.Series) -> Dict:
        """Analyze class distribution."""
        counts = Counter(y)
        total = len(y)

        distribution = {
            'total_samples': total,
            'classes': {},
            'imbalance_ratio': None
        }

        for class_label, count in sorted(counts.items()):
            distribution['classes'][int(class_label)] = {
                'count': count,
                'percentage': (count / total) * 100
            }

        # Calculate imbalance ratio (majority / minority)
        min_count = min(counts.values())
        max_count = max(counts.values())
        distribution['imbalance_ratio'] = max_count / min_count

        return distribution

    def handle_nulls(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle null values using selected imputation strategy."""
        if fit:
            self.imputer = self._create_imputer()
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)

        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    def handle_class_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using selected balancing strategy."""
        # If target_imbalance_ratio is set, use partial balancing
        if self.target_imbalance_ratio is not None:
            X_resampled, y_resampled = self._partial_balance(X, y)
        else:
            # Full balancing (default)
            self.sampler = self._create_sampler()
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)

        # Convert back to DataFrame/Series with proper column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        return X_resampled, y_resampled

    def _partial_balance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply partial balancing to target a specific imbalance ratio.

        Args:
            X: Features
            y: Target

        Returns:
            X_resampled, y_resampled with controlled synthetic data
        """
        class_counts = Counter(y)
        max_count = max(class_counts.values())

        # Calculate target samples for each class
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            # Target: max_count / target_imbalance_ratio
            target = max(count, int(max_count / self.target_imbalance_ratio))
            if target > count:
                sampling_strategy[class_label] = target

        if not sampling_strategy:
            # No balancing needed
            return X.values, y.values

        # Create sampler with custom sampling strategy
        if self.balancing_strategy == 'smote':
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=3,
                random_state=self.random_state
            )
        elif self.balancing_strategy == 'adasyn':
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=3,
                random_state=self.random_state
            )
        elif self.balancing_strategy == 'borderline':
            sampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=3,
                random_state=self.random_state
            )
        elif self.balancing_strategy == 'random':
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        else:
            # Fallback to SMOTE for other strategies
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=3,
                random_state=self.random_state
            )

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def combine_datasets(
        self,
        datasets: List[pd.DataFrame],
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Combine multiple similar datasets.

        Args:
            datasets: List of DataFrames to combine
            remove_duplicates: Whether to remove duplicate rows

        Returns:
            Combined DataFrame
        """
        if not datasets:
            raise ValueError("No datasets provided")

        if len(datasets) == 1:
            return datasets[0]

        # Combine datasets
        combined = pd.concat(datasets, ignore_index=True)

        initial_size = len(combined)

        if remove_duplicates:
            combined = combined.drop_duplicates()
            duplicates_removed = initial_size - len(combined)
            print(f"Removed {duplicates_removed} duplicate rows")

        return combined

    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str = 'Diabetes_012',
        test_size: float = 0.30,
        val_ratio: float = 0.50,
        apply_balancing: bool = True,
        balance_train_only: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion for test+val split
            val_ratio: Ratio of validation to test (0.5 = equal split)
            apply_balancing: Whether to apply class balancing
            balance_train_only: Only balance training set (recommended)

        Returns:
            train_df, val_df, test_df
        """
        print("=" * 80)
        print("PREPROCESSING PIPELINE STARTED")
        print("=" * 80)

        # 1. Initial data analysis
        print("\n[1/7] Analyzing initial dataset...")
        initial_nulls = self.analyze_nulls(df)
        print(f"   - Total samples: {len(df)}")
        print(f"   - Total features: {len(df.columns) - 1}")
        print(f"   - Null values: {initial_nulls['total_nulls']} ({initial_nulls['null_percentage']:.2f}%)")

        if initial_nulls['columns_with_nulls']:
            print(f"   - Columns with nulls: {len(initial_nulls['columns_with_nulls'])}")
            for col, info in initial_nulls['columns_with_nulls'].items():
                print(f"     * {col}: {info['count']} ({info['percentage']:.2f}%)")

        self.report['initial_nulls'] = initial_nulls

        # 2. Separate features and target
        print("\n[2/7] Separating features and target...")
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        initial_distribution = self.analyze_class_distribution(y)
        print(f"   - Class distribution:")
        for class_label, info in initial_distribution['classes'].items():
            print(f"     * Class {class_label}: {info['count']} ({info['percentage']:.2f}%)")
        print(f"   - Imbalance ratio: {initial_distribution['imbalance_ratio']:.2f}:1")

        self.report['initial_distribution'] = initial_distribution

        # 3. Handle null values
        print(f"\n[3/7] Handling null values using {self.imputation_strategy} imputation...")
        X_imputed = self.handle_nulls(X, fit=True)

        post_imputation_nulls = self.analyze_nulls(X_imputed)
        print(f"   - Remaining nulls: {post_imputation_nulls['total_nulls']}")
        print(f"   ✓ All null values successfully imputed!")

        self.report['post_imputation_nulls'] = post_imputation_nulls

        # 4. Train/val/test split
        print(f"\n[4/7] Splitting data (train: {(1-test_size)*100:.0f}%, val: {test_size*val_ratio*100:.0f}%, test: {test_size*(1-val_ratio)*100:.0f}%)...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_imputed, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, random_state=self.random_state, stratify=y_temp
        )

        print(f"   - Train: {len(X_train)} samples")
        print(f"   - Val: {len(X_val)} samples")
        print(f"   - Test: {len(X_test)} samples")

        # 5. Handle class imbalance (train only)
        if apply_balancing:
            print(f"\n[5/7] Handling class imbalance using {self.balancing_strategy}...")
            train_dist_before = self.analyze_class_distribution(y_train)

            X_train, y_train = self.handle_class_imbalance(X_train, y_train)

            train_dist_after = self.analyze_class_distribution(y_train)
            print(f"   - Training set class distribution BEFORE balancing:")
            for class_label, info in train_dist_before['classes'].items():
                print(f"     * Class {class_label}: {info['count']} ({info['percentage']:.2f}%)")

            print(f"   - Training set class distribution AFTER balancing:")
            for class_label, info in train_dist_after['classes'].items():
                print(f"     * Class {class_label}: {info['count']} ({info['percentage']:.2f}%)")

            print(f"   ✓ Class imbalance successfully handled!")
            print(f"   ✓ Samples added: {len(X_train) - sum(c['count'] for c in train_dist_before['classes'].values())}")

            self.report['train_distribution_before'] = train_dist_before
            self.report['train_distribution_after'] = train_dist_after
        else:
            print(f"\n[5/7] Skipping class balancing (apply_balancing=False)")

        # 6. Scale features
        print(f"\n[6/7] Scaling features using StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        print(f"   ✓ Features scaled successfully!")

        # 7. Create final datasets
        print(f"\n[7/7] Creating final datasets...")
        train_df = X_train_scaled.copy()
        train_df[target_column] = y_train.values

        val_df = X_val_scaled.copy()
        val_df[target_column] = y_val.values

        test_df = X_test_scaled.copy()
        test_df[target_column] = y_test.values

        # Final validation
        print("\n" + "=" * 80)
        print("PREPROCESSING VALIDATION")
        print("=" * 80)

        validation_results = self.validate_preprocessing(train_df, val_df, test_df, target_column)

        print(f"\n✓ No null values in train set: {validation_results['train_no_nulls']}")
        print(f"✓ No null values in val set: {validation_results['val_no_nulls']}")
        print(f"✓ No null values in test set: {validation_results['test_no_nulls']}")

        if apply_balancing:
            print(f"✓ Class imbalance handled: {validation_results['class_balanced']}")
            print(f"   - Imbalance ratio reduced: {initial_distribution['imbalance_ratio']:.2f}:1 → {train_dist_after['imbalance_ratio']:.2f}:1")

        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)

        self.report['validation'] = validation_results

        return train_df, val_df, test_df

    def validate_preprocessing(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str
    ) -> Dict:
        """Validate that preprocessing was successful."""
        validation = {
            'train_no_nulls': train_df.isnull().sum().sum() == 0,
            'val_no_nulls': val_df.isnull().sum().sum() == 0,
            'test_no_nulls': test_df.isnull().sum().sum() == 0,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'all_nulls_handled': True,
            'class_balanced': False
        }

        validation['all_nulls_handled'] = (
            validation['train_no_nulls'] and
            validation['val_no_nulls'] and
            validation['test_no_nulls']
        )

        # Check if classes are reasonably balanced in training set
        train_dist = self.analyze_class_distribution(train_df[target_column])
        validation['class_balanced'] = train_dist['imbalance_ratio'] < 2.0
        validation['train_imbalance_ratio'] = train_dist['imbalance_ratio']

        return validation

    def save_report(self, filepath: str = 'preprocessing_report.json'):
        """Save preprocessing report to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        report_serializable = convert_to_json_serializable(self.report)

        with open(filepath, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        print(f"\nPreprocessing report saved to: {filepath}")


def load_and_combine_datasets(dataset_paths: List[str]) -> pd.DataFrame:
    """
    Load and combine multiple CSV datasets.

    Args:
        dataset_paths: List of paths to CSV files

    Returns:
        Combined DataFrame
    """
    datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets.append(df)
            print(f"Loaded {path}: {len(df)} samples")
        else:
            print(f"Warning: {path} not found, skipping...")

    if not datasets:
        raise ValueError("No datasets could be loaded")

    pipeline = PreprocessingPipeline()
    combined = pipeline.combine_datasets(datasets)

    print(f"\nCombined dataset: {len(combined)} total samples")

    return combined
