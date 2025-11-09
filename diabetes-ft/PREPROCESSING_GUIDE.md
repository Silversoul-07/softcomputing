# Advanced Preprocessing Pipeline Guide

## Overview

This preprocessing pipeline provides comprehensive data preparation with:
- **Advanced null value imputation** using multiple techniques
- **Class imbalance handling** with various oversampling strategies
- **Dataset combination** capabilities
- **Comprehensive validation and reporting**
- **Measurable outcomes** with detailed statistics

## Features

### 1. Null Value Imputation

Multiple imputation strategies available:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `knn` | K-Nearest Neighbors Imputation | Default, captures local patterns |
| `iterative` | MICE (Multivariate Iterative Imputation) | Complex feature relationships |
| `mean` | Simple mean imputation | Fast, numerical features |
| `median` | Simple median imputation | Robust to outliers |
| `most_frequent` | Mode imputation | Categorical features |

### 2. Class Imbalance Handling

Multiple oversampling strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `smote` | SMOTE (Synthetic Minority Over-sampling) | Default, balanced datasets |
| `adasyn` | Adaptive Synthetic Sampling | Focus on hard examples |
| `borderline` | BorderlineSMOTE | Borderline cases |
| `svmsmote` | SVM-based SMOTE | High-dimensional data |
| `random` | Random oversampling | Quick baseline |
| `smote_tomek` | SMOTE + Tomek links (hybrid) | Clean boundaries |
| `smote_enn` | SMOTE + ENN (hybrid) | Remove noisy samples |

## Usage Examples

### Basic Usage

```python
from preprocessing_pipeline import PreprocessingPipeline
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Initialize pipeline with default settings
pipeline = PreprocessingPipeline(
    imputation_strategy='knn',
    balancing_strategy='smote',
    random_state=42
)

# Run preprocessing
train_df, val_df, test_df = pipeline.preprocess(
    df=df,
    target_column='target',
    test_size=0.30,
    val_ratio=0.50,
    apply_balancing=True
)

# Save report
pipeline.save_report('preprocessing_report.json')
```

### Using data.py with Custom Settings

```python
from data import download_and_prepare_data

# Option 1: KNN imputation + SMOTE balancing (default)
train_df, val_df, test_df = download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=True
)

# Option 2: Iterative imputation + ADASYN balancing
train_df, val_df, test_df = download_and_prepare_data(
    imputation_strategy='iterative',
    balancing_strategy='adasyn',
    apply_balancing=True
)

# Option 3: Mean imputation + BorderlineSMOTE
train_df, val_df, test_df = download_and_prepare_data(
    imputation_strategy='mean',
    balancing_strategy='borderline',
    apply_balancing=True
)

# Option 4: No balancing (for baseline comparison)
train_df, val_df, test_df = download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=False
)
```

### Combining Multiple Datasets

```python
from data import download_and_prepare_data

# Combine main dataset with additional datasets
train_df, val_df, test_df = download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=True,
    combine_datasets_list=[
        'additional_data1.csv',
        'additional_data2.csv'
    ]
)
```

### Manual Dataset Combination

```python
from preprocessing_pipeline import load_and_combine_datasets

# Load and combine multiple CSV files
combined_df = load_and_combine_datasets([
    'dataset1.csv',
    'dataset2.csv',
    'dataset3.csv'
])

# Then use in preprocessing
pipeline = PreprocessingPipeline()
train_df, val_df, test_df = pipeline.preprocess(combined_df)
```

## Results from Current Run

### Initial Dataset
- **Total samples**: 253,680
- **Features**: 21
- **Null values**: 0 (0.00%)

### Class Distribution (Initial)
- **Class 0 (No Diabetes)**: 213,703 (84.24%)
- **Class 1 (Prediabetes)**: 4,631 (1.83%)
- **Class 2 (Diabetes)**: 35,346 (13.93%)
- **Imbalance ratio**: 46.15:1 (highly imbalanced!)

### After Preprocessing
- **Train set**: 448,776 samples (balanced with SMOTE)
- **Validation set**: 38,052 samples (original distribution)
- **Test set**: 38,052 samples (original distribution)

### Class Distribution (Training Set After SMOTE)
- **Class 0**: 149,592 (33.33%)
- **Class 1**: 149,592 (33.33%)
- **Class 2**: 149,592 (33.33%)
- **Imbalance ratio**: 1.00:1 (perfectly balanced!)
- **Samples added**: 271,200 synthetic samples

### Validation Results
✅ **No null values in train set**: True
✅ **No null values in val set**: True
✅ **No null values in test set**: True
✅ **Class imbalance handled**: True
✅ **Imbalance ratio reduced**: 46.15:1 → 1.00:1

## Pipeline Output

The preprocessing pipeline creates:

1. **data/train.csv** - Balanced training set with synthetic samples
2. **data/val.csv** - Validation set with original distribution
3. **data/test.csv** - Test set with original distribution
4. **data/preprocessing_report.json** - Comprehensive preprocessing report

## Preprocessing Report

The JSON report includes:

```json
{
  "initial_nulls": {
    "total_cells": 5581080,
    "total_nulls": 0,
    "null_percentage": 0.0,
    "columns_with_nulls": {},
    "rows_with_nulls": 0
  },
  "initial_distribution": {
    "total_samples": 253680,
    "classes": {
      "0": {"count": 213703, "percentage": 84.24},
      "1": {"count": 4631, "percentage": 1.83},
      "2": {"count": 35346, "percentage": 13.93}
    },
    "imbalance_ratio": 46.15
  },
  "train_distribution_before": {...},
  "train_distribution_after": {...},
  "validation": {
    "train_no_nulls": true,
    "val_no_nulls": true,
    "test_no_nulls": true,
    "all_nulls_handled": true,
    "class_balanced": true,
    "train_imbalance_ratio": 1.0
  }
}
```

## Best Practices

### 1. Choosing Imputation Strategy

- **KNN (default)**: Best for most cases, captures local patterns
- **Iterative**: Use when features have complex relationships
- **Mean/Median**: Fast, good for simple missing patterns
- **Most Frequent**: Use for categorical features

### 2. Choosing Balancing Strategy

- **SMOTE (default)**: Good starting point for most datasets
- **ADASYN**: Better for datasets with varying density
- **BorderlineSMOTE**: Focus on decision boundary
- **SMOTE-Tomek/SMOTE-ENN**: Use when you want cleaner boundaries

### 3. When to Apply Balancing

**Apply balancing (True)** when:
- Imbalance ratio > 3:1
- Minority class performance is critical
- Using tree-based models or neural networks

**Skip balancing (False)** when:
- Dataset is already balanced
- Using algorithms that handle imbalance well (XGBoost with scale_pos_weight)
- Want to maintain original distribution for baseline

### 4. Validation Strategy

Always:
- Keep validation and test sets unbalanced (original distribution)
- Only balance the training set
- Evaluate on original distribution to match real-world performance

## Command Line Usage

Run preprocessing with default settings:
```bash
.venv/bin/python data.py
```

## Comparison with Previous Approach

| Feature | Old Approach | New Approach |
|---------|-------------|--------------|
| Null handling | Simple (StandardScaler only) | Advanced (KNN, Iterative, etc.) |
| Class balancing | None | 7 different strategies |
| Dataset combination | Not supported | Full support |
| Reporting | Minimal console output | Comprehensive JSON report |
| Validation | None | Full validation suite |
| Measurability | Limited | Complete metrics |

## Troubleshooting

### Issue: "Out of memory"
**Solution**: Use simpler imputation strategy (mean/median) or reduce batch size

### Issue: "SMOTE fails with k_neighbors error"
**Solution**: Minority class too small, try reducing k_neighbors in strategy or use random oversampling

### Issue: "Training takes too long"
**Solution**:
- Use 'random' balancing instead of SMOTE
- Use 'mean' imputation instead of 'knn' or 'iterative'
- Reduce dataset size before preprocessing

## Advanced Configuration

### Custom Pipeline Example

```python
from preprocessing_pipeline import PreprocessingPipeline

# Create custom pipeline
pipeline = PreprocessingPipeline(
    imputation_strategy='iterative',
    balancing_strategy='borderline',
    random_state=42
)

# Analyze nulls before processing
null_info = pipeline.analyze_nulls(df)
print(f"Null percentage: {null_info['null_percentage']:.2f}%")

# Analyze class distribution
dist_info = pipeline.analyze_class_distribution(df['target'])
print(f"Imbalance ratio: {dist_info['imbalance_ratio']:.2f}:1")

# Custom preprocessing
train_df, val_df, test_df = pipeline.preprocess(
    df=df,
    target_column='target',
    test_size=0.20,  # 80/10/10 split
    val_ratio=0.50,
    apply_balancing=True,
    balance_train_only=True
)
```

## Metrics and Validation

The pipeline ensures:

1. ✅ **Zero null values** in all output datasets
2. ✅ **Balanced classes** in training set (if enabled)
3. ✅ **Original distribution** preserved in validation/test sets
4. ✅ **Proper scaling** with StandardScaler
5. ✅ **No data leakage** (scaler/imputer fit only on training data)
6. ✅ **Stratified splits** maintaining class proportions

## Performance Impact

With current dataset (253,680 samples, 21 features):

- **Processing time**: ~60 seconds
- **Memory usage**: ~2GB peak
- **Output size**: ~210MB (train + val + test)
- **Synthetic samples created**: 271,200

## Next Steps

After preprocessing:

1. Train models using `train.py` or `train_optimized.py`
2. Evaluate using `evaluate.py`
3. Compare different preprocessing strategies
4. Tune hyperparameters based on validation performance

## References

- SMOTE: Chawla et al. (2002)
- ADASYN: He et al. (2008)
- KNN Imputation: Troyanskaya et al. (2001)
- Iterative Imputation (MICE): van Buuren & Groothuis-Oudshoorn (2011)
