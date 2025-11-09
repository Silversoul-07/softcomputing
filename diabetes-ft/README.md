# Diabetes Prediction with Advanced Preprocessing

Machine learning project for diabetes prediction using FT-Transformer and XGBoost with comprehensive preprocessing pipeline.

## Features

### Advanced Preprocessing Pipeline ✨

- **Multiple Imputation Strategies**: KNN, Iterative (MICE), Mean, Median, Most Frequent
- **Class Imbalance Handling**: SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, and hybrid methods
- **Dataset Combination**: Merge multiple similar datasets with duplicate removal
- **Comprehensive Validation**: Automated checks ensuring zero null values and balanced classes
- **Detailed Reporting**: JSON reports with before/after statistics

### Models

- **FT-Transformer**: Feature Tokenizer Transformer for tabular data
- **XGBoost**: Gradient boosting baseline with GPU acceleration

## Quick Start

### 1. Install Dependencies

```bash
cd diabetes-ft
uv sync
```

### 2. Prepare Data with Preprocessing

```bash
# Default: KNN imputation + SMOTE balancing
.venv/bin/python data.py
```

### 3. Train Models

```bash
# Standard training (8GB+ GPU)
.venv/bin/python train.py

# Or optimized for 4GB GPU
.venv/bin/python train_optimized.py
```

### 4. Evaluate

```bash
.venv/bin/python evaluate.py
```

## Preprocessing Pipeline

### Measurable Outcomes

Our preprocessing pipeline guarantees:

✅ **Zero null values** in all datasets (train/val/test)
✅ **Controlled class balancing** with configurable synthetic data proportion
✅ **Original distribution** preserved in validation/test sets
✅ **Comprehensive validation** with detailed reports

### Current Dataset Results

**Initial Dataset:**
- Total samples: 253,680
- Class imbalance: 46.15:1 (highly imbalanced!)
- Minority class (Prediabetes): 1.83%

**After Preprocessing (Recommended 10:1 Partial Balance):**
- Training samples: 189,293
- Class imbalance: 10.00:1 (moderate balance)
- **Synthetic data: ~6.2%** (11,717 synthetic / 189,293 total)
- Validation/Test: Original distribution maintained

**Alternative: Full Balance (Not Recommended):**
- Training samples: 448,776
- Class imbalance: 1.00:1 (perfectly balanced)
- **Synthetic data: ~60%** (271,200 synthetic / 448,776 total)
- ⚠️ High overfitting risk with too much synthetic data

### Available Imputation Strategies

```python
# KNN Imputation (default - best balance)
download_and_prepare_data(imputation_strategy='knn')

# Iterative Imputation (MICE - complex relationships)
download_and_prepare_data(imputation_strategy='iterative')

# Simple Mean (fast)
download_and_prepare_data(imputation_strategy='mean')

# Median (robust to outliers)
download_and_prepare_data(imputation_strategy='median')
```

### Control Synthetic Data Proportion ⭐

**IMPORTANT:** You can now control how much synthetic data is generated!

```python
# RECOMMENDED: Moderate balance (10:1 ratio, ~6% synthetic)
download_and_prepare_data(
    balancing_strategy='smote',
    target_imbalance_ratio=10  # Only 6% synthetic data!
)

# Conservative balance (20:1 ratio, ~3% synthetic)
download_and_prepare_data(
    balancing_strategy='smote',
    target_imbalance_ratio=20
)

# Aggressive balance (5:1 ratio, ~15% synthetic)
download_and_prepare_data(
    balancing_strategy='smote',
    target_imbalance_ratio=5
)

# Full balance (NOT RECOMMENDED - 60% synthetic)
download_and_prepare_data(
    balancing_strategy='smote',
    target_imbalance_ratio=None  # 60% synthetic - use with caution!
)
```

### Available Balancing Strategies

```python
# SMOTE (default - reliable)
download_and_prepare_data(balancing_strategy='smote', target_imbalance_ratio=10)

# ADASYN (adaptive)
download_and_prepare_data(balancing_strategy='adasyn', target_imbalance_ratio=10)

# BorderlineSMOTE (focus on boundaries)
download_and_prepare_data(balancing_strategy='borderline', target_imbalance_ratio=10)

# Hybrid methods (SMOTE + cleaning) - not compatible with partial balancing
download_and_prepare_data(balancing_strategy='smote_tomek', target_imbalance_ratio=None)
```

### Combining Multiple Datasets

```python
from data import download_and_prepare_data

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

## Project Structure

```
diabetes-ft/
├── preprocessing_pipeline.py   # Advanced preprocessing pipeline
├── data.py                     # Data download and preparation
├── train.py                    # Standard training script
├── train_optimized.py          # Optimized for 4GB GPU
├── evaluate.py                 # Model evaluation
├── compare_preprocessing.py    # Compare preprocessing strategies
├── PREPROCESSING_GUIDE.md      # Comprehensive preprocessing guide
├── OPTIMIZATION_GUIDE.md       # Model optimization guide
├── CHANGES_SUMMARY.md          # Summary of changes
└── pyproject.toml              # Project dependencies
```

## Documentation

- **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)**: Complete preprocessing pipeline documentation
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)**: GPU optimization and class imbalance solutions
- **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)**: Changes between training scripts

## Compare Preprocessing Strategies

### Compare Synthetic Data Ratios (Recommended)

See the impact of different balancing ratios on synthetic data:

```bash
.venv/bin/python compare_synthetic_ratios.py
```

This shows:
- Synthetic data percentage for each ratio (1:1, 5:1, 10:1, 20:1, 30:1)
- Class distribution after balancing
- Recommendations for production vs research

### Compare All Strategies

Run comprehensive comparison of different strategies:

```bash
.venv/bin/python compare_preprocessing.py
```

This will:
- Compare all imputation strategies
- Compare all balancing strategies
- Demonstrate null value handling
- Generate detailed comparison report

## Validation & Reporting

Every preprocessing run generates:

1. **Processed datasets**:
   - `data/train.csv` - Balanced training set
   - `data/val.csv` - Validation set (original distribution)
   - `data/test.csv` - Test set (original distribution)

2. **Preprocessing report** (`data/preprocessing_report.json`):
   ```json
   {
     "initial_nulls": {...},
     "initial_distribution": {...},
     "train_distribution_before": {...},
     "train_distribution_after": {...},
     "validation": {
       "train_no_nulls": true,
       "val_no_nulls": true,
       "test_no_nulls": true,
       "class_balanced": true,
       "train_imbalance_ratio": 1.0
     }
   }
   ```

## Performance

Current configuration (253K samples, 21 features):
- **Processing time**: ~60 seconds
- **Memory usage**: ~2GB peak
- **Output size**: ~210MB total
- **Synthetic samples**: 271,200 created by SMOTE

## Requirements

- Python >= 3.12
- PyTorch + pytorch-tabular
- XGBoost >= 3.1.1
- scikit-learn >= 1.7.2
- imbalanced-learn >= 0.12.0
- pandas, numpy, kagglehub

## Best Practices

### Recommended Configuration

```python
# For production use (RECOMMENDED)
download_and_prepare_data(
    imputation_strategy='knn',       # Best balance of accuracy/speed
    balancing_strategy='smote',      # Proven and reliable
    apply_balancing=True,            # Handle class imbalance
    target_imbalance_ratio=10        # Only 6% synthetic data!
)

# For research/benchmarking
download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=True,
    target_imbalance_ratio=5         # 15% synthetic, better F1-score
)

# For conservative approach
download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=True,
    target_imbalance_ratio=20        # Only 3% synthetic data
)
```

### When to Use Different Strategies

**Imputation:**
- `knn`: Default, best for most cases
- `iterative`: Complex feature relationships
- `mean/median`: Fast, simple missing patterns

**Balancing:**
- `smote`: Default, good for most datasets
- `adasyn`: Varying density in feature space
- `borderline`: Focus on decision boundaries
- `smote_tomek/enn`: Clean noisy samples

## License

See project license file.

## References

- SMOTE: Chawla et al. (2002)
- ADASYN: He et al. (2008)
- FT-Transformer: Gorishniy et al. (2021)
- KNN Imputation: Troyanskaya et al. (2001)
