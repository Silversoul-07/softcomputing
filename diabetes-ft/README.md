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
✅ **Balanced classes** in training set (1:1:1 ratio)
✅ **Original distribution** preserved in validation/test sets
✅ **Comprehensive validation** with detailed reports

### Current Dataset Results

**Before Preprocessing:**
- Total samples: 253,680
- Class imbalance: 46.15:1 (highly imbalanced)
- Minority class (Prediabetes): 1.83%

**After Preprocessing:**
- Training samples: 448,776 (balanced with SMOTE)
- Class imbalance: 1.00:1 (perfectly balanced)
- All classes: 33.33% each
- Validation/Test: Original distribution maintained

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

### Available Balancing Strategies

```python
# SMOTE (default - reliable)
download_and_prepare_data(balancing_strategy='smote')

# ADASYN (adaptive)
download_and_prepare_data(balancing_strategy='adasyn')

# BorderlineSMOTE (focus on boundaries)
download_and_prepare_data(balancing_strategy='borderline')

# Hybrid methods (SMOTE + cleaning)
download_and_prepare_data(balancing_strategy='smote_tomek')
download_and_prepare_data(balancing_strategy='smote_enn')
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
# For production use
download_and_prepare_data(
    imputation_strategy='knn',      # Best balance of accuracy/speed
    balancing_strategy='smote',     # Proven and reliable
    apply_balancing=True            # Handle class imbalance
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
