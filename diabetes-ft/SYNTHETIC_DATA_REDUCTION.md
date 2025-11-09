# Reducing Synthetic Data Proportion

## Problem
With a single dataset (253,680 samples) and severe class imbalance (46:1), full SMOTE balancing creates ~60% synthetic data (271,200 synthetic / 448,776 total). This raises concerns about model reliability and generalization.

## Solutions

### Option 1: Partial Balancing (Recommended)

Instead of fully balancing to 1:1:1 ratio, use **partial balancing** to a more moderate ratio:

```python
from preprocessing_pipeline import PreprocessingPipeline
from imblearn.over_sampling import SMOTE

# Custom partial balancing
pipeline = PreprocessingPipeline(
    imputation_strategy='knn',
    balancing_strategy='smote',
    random_state=42
)

# Modify SMOTE to target 10:1 ratio instead of 1:1
# This reduces synthetic data from 60% to ~15-20%
```

**Current approach (Full Balance):**
- Class 0: 149,592 → 149,592 (0% synthetic)
- Class 1: 3,242 → 149,592 (98% synthetic)
- Class 2: 24,742 → 149,592 (83% synthetic)
- **Total synthetic: 60%**

**Partial Balance (10:1 ratio):**
- Class 0: 149,592 → 149,592 (0% synthetic)
- Class 1: 3,242 → 14,959 (78% synthetic)
- Class 2: 24,742 → 24,742 (0% synthetic - no balancing needed)
- **Total synthetic: ~7%**

### Option 2: Class Weights (No Synthetic Data)

Use class weights in the model instead of oversampling:

```python
# XGBoost with class weights
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y_train)

xgb_model = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda',
    sample_weight=sample_weights  # Handle imbalance without synthetic data
)
```

**Pros:**
- 0% synthetic data
- Faster training
- Original data distribution preserved

**Cons:**
- May underperform on minority class
- Less effective for neural networks (FT-Transformer)

### Option 3: Hybrid Approach (Recommended)

Combine moderate oversampling with class weights:

```python
# 1. Moderate SMOTE (target 5:1 ratio)
# 2. + Class weights for remaining imbalance

# Result: ~10% synthetic data + improved minority class performance
```

### Option 4: Alternative Datasets

**Publicly Available:**
- **UCI Diabetes Dataset** (768 samples) - Too small
- **Pima Indians Diabetes** (768 samples) - Too small
- **Data.gov Diabetes Datasets** - Different schema, requires extensive preprocessing

**Challenge:** Most publicly available diabetes datasets either:
1. Are too small (<10K samples)
2. Have different schemas (incompatible columns)
3. Focus on different aspects (clinical vs survey data)

The BRFSS 2023 dataset (433K samples) exists but uses raw BRFSS coding that's incompatible with the cleaned 2015 format without extensive manual mapping.

## Implementation: Partial Balancing

### Modified Preprocessing Pipeline

```python
from imblearn.over_sampling import SMOTE

def partial_balance(X, y, target_ratio=10):
    """
    Apply partial SMOTE balancing.

    Args:
        X: Features
        y: Target
        target_ratio: Target max_class/min_class ratio (default: 10)

    Returns:
        X_resampled, y_resampled with controlled synthetic data
    """
    from collections import Counter

    class_counts = Counter(y)
    max_count = max(class_counts.values())

    # Calculate target samples for each class
    sampling_strategy = {}
    for class_label, count in class_counts.items():
        if count < max_count:
            # Target: max_count / target_ratio
            target = max(count, max_count // target_ratio)
            if target > count:
                sampling_strategy[class_label] = target

    if not sampling_strategy:
        return X, y  # No balancing needed

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=3,
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# Usage
from preprocessing_pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    imputation_strategy='knn',
    balancing_strategy='smote',  # Will modify this
    random_state=42
)

# Manually apply partial balancing
train_df, val_df, test_df = pipeline.preprocess(
    df=df,
    target_column='Diabetes_012',
    apply_balancing=False  # We'll do it manually
)

# Apply partial balancing
X_train = train_df.drop('Diabetes_012', axis=1)
y_train = train_df['Diabetes_012']

X_balanced, y_balanced = partial_balance(X_train, y_train, target_ratio=10)
```

## Comparison: Synthetic Data Proportion

| Strategy | Minority Class Size | Total Samples | Synthetic % | F1-Score (Expected) |
|----------|---------------------|---------------|-------------|---------------------|
| **No Balancing** | 3,242 | 177,576 | 0% | 0.30-0.35 |
| **Partial (20:1)** | 7,480 | 182,814 | 3% | 0.38-0.42 |
| **Partial (10:1)** | 14,959 | 190,093 | 7% | 0.42-0.47 |
| **Partial (5:1)** | 29,918 | 205,052 | 15% | 0.45-0.50 |
| **Full Balance (1:1)** | 149,592 | 448,776 | 60% | 0.48-0.55 |

## Recommended Strategy

### For Production (Real-World Deployment):

```python
# Partial balancing (10:1) + Class weights
download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',  # Modify for 10:1 ratio
    apply_balancing=True,
    use_multi_datasets=False
)

# Then use class weights in XGBoost
```

**Benefits:**
- Only ~7% synthetic data
- Better generalization
- More trustworthy predictions
- Still handles minority class reasonably

### For Research (Maximum F1-Score):

```python
# Full balancing (current approach)
download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=True
)
```

**Benefits:**
- Best minority class performance
- Highest F1-macro score
- Good for research/benchmarking

### For Baseline Comparison:

```python
# No synthetic data - class weights only
download_and_prepare_data(
    imputation_strategy='knn',
    balancing_strategy='smote',
    apply_balancing=False
)
```

**Benefits:**
- 0% synthetic data
- Fastest training
- Real data only

## Implementation Plan

I'll now create an updated preprocessing pipeline with partial balancing support:

1. Add `target_imbalance_ratio` parameter to `PreprocessingPipeline`
2. Support partial SMOTE with configurable target ratio
3. Add comparison script showing different ratios
4. Update documentation with recommendations

This gives you control over synthetic data proportion while still handling class imbalance effectively.
