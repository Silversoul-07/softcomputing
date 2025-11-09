import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score
)
from pytorch_tabular import TabularModel
from omegaconf.dictconfig import DictConfig

# Add safe globals for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([DictConfig])


def evaluate_ft_transformer(test_df):
    """Evaluate FT-Transformer on test set."""

    print("="*60)
    print("Evaluating FT-Transformer")
    print("="*60)

    # Load model
    ft_model = TabularModel.load_model('models/ft_transformer')

    # Predict
    pred_df = ft_model.predict(test_df)

    # Get predictions
    y_true = test_df['Diabetes_012'].values
    y_pred = pred_df['Diabetes_012_prediction'].values

    # Get probabilities for ROC-AUC
    prob_cols = [col for col in pred_df.columns if col.endswith('_probability')]
    y_pred_proba = pred_df[prob_cols].values

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = None

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    if roc_auc:
        print(f"ROC-AUC (OVR): {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc
    }


def evaluate_xgboost(test_df):
    """Evaluate XGBoost on test set."""

    print("\n" + "="*60)
    print("Evaluating XGBoost")
    print("="*60)

    # Load model
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # Prepare test data
    X_test = test_df.drop('Diabetes_012', axis=1)
    y_true = test_df['Diabetes_012'].values

    # Predict
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = None

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    if roc_auc:
        print(f"ROC-AUC (OVR): {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc
    }


def compare_models(ft_metrics, xgb_metrics):
    """Compare FT-Transformer vs XGBoost."""

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)

    print(f"\n{'Metric':<20} {'FT-Transformer':<20} {'XGBoost':<20} {'Winner':<20}")
    print("-"*80)

    for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']:
        ft_val = ft_metrics.get(metric)
        xgb_val = xgb_metrics.get(metric)

        if ft_val is None or xgb_val is None:
            continue

        winner = 'FT-Transformer' if ft_val > xgb_val else 'XGBoost' if xgb_val > ft_val else 'Tie'
        print(f"{metric:<20} {ft_val:<20.4f} {xgb_val:<20.4f} {winner:<20}")


if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv('data/test.csv')
    print(f"Test set size: {len(test_df)}")

    # Evaluate FT-Transformer
    ft_metrics = evaluate_ft_transformer(test_df)

    # Evaluate XGBoost
    xgb_metrics = evaluate_xgboost(test_df)

    # Compare models
    compare_models(ft_metrics, xgb_metrics)
