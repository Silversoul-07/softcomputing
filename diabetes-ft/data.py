import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub


def download_and_prepare_data():
    """Download diabetes dataset from Kaggle and prepare train/val/test splits."""

    print("Downloading diabetes dataset from Kaggle...")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

    # Load the dataset
    csv_file = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
    print(f"Loading dataset from {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Diabetes_012'].value_counts()}")

    # Separate features and target
    X = df.drop('Diabetes_012', axis=1)
    y = df['Diabetes_012']

    # First split: 70% train, 30% temp (for val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Second split: split temp into 50% val, 50% test (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\nSplit sizes:")
    print(f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Val: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Reset indices
    X_train_scaled.reset_index(drop=True, inplace=True)
    X_val_scaled.reset_index(drop=True, inplace=True)
    X_test_scaled.reset_index(drop=True, inplace=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Save train/val/test sets
    train_df = X_train_scaled.copy()
    train_df['Diabetes_012'] = y_train
    train_df.to_csv('data/train.csv', index=False)

    val_df = X_val_scaled.copy()
    val_df['Diabetes_012'] = y_val
    val_df.to_csv('data/val.csv', index=False)

    test_df = X_test_scaled.copy()
    test_df['Diabetes_012'] = y_test
    test_df.to_csv('data/test.csv', index=False)

    print("\nData saved to data/ folder:")
    print("- data/train.csv")
    print("- data/val.csv")
    print("- data/test.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    download_and_prepare_data()
