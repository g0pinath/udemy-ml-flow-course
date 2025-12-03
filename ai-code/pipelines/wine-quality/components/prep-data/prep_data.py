"""
Data Preparation Component for Wine Quality Pipeline

This component loads, cleans, and splits wine quality data.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

mlflow.sklearn.autolog()


def parse_args():
    parser = argparse.ArgumentParser("prep_data")
    parser.add_argument("--input_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to output training data")
    parser.add_argument("--test_data", type=str, help="Path to output test data")
    parser.add_argument("--test_split_ratio", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("DATA PREPARATION COMPONENT")
    print("="*70)
    
    # Load data
    print(f"\n[INFO] Loading data from: {args.input_data}")
    data = pd.read_csv(args.input_data)
    print(f"[OK] Dataset shape: {data.shape}")
    
    # Remove unnamed index column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
        print("[OK] Removed unnamed index column")
    
    # Separate features and target
    X = data.drop(['quality'], axis=1)
    y = data['quality']
    
    print(f"[OK] Features: {list(X.columns)}")
    print(f"[OK] Target: quality (values: {sorted(y.unique())})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_split_ratio, 
        random_state=args.random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Combine features and target
    train_data = X_train_scaled.copy()
    train_data['quality'] = y_train
    
    test_data = X_test_scaled.copy()
    test_data['quality'] = y_test
    
    # Save outputs
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    
    train_path = Path(args.train_data) / "train.csv"
    test_path = Path(args.test_data) / "test.csv"
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"\n[OK] Training data saved: {train_path} ({len(train_data)} samples)")
    print(f"[OK] Test data saved: {test_path} ({len(test_data)} samples)")
    print(f"[OK] Features scaled using StandardScaler")
    print("="*70)


if __name__ == "__main__":
    main()
