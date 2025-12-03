"""
Training Component for Wine Quality Pipeline

This component trains an ElasticNet model with specified hyperparameters.
Designed to be used in Azure ML Sweep jobs for hyperparameter optimization.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def parse_args():
    parser = argparse.ArgumentParser("train_model")
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path to save trained model")
    
    # Hyperparameters to tune
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="L1 ratio (0=Ridge, 1=Lasso)")
    parser.add_argument("--max_iter", type=int, default=10000, help="Maximum iterations")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    
    return parser.parse_args()


def load_data(data_path):
    """Load data from CSV file"""
    csv_files = list(Path(data_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    
    data = pd.read_csv(csv_files[0])
    return data


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2, y_pred


def main():
    args = parse_args()
    
    print("="*70)
    print("MODEL TRAINING COMPONENT")
    print("="*70)
    
    # Start MLflow run (Azure ML will create parent run automatically)
    mlflow.sklearn.autolog(log_models=False)  # We'll log manually for better control
    
    # Load training data
    print(f"\n[INFO] Loading training data from: {args.train_data}")
    train_data = load_data(args.train_data)
    print(f"[OK] Training data shape: {train_data.shape}")
    
    # Load test data
    print(f"[INFO] Loading test data from: {args.test_data}")
    test_data = load_data(args.test_data)
    print(f"[OK] Test data shape: {test_data.shape}")
    
    # Separate features and target
    X_train = train_data.drop(['quality'], axis=1)
    y_train = train_data['quality']
    X_test = test_data.drop(['quality'], axis=1)
    y_test = test_data['quality']
    
    # Log hyperparameters
    print(f"\n[INFO] Hyperparameters:")
    print(f"  - alpha: {args.alpha}")
    print(f"  - l1_ratio: {args.l1_ratio}")
    print(f"  - max_iter: {args.max_iter}")
    
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("l1_ratio", args.l1_ratio)
    mlflow.log_param("max_iter", args.max_iter)
    
    # Train model
    print(f"\n[INFO] Training ElasticNet model...")
    model = ElasticNet(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    print("[OK] Model trained successfully")
    
    # Evaluate on test set
    print(f"\n[INFO] Evaluating model on test set...")
    rmse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    # Log metrics
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    
    print(f"[RESULTS] Test RMSE: {rmse:.4f}")
    print(f"[RESULTS] Test MAE:  {mae:.4f}")
    print(f"[RESULTS] Test R²:   {r2:.4f}")
    
    # Save model
    print(f"\n[INFO] Saving model to: {args.model_output}")
    os.makedirs(args.model_output, exist_ok=True)
    
    # Create model signature for deployment
    signature = infer_signature(X_train, y_pred)
    
    # Save model with MLflow
    mlflow.sklearn.save_model(
        sk_model=model,
        path=args.model_output,
        signature=signature
    )
    
    print("[OK] Model saved successfully")
    print("="*70)


if __name__ == "__main__":
    main()
