"""
ElasticNet Hyperparameter Optimization for Wine Quality Prediction
This script performs grid search to find the best alpha and l1_ratio values
for minimum RMSE, with MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings
import os

# Azure ML imports
try:
    from azureml.core import Workspace
    from azure_config import AZURE_ML_CONFIG
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    print("Azure ML SDK not available. Install with: pip install azureml-sdk")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# MLflow experiment configuration
EXPERIMENT_NAME = "wine-quality-elasticnet-optimization"

# Azure ML configuration
USE_AZURE_ML = True  # Set to False to use local tracking

def setup_mlflow_tracking():
    """
    Setup MLflow tracking - either Azure ML workspace or local file tracking
    Returns: Workspace object if Azure ML is configured, None otherwise
    """
    if USE_AZURE_ML and AZURE_ML_AVAILABLE:
        print("\nConfiguring Azure ML workspace for MLflow tracking...")
        try:
            # Try loading from config.json first
            try:
                ws = Workspace.from_config()
                print(f"✓ Loaded workspace from config.json: {ws.name}")
            except:
                # If config.json doesn't exist, try loading from azure_config.py
                ws = Workspace(
                    subscription_id=AZURE_ML_CONFIG['subscription_id'],
                    resource_group=AZURE_ML_CONFIG['resource_group'],
                    workspace_name=AZURE_ML_CONFIG['workspace_name']
                )
                print(f"✓ Connected to workspace: {ws.name}")
            
            # Get MLflow tracking URI from Azure ML workspace
            tracking_uri = ws.get_mlflow_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            
            print(f"✓ Subscription: {ws.subscription_id}")
            print(f"✓ Resource Group: {ws.resource_group}")
            print(f"✓ MLflow Tracking URI: {tracking_uri}")
            print(f"✓ Experiments will be tracked in Azure ML workspace")
            
            return ws
            
        except Exception as e:
            print(f"⚠ Failed to connect to Azure ML workspace: {e}")
            print("Falling back to local file tracking...")
            mlflow.set_tracking_uri("file:./mlruns")
            print(f"✓ Using local tracking: file:./mlruns")
            return None
    else:
        # Use local file tracking
        mlflow.set_tracking_uri("file:./mlruns")
        print(f"\nUsing local MLflow tracking: file:./mlruns")
        if not AZURE_ML_AVAILABLE:
            print("Tip: Install azureml-sdk for Azure ML integration: pip install azureml-sdk")
        return None

def load_and_prepare_data(filepath):
    """Load and prepare wine quality dataset"""
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        print(f"Dataset shape: {data.shape}")
        
        # Remove unnamed index column if it exists
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
            
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None

def optimize_elasticnet(X_train, y_train, X_test, y_test):
    """Perform grid search to find optimal ElasticNet parameters with MLflow tracking"""
    
    # Define parameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
        'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    
    # Create ElasticNet model
    elasticnet = ElasticNet(random_state=42, max_iter=1000)
    
    # Perform grid search with cross-validation
    print("Performing grid search for optimal parameters...")
    grid_search = GridSearchCV(
        estimator=elasticnet,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # Optimize for RMSE
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Start MLflow run for grid search
    with mlflow.start_run(run_name="grid_search_optimization", nested=True):
        # Log grid search parameters
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("alpha_values", param_grid['alpha'])
        mlflow.log_param("l1_ratio_values", param_grid['l1_ratio'])
        mlflow.log_param("total_combinations", len(param_grid['alpha']) * len(param_grid['l1_ratio']))
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)  # Convert to RMSE
        
        # Log best parameters and CV score
        mlflow.log_param("best_alpha", best_params['alpha'])
        mlflow.log_param("best_l1_ratio", best_params['l1_ratio'])
        mlflow.log_metric("cv_rmse", best_score)
        
        print(f"\nBest parameters found:")
        print(f"  Alpha: {best_params['alpha']}")
        print(f"  L1 Ratio: {best_params['l1_ratio']}")
        print(f"  Cross-validation RMSE: {best_score:.4f}")
    
    # Train final model with best parameters
    best_model = ElasticNet(
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        random_state=42,
        max_iter=1000
    )
    
    best_model.fit(X_train, y_train)
    
    return best_model, best_params, best_score

def evaluate_model(model, X_train, X_test, y_test, params):
    """Evaluate model performance and log to MLflow"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics to MLflow
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    
    # Feature importance analysis
    non_zero_coefs = np.sum(model.coef_ != 0)
    total_coefs = len(model.coef_)
    
    mlflow.log_metric("selected_features", non_zero_coefs)
    mlflow.log_metric("total_features", total_coefs)
    mlflow.log_metric("intercept", model.intercept_)
    
    # Log model signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log the model
    mlflow.sklearn.log_model(
        model, 
        "model",
        signature=signature
    )
    
    print("\n" + "="*60)
    print("OPTIMIZED MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Best ElasticNet Parameters:")
    print(f"  Alpha (regularization strength): {params['alpha']}")
    print(f"  L1 Ratio (Lasso vs Ridge balance): {params['l1_ratio']}")
    print(f"\nTest Set Performance Metrics:")
    print(f"  RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"\nFeature Selection:")
    print(f"  Selected features: {non_zero_coefs}/{total_coefs}")
    print(f"  Model intercept: {model.intercept_:.4f}")
    print("="*60)
    
    return rmse, mae, r2

def compare_with_baseline(baseline_rmse, optimized_rmse):
    """Compare optimized model with baseline and log to MLflow"""
    improvement = baseline_rmse - optimized_rmse
    improvement_pct = (improvement / baseline_rmse) * 100
    
    # Log comparison metrics
    mlflow.log_metric("baseline_rmse", baseline_rmse)
    mlflow.log_metric("improvement_rmse", improvement)
    mlflow.log_metric("improvement_percentage", improvement_pct)
    
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  Baseline RMSE (α=0.5, l1=0.5): {baseline_rmse:.4f}")
    print(f"  Optimized RMSE: {optimized_rmse:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

def main():
    """Main execution function with MLflow experiment tracking"""
    np.random.seed(42)
    
    # Configuration
    DATA_PATH = "../data/red-wine-quality.csv"
    
    print("ElasticNet Hyperparameter Optimization for Wine Quality Prediction")
    print("="*70)
    
    # Setup MLflow tracking (Azure ML or local)
    workspace = setup_mlflow_tracking()
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Experiment: {EXPERIMENT_NAME}")
    print("="*70)
    
    # Load and prepare data
    data = load_and_prepare_data(DATA_PATH)
    if data is None:
        return
    
    # Prepare features and target
    X = data.drop(['quality'], axis=1)
    y = data['quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Scale features for better optimization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {data.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Start main MLflow run
    with mlflow.start_run(run_name="elasticnet_optimization_main"):
        # Log dataset information
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("total_samples", data.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_scaling", "StandardScaler")
        
        # Log feature names
        mlflow.log_param("feature_names", list(X.columns))
        
        # Train baseline model for comparison
        with mlflow.start_run(run_name="baseline_model", nested=True):
            mlflow.log_param("model_type", "baseline")
            mlflow.log_param("alpha", 0.5)
            mlflow.log_param("l1_ratio", 0.5)
            
            baseline_model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
            baseline_model.fit(X_train_scaled, y_train)
            baseline_pred = baseline_model.predict(X_test_scaled)
            baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
            baseline_mae = mean_absolute_error(y_test, baseline_pred)
            baseline_r2 = r2_score(y_test, baseline_pred)
            
            mlflow.log_metric("baseline_rmse", baseline_rmse)
            mlflow.log_metric("baseline_mae", baseline_mae)
            mlflow.log_metric("baseline_r2", baseline_r2)
            
            # Log baseline model
            signature = infer_signature(X_train_scaled, baseline_model.predict(X_train_scaled))
            mlflow.sklearn.log_model(baseline_model, "baseline_model", signature=signature)
        
        # Optimize hyperparameters
        best_model, best_params, cv_rmse = optimize_elasticnet(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Evaluate optimized model
        test_rmse, test_mae, test_r2 = evaluate_model(
            best_model, X_train_scaled, X_test_scaled, y_test, best_params
        )
        
        # Compare with baseline
        compare_with_baseline(baseline_rmse, test_rmse)
        
        # Add tags to the run
        mlflow.set_tag("model_type", "optimized")
        mlflow.set_tag("optimization_method", "GridSearchCV")
        mlflow.set_tag("author", "MLflow ElasticNet Optimizer")
        mlflow.set_tag("dataset", "Red Wine Quality")
        
        print(f"\n✓ Experiment logged successfully to MLflow")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
        if workspace:  # Azure ML workspace is available
            print(f"✓ View results in Azure ML Studio: {workspace.get_portal_url()}")
        else:
            print(f"✓ View results: python -m mlflow ui --backend-store-uri file:./mlruns")
    
    return best_model, best_params

if __name__ == "__main__":
    optimized_model, best_parameters = main()