"""
MLflow Experiment Series - Finding Optimal RMSE for Wine Quality Prediction

This script runs multiple experiments with different strategies:
1. Coarse grid search - Wide parameter range
2. Fine grid search - Narrow range around best results
3. Random search - Random parameter combinations
4. Custom experiments - Specific parameter combinations based on domain knowledge
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from scipy.stats import uniform, loguniform
import warnings
import time
import os

warnings.filterwarnings("ignore")

# MLflow configuration - Azure ML
EXPERIMENT_NAME = "wine-quality-rmse-optimization-series"
USE_AZURE_ML = True  # Set to False to use local tracking

# Azure ML Workspace configuration
try:
    from azure_config import AZURE_ML_CONFIG
    from azureml.core import Workspace
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure ML SDK not installed. Install with: pip install azureml-sdk")

def setup_mlflow_tracking():
    """Configure MLflow tracking for Azure ML or local"""
    if USE_AZURE_ML and AZURE_AVAILABLE:
        try:
            # Connect to Azure ML Workspace
            ws = Workspace(
                subscription_id=AZURE_ML_CONFIG["subscription_id"],
                resource_group=AZURE_ML_CONFIG["resource_group"],
                workspace_name=AZURE_ML_CONFIG["workspace_name"]
            )
            
            # Set MLflow tracking URI to Azure ML
            tracking_uri = ws.get_mlflow_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            
            print(f"✓ Connected to Azure ML Workspace: {ws.name}")
            print(f"✓ MLflow Tracking URI: {tracking_uri}")
            print(f"✓ Region: {ws.location}")
            return ws
        except Exception as e:
            print(f"✗ Failed to connect to Azure ML: {e}")
            print("Falling back to local tracking...")
            mlflow.set_tracking_uri("file:./mlruns")
            return None
    else:
        # Use local file-based tracking
        mlflow.set_tracking_uri("file:./mlruns")
        print("✓ Using local MLflow tracking: ./mlruns")
        return None


# Try to import Azure ML
try:
    from azureml.core import Workspace, Experiment
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    print("Azure ML SDK not installed. Using local tracking.")
    USE_AZURE_ML = False

def load_and_prepare_data(filepath):
    """Load and prepare wine quality dataset"""
    data = pd.read_csv(filepath)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics"""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2, y_pred

def run_experiment_1_coarse_grid(X_train, y_train, X_test, y_test, feature_names):
    """Experiment 1: Coarse Grid Search - Wide parameter exploration"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Coarse Grid Search")
    print("="*70)
    
    with mlflow.start_run(run_name="exp1_coarse_grid_search", nested=True):
        mlflow.set_tag("experiment_type", "coarse_grid_search")
        mlflow.set_tag("strategy", "wide_parameter_exploration")
        
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        total_combinations = len(param_grid['alpha']) * len(param_grid['l1_ratio'])
        mlflow.log_param("total_combinations", total_combinations)
        
        grid_search = GridSearchCV(
            ElasticNet(random_state=42, max_iter=10000),
            param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        rmse, mae, r2, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # Log parameters
        mlflow.log_param("best_alpha", grid_search.best_params_['alpha'])
        mlflow.log_param("best_l1_ratio", grid_search.best_params_['l1_ratio'])
        mlflow.log_param("cv_folds", 5)
        
        # Log metrics
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_best_score", -grid_search.best_score_)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.2f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        return grid_search.best_params_, rmse

def run_experiment_2_fine_grid(X_train, y_train, X_test, y_test, feature_names, coarse_params):
    """Experiment 2: Fine Grid Search - Narrow range around coarse grid best results"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Fine Grid Search (Around Coarse Grid Best)")
    print("="*70)
    
    with mlflow.start_run(run_name="exp2_fine_grid_search", nested=True):
        mlflow.set_tag("experiment_type", "fine_grid_search")
        mlflow.set_tag("strategy", "refinement_around_coarse_best")
        mlflow.set_tag("based_on_experiment", "exp1_coarse_grid_search")
        
        # Create fine grid around best coarse parameters
        best_alpha = coarse_params['alpha']
        best_l1 = coarse_params['l1_ratio']
        
        # Fine-tune around the best alpha (±50% range with smaller steps)
        alpha_range = np.linspace(best_alpha * 0.5, best_alpha * 1.5, 10)
        # Fine-tune around the best l1_ratio (±0.2 range)
        l1_range = np.linspace(max(0.0, best_l1 - 0.2), min(1.0, best_l1 + 0.2), 9)
        
        param_grid = {
            'alpha': alpha_range.tolist(),
            'l1_ratio': l1_range.tolist()
        }
        
        mlflow.log_param("alpha_range_center", best_alpha)
        mlflow.log_param("l1_ratio_range_center", best_l1)
        mlflow.log_param("total_combinations", len(alpha_range) * len(l1_range))
        
        grid_search = GridSearchCV(
            ElasticNet(random_state=42, max_iter=10000),
            param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        rmse, mae, r2, y_pred = evaluate_model(best_model, X_test, y_test)
        
        mlflow.log_param("best_alpha", grid_search.best_params_['alpha'])
        mlflow.log_param("best_l1_ratio", grid_search.best_params_['l1_ratio'])
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_best_score", -grid_search.best_score_)
        mlflow.log_metric("training_time_seconds", training_time)
        
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        return grid_search.best_params_, rmse

def run_experiment_3_random_search(X_train, y_train, X_test, y_test, feature_names):
    """Experiment 3: Random Search - Random parameter combinations"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Random Search")
    print("="*70)
    
    with mlflow.start_run(run_name="exp3_random_search", nested=True):
        mlflow.set_tag("experiment_type", "random_search")
        mlflow.set_tag("strategy", "random_parameter_sampling")
        
        param_distributions = {
            'alpha': loguniform(0.0001, 10),
            'l1_ratio': uniform(0.0, 1.0)
        }
        
        n_iter = 100
        mlflow.log_param("n_iterations", n_iter)
        
        random_search = RandomizedSearchCV(
            ElasticNet(random_state=42, max_iter=10000),
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = random_search.best_estimator_
        rmse, mae, r2, y_pred = evaluate_model(best_model, X_test, y_test)
        
        mlflow.log_param("best_alpha", random_search.best_params_['alpha'])
        mlflow.log_param("best_l1_ratio", random_search.best_params_['l1_ratio'])
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_best_score", -random_search.best_score_)
        mlflow.log_metric("training_time_seconds", training_time)
        
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={random_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={random_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        return random_search.best_params_, rmse

def run_experiment_4_domain_specific(X_train, y_train, X_test, y_test, feature_names):
    """Experiment 4: Domain-specific configurations"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Domain-Specific Configurations")
    print("="*70)
    
    configurations = [
        {"name": "lasso_heavy", "alpha": 0.01, "l1_ratio": 0.95, "description": "Heavy Lasso bias for feature selection"},
        {"name": "ridge_heavy", "alpha": 0.01, "l1_ratio": 0.05, "description": "Heavy Ridge bias for coefficient shrinkage"},
        {"name": "balanced", "alpha": 0.01, "l1_ratio": 0.5, "description": "Balanced L1/L2 regularization"},
        {"name": "minimal_regularization", "alpha": 0.001, "l1_ratio": 0.5, "description": "Minimal regularization"},
        {"name": "strong_regularization", "alpha": 1.0, "l1_ratio": 0.5, "description": "Strong regularization"}
    ]
    
    best_config = None
    best_rmse = float('inf')
    
    for config in configurations:
        with mlflow.start_run(run_name=f"exp4_{config['name']}", nested=True):
            mlflow.set_tag("experiment_type", "domain_specific")
            mlflow.set_tag("configuration_name", config['name'])
            mlflow.set_tag("description", config['description'])
            
            model = ElasticNet(
                alpha=config['alpha'],
                l1_ratio=config['l1_ratio'],
                random_state=42,
                max_iter=10000
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            rmse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)
            
            mlflow.log_param("alpha", config['alpha'])
            mlflow.log_param("l1_ratio", config['l1_ratio'])
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("n_features_selected", np.sum(model.coef_ != 0))
            
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(model, "model", signature=signature)
            
            print(f"\n{config['name']}: RMSE={rmse:.4f}, Features={np.sum(model.coef_ != 0)}/11")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config
    
    print(f"\nBest configuration: {best_config['name']} with RMSE={best_rmse:.4f}")
    return best_config, best_rmse

def run_experiment_5_ultra_fine(X_train, y_train, X_test, y_test, feature_names, fine_params):
    """Experiment 5: Ultra-fine grid search around the best fine-tuned parameters"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Ultra-Fine Grid Search")
    print("="*70)
    
    with mlflow.start_run(run_name="exp5_ultra_fine_search", nested=True):
        mlflow.set_tag("experiment_type", "ultra_fine_search")
        mlflow.set_tag("strategy", "precision_optimization")
        mlflow.set_tag("based_on_experiment", "exp2_fine_grid_search")
        
        best_alpha = fine_params['alpha']
        best_l1 = fine_params['l1_ratio']
        
        # Ultra-fine grid (±10% range with very small steps)
        alpha_range = np.linspace(best_alpha * 0.9, best_alpha * 1.1, 15)
        l1_range = np.linspace(max(0.0, best_l1 - 0.1), min(1.0, best_l1 + 0.1), 15)
        
        param_grid = {
            'alpha': alpha_range.tolist(),
            'l1_ratio': l1_range.tolist()
        }
        
        mlflow.log_param("alpha_range_center", best_alpha)
        mlflow.log_param("l1_ratio_range_center", best_l1)
        mlflow.log_param("total_combinations", len(alpha_range) * len(l1_range))
        
        grid_search = GridSearchCV(
            ElasticNet(random_state=42, max_iter=10000),
            param_grid,
            cv=10,  # More folds for better validation
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        rmse, mae, r2, y_pred = evaluate_model(best_model, X_test, y_test)
        
        mlflow.log_param("best_alpha", grid_search.best_params_['alpha'])
        mlflow.log_param("best_l1_ratio", grid_search.best_params_['l1_ratio'])
        mlflow.log_param("cv_folds", 10)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_best_score", -grid_search.best_score_)
        mlflow.log_metric("training_time_seconds", training_time)
        
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.6f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.6f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        return grid_search.best_params_, rmse

def main():
    """Main execution function"""
    print("="*70)
    print("MLflow Experiment Series - Wine Quality RMSE Optimization")
    print("="*70)
    
    # Setup MLflow tracking (Azure ML or local)
    workspace = setup_mlflow_tracking()
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    filepath = "../data/red-wine-quality.csv"
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(filepath)
    
    print(f"\nDataset: {len(y_train)} training samples, {len(y_test)} test samples")
    print(f"Features: {len(feature_names)}")
    
    # Track best results
    results = {}
    
    # Run experiments sequentially
    with mlflow.start_run(run_name="experiment_series_master"):
        mlflow.set_tag("series_type", "systematic_rmse_optimization")
        mlflow.set_tag("total_experiments", 5)
        
        # Experiment 1: Coarse grid
        coarse_params, coarse_rmse = run_experiment_1_coarse_grid(
            X_train, y_train, X_test, y_test, feature_names
        )
        results['coarse_grid'] = coarse_rmse
        
        # Experiment 2: Fine grid (based on coarse results)
        fine_params, fine_rmse = run_experiment_2_fine_grid(
            X_train, y_train, X_test, y_test, feature_names, coarse_params
        )
        results['fine_grid'] = fine_rmse
        
        # Experiment 3: Random search
        random_params, random_rmse = run_experiment_3_random_search(
            X_train, y_train, X_test, y_test, feature_names
        )
        results['random_search'] = random_rmse
        
        # Experiment 4: Domain-specific
        domain_config, domain_rmse = run_experiment_4_domain_specific(
            X_train, y_train, X_test, y_test, feature_names
        )
        results['domain_specific'] = domain_rmse
        
        # Experiment 5: Ultra-fine (based on fine results)
        ultra_params, ultra_rmse = run_experiment_5_ultra_fine(
            X_train, y_train, X_test, y_test, feature_names, fine_params
        )
        results['ultra_fine'] = ultra_rmse
        
        # Log summary metrics
        mlflow.log_metric("best_rmse_overall", min(results.values()))
        mlflow.log_metric("rmse_coarse_grid", coarse_rmse)
        mlflow.log_metric("rmse_fine_grid", fine_rmse)
        mlflow.log_metric("rmse_random_search", random_rmse)
        mlflow.log_metric("rmse_domain_specific", domain_rmse)
        mlflow.log_metric("rmse_ultra_fine", ultra_rmse)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SERIES SUMMARY")
    print("="*70)
    for exp_name, rmse in results.items():
        print(f"{exp_name:.<30} RMSE: {rmse:.4f}")
    
    best_experiment = min(results, key=results.get)
    best_rmse = results[best_experiment]
    print(f"\n{'Best Experiment':.>30} {best_experiment} (RMSE: {best_rmse:.4f})")
    print("="*70)
    print(f"\n✓ All experiments completed successfully!")
    if workspace:  # Azure ML workspace is available
        print(f"✓ View results in Azure ML Studio: {workspace.get_portal_url()}")
    else:
        print(f"✓ View results: python -m mlflow ui --backend-store-uri file:./mlruns")
    print("="*70)

if __name__ == "__main__":
    main()
