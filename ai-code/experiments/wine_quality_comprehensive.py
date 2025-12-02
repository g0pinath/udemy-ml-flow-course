"""
Comprehensive Wine Quality Prediction - MLflow Experiment Suite

This unified script combines all optimization approaches:
1. Baseline model evaluation
2. Standard grid search optimization
3. Five different optimization strategies:
   - Coarse grid search (wide parameter range)
   - Fine grid search (narrow range around best results)
   - Random search (random parameter combinations)
   - Domain-specific (wine quality domain knowledge)
   - Ultra-fine grid search (precise tuning)

Supports both local and Azure ML workspace tracking.
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

# Azure ML imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

# Try to import azure_config (available locally, not in pipeline)
try:
    from azure_config import AZURE_ML_CONFIG
except ImportError:
    AZURE_ML_CONFIG = None  # Will use environment variables instead

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
EXPERIMENT_NAME = "wine-quality-comprehensive-optimization"
USE_AZURE_ML = True  # Always use Azure ML workspace tracking
LOG_MODELS = False  # Set to False to skip model logging (Azure ML logged-models API not supported)

# ============================================================================
# AZURE ML WORKSPACE SETUP
# ============================================================================

def setup_mlflow_tracking():
    """
    Setup MLflow tracking to Azure ML workspace (required)
    Uses environment variables for authentication:
    - AZURE_TENANT_ID
    - AZURE_CLIENT_ID
    - AZURE_CLIENT_SECRET
    Returns: Workspace object
    """
    print("\nConfiguring Azure ML workspace for MLflow tracking...")
    
    # Check for service principal environment variables (try TF_VAR_* first, fallback to AZURE_*)
    tenant_id = os.getenv('TF_VAR_AZURE_TENANT_ID') or os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('TF_VAR_AZURE_CLIENT_ID') or os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('TF_VAR_AZURE_CLIENT_SECRET') or os.getenv('AZURE_CLIENT_SECRET')
    subscription_id = os.getenv('TF_VAR_AZURE_SUBSCRIPTION_ID') or os.getenv('AZURE_SUBSCRIPTION_ID')
    
    if not all([tenant_id, client_id, client_secret]):
        raise EnvironmentError(
            "Missing required environment variables. Run scripts/load-env.ps1 first.\n"
            "Required: TF_VAR_AZURE_TENANT_ID, TF_VAR_AZURE_CLIENT_ID, TF_VAR_AZURE_CLIENT_SECRET\n"
            "Or: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
        )
    
    print("[OK] Using Service Principal authentication from environment variables")
    auth = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=client_id,
        service_principal_password=client_secret
    )
    
    # Connect to workspace (prefer env vars, fall back to config file)
    # In pipeline: all values from environment variables
    # Locally: can use azure_config.py for workspace details
    resource_group = os.getenv('AZURE_RESOURCE_GROUP') or (AZURE_ML_CONFIG and AZURE_ML_CONFIG['resource_group'])
    workspace_name = os.getenv('AZURE_WORKSPACE_NAME') or (AZURE_ML_CONFIG and AZURE_ML_CONFIG['workspace_name'])
    
    if not subscription_id:
        subscription_id = AZURE_ML_CONFIG and AZURE_ML_CONFIG['subscription_id']
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise EnvironmentError(
            "Missing workspace configuration. Provide via environment variables:\n"
            "  TF_VAR_AZURE_SUBSCRIPTION_ID (or AZURE_SUBSCRIPTION_ID)\n"
            "  AZURE_RESOURCE_GROUP\n"
            "  AZURE_WORKSPACE_NAME\n"
            "Or configure azure_config.py for local development."
        )
    
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        auth=auth
    )
    print(f"[OK] Connected to workspace: {ws.name}")
    
    # Get MLflow tracking URI from Azure ML workspace
    tracking_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"[OK] Subscription: {ws.subscription_id}")
    print(f"[OK] Resource Group: {ws.resource_group}")
    print(f"[OK] MLflow Tracking URI: {tracking_uri}")
    print(f"[OK] Experiments will be tracked in Azure ML workspace")
    
    # Construct portal URL
    workspace_id = f"/subscriptions/{ws.subscription_id}/resourcegroups/{ws.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{ws.name}"
    portal_url = f"https://ml.azure.com?tid={tenant_id}&wsid={workspace_id}"
    print(f"[OK] View results at: {portal_url}")
    
    return ws

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data(filepath):
    """Load and prepare wine quality dataset"""
    try:
        data = pd.read_csv(filepath)
        print(f"[OK] Data loaded from {filepath}")
        print(f"[OK] Dataset shape: {data.shape}")
        
        # Remove unnamed index column if it exists
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
            
        return data
    except FileNotFoundError:
        print(f"[ERROR] Error: File {filepath} not found")
        return None

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2, y_pred

# ============================================================================
# BASELINE MODEL
# ============================================================================

def run_baseline_model(X_train, y_train, X_test, y_test, feature_names):
    """Run baseline ElasticNet model with default parameters"""
    print("\n" + "="*70)
    print("BASELINE MODEL")
    print("="*70)
    
    with mlflow.start_run(run_name="baseline_model", nested=True):
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("description", "Default ElasticNet parameters")
        
        # Train baseline model
        baseline_model = ElasticNet(random_state=42)
        baseline_model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("alpha", baseline_model.alpha)
        mlflow.log_param("l1_ratio", baseline_model.l1_ratio)
        mlflow.log_param("max_iter", baseline_model.max_iter)
        
        # Evaluate
        rmse, mae, r2, y_pred = evaluate_model(baseline_model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        
        # Log model (if enabled)
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(baseline_model, "model", signature=signature)
        
        print(f"Baseline RMSE: {rmse:.4f}")
        print(f"Baseline MAE: {mae:.4f}")
        print(f"Baseline R²: {r2:.4f}")
        
        return baseline_model, rmse

# ============================================================================
# STANDARD GRID SEARCH
# ============================================================================

def run_standard_grid_search(X_train, y_train, X_test, y_test, feature_names):
    """Standard comprehensive grid search"""
    print("\n" + "="*70)
    print("STANDARD GRID SEARCH OPTIMIZATION")
    print("="*70)
    
    with mlflow.start_run(run_name="standard_grid_search", nested=True):
        mlflow.set_tag("optimization_type", "standard_grid_search")
        mlflow.set_tag("strategy", "comprehensive_parameter_sweep")
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
            'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        
        mlflow.log_param("alpha_values", param_grid['alpha'])
        mlflow.log_param("l1_ratio_values", param_grid['l1_ratio'])
        mlflow.log_param("total_combinations", len(param_grid['alpha']) * len(param_grid['l1_ratio']))
        mlflow.log_param("cv_folds", 5)
        
        # Grid search
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
        
        # Log results
        mlflow.log_param("best_alpha", grid_search.best_params_['alpha'])
        mlflow.log_param("best_l1_ratio", grid_search.best_params_['l1_ratio'])
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_best_score", -grid_search.best_score_)
        mlflow.log_metric("training_time_seconds", training_time)
        
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        return grid_search.best_params_, rmse

# ============================================================================
# EXPERIMENT 1: COARSE GRID SEARCH
# ============================================================================

def run_experiment_1_coarse_grid(X_train, y_train, X_test, y_test, feature_names):
    """Experiment 1: Coarse Grid Search - Wide parameter exploration"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Coarse Grid Search")
    print("="*70)
    
    with mlflow.start_run(run_name="exp1_coarse_grid", nested=True):
        mlflow.set_tag("experiment_type", "coarse_grid_search")
        mlflow.set_tag("strategy", "wide_parameter_exploration")
        
        param_grid = {
            'alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        
        mlflow.log_param("total_combinations", len(param_grid['alpha']) * len(param_grid['l1_ratio']))
        
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
        
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
        return grid_search.best_params_, rmse

# ============================================================================
# EXPERIMENT 2: FINE GRID SEARCH
# ============================================================================

def run_experiment_2_fine_grid(X_train, y_train, X_test, y_test, feature_names, coarse_params):
    """Experiment 2: Fine Grid Search - Narrow search around best coarse results"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Fine Grid Search")
    print("="*70)
    
    with mlflow.start_run(run_name="exp2_fine_grid", nested=True):
        mlflow.set_tag("experiment_type", "fine_grid_search")
        mlflow.set_tag("strategy", "refinement_around_coarse_best")
        
        # Create fine grid around best coarse parameters
        best_alpha = coarse_params['alpha']
        best_l1 = coarse_params['l1_ratio']
        
        alpha_range = np.linspace(best_alpha * 0.5, best_alpha * 1.5, 10)
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
        
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
        return grid_search.best_params_, rmse

# ============================================================================
# EXPERIMENT 3: RANDOM SEARCH
# ============================================================================

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
        
        mlflow.log_param("n_iterations", 100)
        
        random_search = RandomizedSearchCV(
            ElasticNet(random_state=42, max_iter=10000),
            param_distributions,
            n_iter=100,
            cv=5,
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=-1,
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
        
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={random_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={random_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
        return random_search.best_params_, rmse

# ============================================================================
# EXPERIMENT 4: DOMAIN-SPECIFIC
# ============================================================================

def run_experiment_4_domain_specific(X_train, y_train, X_test, y_test, feature_names):
    """Experiment 4: Domain-Specific - Based on wine quality domain knowledge"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Domain-Specific Optimization")
    print("="*70)
    
    with mlflow.start_run(run_name="exp4_domain_specific", nested=True):
        mlflow.set_tag("experiment_type", "domain_specific")
        mlflow.set_tag("strategy", "wine_quality_domain_knowledge")
        
        # Wine quality typically benefits from minimal regularization
        # Focus on small alpha values and balanced L1/L2 mix
        param_grid = {
            'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            'l1_ratio': [0.3, 0.4, 0.5, 0.6, 0.7]  # Balanced L1/L2
        }
        
        mlflow.log_param("total_combinations", len(param_grid['alpha']) * len(param_grid['l1_ratio']))
        mlflow.log_param("domain_insight", "minimal_regularization_balanced_mix")
        
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
        
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
        return grid_search.best_params_, rmse

# ============================================================================
# EXPERIMENT 5: ULTRA-FINE GRID
# ============================================================================

def run_experiment_5_ultra_fine(X_train, y_train, X_test, y_test, feature_names, best_params_so_far):
    """Experiment 5: Ultra-Fine Grid - Very precise tuning around best parameters"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Ultra-Fine Grid Search")
    print("="*70)
    
    with mlflow.start_run(run_name="exp5_ultra_fine", nested=True):
        mlflow.set_tag("experiment_type", "ultra_fine_grid")
        mlflow.set_tag("strategy", "precise_final_tuning")
        
        best_alpha = best_params_so_far['alpha']
        best_l1 = best_params_so_far['l1_ratio']
        
        # Very narrow range with high granularity
        alpha_range = np.linspace(best_alpha * 0.8, best_alpha * 1.2, 15)
        l1_range = np.linspace(max(0.0, best_l1 - 0.1), min(1.0, best_l1 + 0.1), 11)
        
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
        
        if LOG_MODELS:
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        print(f"Best params: alpha={grid_search.best_params_['alpha']:.4f}, "
              f"l1_ratio={grid_search.best_params_['l1_ratio']:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
        return grid_search.best_params_, rmse

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    np.random.seed(42)
    
    print("="*70)
    print("COMPREHENSIVE WINE QUALITY PREDICTION - MLFLOW EXPERIMENT SUITE")
    print("="*70)
    
    # Setup MLflow tracking
    workspace = setup_mlflow_tracking()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data from g0pinath/data folder
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    filepath = os.path.join(data_dir, 'red-wine-quality.csv')
    data = load_and_prepare_data(filepath)
    if data is None:
        return
    
    # Prepare features and target
    X = data.drop(['quality'], axis=1)
    y = data['quality']
    feature_names = X.columns.tolist()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n[OK] Dataset: {len(y_train)} training, {len(y_test)} test samples")
    print(f"[OK] Features: {len(feature_names)}")
    
    # Start main MLflow run
    with mlflow.start_run(run_name="comprehensive_optimization_suite"):
        mlflow.set_tag("suite_type", "comprehensive")
        mlflow.set_tag("dataset", "red_wine_quality")
        mlflow.set_tag("author", "MLflow Comprehensive Suite")
        
        # Track all results
        results = {}
        
        # 1. Baseline Model
        print("\n" + "="*70)
        print("PHASE 1: BASELINE EVALUATION")
        print("="*70)
        _, baseline_rmse = run_baseline_model(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        results['baseline'] = baseline_rmse
        
        # 2. Standard Grid Search
        print("\n" + "="*70)
        print("PHASE 2: STANDARD OPTIMIZATION")
        print("="*70)
        standard_params, standard_rmse = run_standard_grid_search(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        results['standard_grid'] = standard_rmse
        
        # 3. Advanced Strategy Series
        print("\n" + "="*70)
        print("PHASE 3: ADVANCED STRATEGY COMPARISON (5 Approaches)")
        print("="*70)
        
        # Experiment 1: Coarse Grid
        coarse_params, coarse_rmse = run_experiment_1_coarse_grid(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        results['coarse_grid'] = coarse_rmse
        
        # Experiment 2: Fine Grid (based on coarse results)
        fine_params, fine_rmse = run_experiment_2_fine_grid(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names, coarse_params
        )
        results['fine_grid'] = fine_rmse
        
        # Experiment 3: Random Search
        random_params, random_rmse = run_experiment_3_random_search(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        results['random_search'] = random_rmse
        
        # Experiment 4: Domain-Specific
        domain_params, domain_rmse = run_experiment_4_domain_specific(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        results['domain_specific'] = domain_rmse
        
        # Find best approach so far
        best_approach = min(results, key=results.get)
        best_params_so_far = {
            'coarse_grid': coarse_params,
            'fine_grid': fine_params,
            'random_search': random_params,
            'domain_specific': domain_params,
            'standard_grid': standard_params
        }.get(best_approach, domain_params)
        
        # Experiment 5: Ultra-Fine (based on best so far)
        ultra_params, ultra_rmse = run_experiment_5_ultra_fine(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names, best_params_so_far
        )
        results['ultra_fine'] = ultra_rmse
        
        # Final Summary
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print(f"\n{'Approach':<25} {'RMSE':>10} {'Improvement':>12}")
        print("-"*70)
        
        for approach, rmse in sorted(results.items(), key=lambda x: x[1]):
            improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
            marker = " ⭐" if approach == min(results, key=results.get) else ""
            print(f"{approach:<25} {rmse:>10.4f} {improvement:>11.2f}%{marker}")
        
        best_approach = min(results, key=results.get)
        best_rmse = results[best_approach]
        print("\n" + "="*70)
        print(f"🏆 BEST APPROACH: {best_approach.upper()}")
        print(f"🏆 BEST RMSE: {best_rmse:.4f}")
        print(f"🏆 IMPROVEMENT: {((baseline_rmse - best_rmse) / baseline_rmse * 100):.2f}% over baseline")
        print("="*70)
        
        # Log summary metrics to main run
        mlflow.log_metric("baseline_rmse", baseline_rmse)
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("improvement_percentage", (baseline_rmse - best_rmse) / baseline_rmse * 100)
        mlflow.log_param("best_approach", best_approach)
        
        print(f"\n[OK] All experiments completed successfully!")
        print(f"[OK] Run ID: {mlflow.active_run().info.run_id}")
        if workspace:
            tenant_id = os.getenv('AZURE_TENANT_ID')
            workspace_id = f"/subscriptions/{workspace.subscription_id}/resourcegroups/{workspace.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace.name}"
            portal_url = f"https://ml.azure.com?tid={tenant_id}&wsid={workspace_id}"
            print(f"[OK] View results in Azure ML Studio: {portal_url}")
        else:
            print(f"[OK] View results: python -m mlflow ui --backend-store-uri file:./mlruns")
        print("="*70)

if __name__ == "__main__":
    main()
