"""
Train and register the best model with the optimal parameters found
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

# Try to import azure_config (available locally, not in pipeline)
try:
    from azure_config import AZURE_ML_CONFIG
except ImportError:
    AZURE_ML_CONFIG = None  # Will use environment variables instead

# Best parameters from optimization
BEST_ALPHA = 0.01
BEST_L1_RATIO = 0.9

def train_and_register_best_model():
    """Train a model with the best parameters and register it"""
    print("="*70)
    print("TRAINING AND REGISTERING BEST MODEL")
    print("="*70)
    
    # Setup Azure ML (try TF_VAR_* first, fallback to AZURE_*)
    tenant_id = os.getenv('TF_VAR_AZURE_TENANT_ID') or os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('TF_VAR_AZURE_CLIENT_ID') or os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('TF_VAR_AZURE_CLIENT_SECRET') or os.getenv('AZURE_CLIENT_SECRET')
    subscription_id = os.getenv('TF_VAR_AZURE_SUBSCRIPTION_ID') or os.getenv('AZURE_SUBSCRIPTION_ID')
    
    if not all([tenant_id, client_id, client_secret]):
        print("[ERROR] Missing Azure credentials. Run load-env.ps1 first.")
        print("Required: TF_VAR_AZURE_TENANT_ID, TF_VAR_AZURE_CLIENT_ID, TF_VAR_AZURE_CLIENT_SECRET")
        print("Or: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        return
    
    print("\n[OK] Connecting to Azure ML workspace...")
    auth = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=client_id,
        service_principal_password=client_secret
    )
    
    # Get workspace details (prefer env vars, fall back to config file)
    resource_group = os.getenv('AZURE_RESOURCE_GROUP') or (AZURE_ML_CONFIG and AZURE_ML_CONFIG['resource_group'])
    workspace_name = os.getenv('AZURE_WORKSPACE_NAME') or (AZURE_ML_CONFIG and AZURE_ML_CONFIG['workspace_name'])
    
    if not subscription_id:
        subscription_id = AZURE_ML_CONFIG and AZURE_ML_CONFIG['subscription_id']
    
    if not all([subscription_id, resource_group, workspace_name]):
        print("[ERROR] Missing workspace configuration.")
        print("Provide via environment variables: TF_VAR_AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME")
        print("Or configure azure_config.py for local development.")
        return
    
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        auth=auth
    )
    print(f"[OK] Connected to workspace: {ws.name}")
    
    # Set MLflow tracking URI
    tracking_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[OK] MLflow tracking URI set")
    
    # Load data
    print("\n[OK] Loading data...")
    data_path = "C:/Users/Gopi.Thiruvengadam/Documents/GitHub/udemy/ml-flow-course/data/red-wine-quality.csv"
    data = pd.read_csv(data_path)
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    
    # Split features and target
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"[OK] Data loaded: {len(X_train)} training, {len(X_test)} test samples")
    
    # Start MLflow run
    print(f"\n[OK] Training model with best parameters...")
    print(f"    alpha: {BEST_ALPHA}")
    print(f"    l1_ratio: {BEST_L1_RATIO}")
    
    mlflow.set_experiment("wine-quality-comprehensive-optimization")
    
    with mlflow.start_run(run_name="best_model_for_registration") as run:
        # Train model
        model = ElasticNet(
            alpha=BEST_ALPHA,
            l1_ratio=BEST_L1_RATIO,
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n[OK] Model trained successfully!")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    R²: {r2:.4f}")
        
        # Log parameters
        mlflow.log_param("alpha", BEST_ALPHA)
        mlflow.log_param("l1_ratio", BEST_L1_RATIO)
        mlflow.log_param("max_iter", 1000)
        
        # Log metrics
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        
        # Log tags
        mlflow.set_tag("model_type", "ElasticNet")
        mlflow.set_tag("purpose", "production_registration")
        mlflow.set_tag("optimization_result", "best_from_grid_search")
        
        # Log the model (use save + log_artifact to avoid logged-models API)
        print(f"\n[OK] Logging model to MLflow...")
        
        import tempfile
        
        # Save model to temporary directory
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model")
        
        # Save the model using mlflow format
        mlflow.sklearn.save_model(model, model_path)
        
        # Log the model directory as an artifact
        mlflow.log_artifacts(model_path, artifact_path="model")
        
        run_id = run.info.run_id
        print(f"[OK] Model logged with run ID: {run_id}")
        
        # Register the model using Azure ML SDK (MLflow register_model fails with 404)
        print(f"\n[OK] Registering model as 'my_ver_1' using Azure ML SDK...")
        
        try:
            from azureml.core import Model
            import shutil
            
            # Download the model artifacts from the run
            client = mlflow.tracking.MlflowClient()
            downloaded_model = client.download_artifacts(run_id, "model")
            
            # Copy to current directory
            local_model_dir = "best_model_temp"
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            shutil.copytree(downloaded_model, local_model_dir)
            
            print(f"[OK] Model artifacts copied to: {local_model_dir}")
            
            # Register the model from local path
            model_to_register = Model.register(
                workspace=ws,
                model_name="my_ver_1",
                model_path=local_model_dir,
                description=f"ElasticNet wine quality model (alpha={BEST_ALPHA}, l1_ratio={BEST_L1_RATIO})",
                tags={
                    "alpha": str(BEST_ALPHA),
                    "l1_ratio": str(BEST_L1_RATIO),
                    "test_rmse": f"{rmse:.4f}",
                    "test_mae": f"{mae:.4f}",
                    "test_r2": f"{r2:.4f}",
                    "model_type": "ElasticNet",
                    "purpose": "production",
                    "source": "grid_search_optimization",
                    "mlflow_run_id": run_id
                }
            )
            
            # Cleanup
            shutil.rmtree(local_model_dir)
            
            print(f"\n" + "="*70)
            print("MODEL REGISTRATION COMPLETE!")
            print("="*70)
            print(f"Model Name: {model_to_register.name}")
            print(f"Version: {model_to_register.version}")
            print(f"Test RMSE: {rmse:.4f}")
            print(f"Run ID: {run_id}")
            
            # Construct portal URL
            tenant_id = os.getenv('AZURE_TENANT_ID')
            workspace_id = f"/subscriptions/{ws.subscription_id}/resourcegroups/{ws.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{ws.name}"
            portal_url = f"https://ml.azure.com?tid={tenant_id}&wsid={workspace_id}"
            
            print(f"\nView in Azure ML Studio:")
            print(f"{portal_url}#model/list")
            print("="*70)
            
            return model_to_register
            
        except Exception as e:
            print(f"[ERROR] Failed to register model: {e}")
            print(f"[INFO] Model was logged to run {run_id} but registration failed")
            return None

if __name__ == "__main__":
    train_and_register_best_model()
