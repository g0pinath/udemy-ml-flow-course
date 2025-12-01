"""
MLflow Remote Storage Configuration Examples
"""

import os
import mlflow

# =============================================================================
# Option 1: Remote MLflow Tracking Server (Self-hosted)
# =============================================================================
def configure_remote_server():
    """Configure MLflow to use a remote tracking server"""
    mlflow.set_tracking_uri("http://your-server-ip:5000")
    
    # Optional: Set experiment
    mlflow.set_experiment("my-experiment")
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")


# =============================================================================
# Option 2: Azure Blob Storage Backend
# =============================================================================
def configure_azure_storage():
    """Configure MLflow with Azure Blob Storage for artifacts"""
    
    # Set environment variables for Azure credentials
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "your-connection-string"
    # OR use SAS token
    # os.environ["AZURE_STORAGE_ACCESS_KEY"] = "your-access-key"
    
    # Set tracking URI to remote server
    mlflow.set_tracking_uri("http://your-mlflow-server:5000")
    
    # Server should be configured with Azure backend:
    # mlflow server \
    #   --backend-store-uri sqlite:///mlflow.db \
    #   --default-artifact-root wasbs://container@storageaccount.blob.core.windows.net/mlflow-artifacts \
    #   --host 0.0.0.0 \
    #   --port 5000


# =============================================================================
# Option 3: AWS S3 Backend
# =============================================================================
def configure_s3_storage():
    """Configure MLflow with S3 storage for artifacts"""
    
    # Set AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    
    # Set tracking URI
    mlflow.set_tracking_uri("http://your-mlflow-server:5000")
    
    # Server should be configured with S3 backend:
    # mlflow server \
    #   --backend-store-uri sqlite:///mlflow.db \
    #   --default-artifact-root s3://my-bucket/mlflow-artifacts \
    #   --host 0.0.0.0 \
    #   --port 5000


# =============================================================================
# Option 4: PostgreSQL + S3 (Production Setup)
# =============================================================================
def configure_production():
    """Production MLflow setup with PostgreSQL and S3"""
    
    # Server configuration (run this on your MLflow server):
    """
    mlflow server \
      --backend-store-uri postgresql://user:password@postgres-host:5432/mlflow \
      --default-artifact-root s3://mlflow-bucket/artifacts \
      --host 0.0.0.0 \
      --port 5000
    """
    
    # Client configuration:
    mlflow.set_tracking_uri("http://mlflow-server:5000")


# =============================================================================
# Option 5: Databricks
# =============================================================================
def configure_databricks():
    """Configure MLflow for Databricks"""
    
    # Set Databricks credentials
    os.environ["DATABRICKS_HOST"] = "https://your-workspace.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "your-access-token"
    
    # Set tracking URI
    mlflow.set_tracking_uri("databricks")
    
    # Set experiment path
    mlflow.set_experiment("/Users/your-email@example.com/wine-quality")


# =============================================================================
# Option 6: Azure ML
# =============================================================================
def configure_azure_ml():
    """Configure MLflow for Azure Machine Learning"""
    
    from azureml.core import Workspace
    
    # Connect to workspace
    ws = Workspace.from_config()
    
    # Set MLflow tracking URI to Azure ML
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    
    # Set experiment
    mlflow.set_experiment("wine-quality-optimization")


# =============================================================================
# Option 7: Simple HTTP Authentication
# =============================================================================
def configure_with_auth():
    """Configure MLflow with basic authentication"""
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = "your-username"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-password"
    
    mlflow.set_tracking_uri("http://your-mlflow-server:5000")


# =============================================================================
# Helper: Start Local MLflow Server (for testing)
# =============================================================================
"""
To start a local MLflow server that can accept remote connections:

# Basic server
mlflow server --host 0.0.0.0 --port 5000

# With PostgreSQL backend and local file storage
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000

# With SQLite and S3
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://your-bucket/mlflow \
  --host 0.0.0.0 \
  --port 5000

# Then from client code:
mlflow.set_tracking_uri("http://your-server-ip:5000")
"""


if __name__ == "__main__":
    print("MLflow Remote Storage Configuration Examples")
    print("=" * 70)
    print("\nChoose your backend:")
    print("1. Remote MLflow Server")
    print("2. Azure Blob Storage")
    print("3. AWS S3")
    print("4. Databricks")
    print("5. Azure ML")
    print("\nUpdate TRACKING_URI in your scripts accordingly.")
