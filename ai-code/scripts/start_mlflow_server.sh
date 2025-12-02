# MLflow Remote Server Setup Script
# Run this on a dedicated server machine

# Option 1: Basic MLflow server (SQLite + local files)
# python -m mlflow server --host 0.0.0.0 --port 5000

# Option 2: With PostgreSQL (recommended for production)
# Requires: pip install psycopg2-binary
# mlflow server \
#   --backend-store-uri postgresql://username:password@localhost:5432/mlflow \
#   --default-artifact-root ./mlflow-artifacts \
#   --host 0.0.0.0 \
#   --port 5000

# Option 3: With Azure Blob Storage
# Requires: pip install azure-storage-blob
# Set environment variable: AZURE_STORAGE_CONNECTION_STRING
# mlflow server \
#   --backend-store-uri sqlite:///mlflow.db \
#   --default-artifact-root wasbs://container@account.blob.core.windows.net/mlflow \
#   --host 0.0.0.0 \
#   --port 5000

# Option 4: With AWS S3
# Requires: pip install boto3
# Set AWS credentials in environment
# mlflow server \
#   --backend-store-uri sqlite:///mlflow.db \
#   --default-artifact-root s3://bucket-name/mlflow-artifacts \
#   --host 0.0.0.0 \
#   --port 5000

# For Windows PowerShell, run:
# python -m mlflow server --host 0.0.0.0 --port 5000

# Then update your scripts to use:
# TRACKING_URI = "http://<server-ip>:5000"
