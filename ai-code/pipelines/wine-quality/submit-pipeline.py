"""
Submit Wine Quality Training Pipeline to Azure ML

This script submits the pipeline to Azure ML workspace for execution.
"""

import os
from azure.ai.ml import MLClient, Input, load_job
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute


def get_ml_client():
    """Get Azure ML workspace client using environment variables"""
    subscription_id = os.getenv('TF_VAR_AZURE_SUBSCRIPTION_ID') or os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_WORKSPACE_NAME')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise EnvironmentError(
            "Missing required environment variables:\n"
            "  - TF_VAR_AZURE_SUBSCRIPTION_ID (or AZURE_SUBSCRIPTION_ID)\n"
            "  - AZURE_RESOURCE_GROUP\n"
            "  - AZURE_WORKSPACE_NAME\n"
            "Run scripts/load-env.ps1 first."
        )
    
    # Create ML client
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    return ml_client


def create_compute_if_not_exists(ml_client, compute_name="cpu-cluster"):
    """Create compute cluster if it doesn't exist"""
    try:
        ml_client.compute.get(compute_name)
        print(f"[OK] Compute cluster '{compute_name}' already exists")
    except Exception:
        print(f"[INFO] Creating compute cluster '{compute_name}'...")
        compute = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=300
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"[OK] Compute cluster '{compute_name}' created")


def upload_data_if_not_exists(ml_client, local_data_path):
    """Upload wine quality data to workspace datastore if needed"""
    datastore_name = "workspaceblobstore"
    data_path = "data/red-wine-quality.csv"
    
    print(f"[INFO] Checking if data exists in datastore...")
    
    # For simplicity, we'll always upload (you can add existence check)
    print(f"[INFO] Uploading data from {local_data_path}...")
    
    # Upload using datastore upload
    # Note: You may need to adjust this based on your data location
    print(f"[WARNING] Please ensure data is uploaded to:")
    print(f"  azureml://datastores/{datastore_name}/paths/{data_path}")
    print(f"  You can upload via Azure ML Studio or Azure Storage Explorer")


def main():
    print("="*70)
    print("WINE QUALITY PIPELINE SUBMISSION")
    print("="*70)
    
    # Get ML client
    print("\n[INFO] Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    print(f"[OK] Connected to workspace: {ml_client.workspace_name}")
    
    # Create compute if needed
    print("\n[INFO] Checking compute cluster...")
    create_compute_if_not_exists(ml_client)
    
    # Load pipeline
    print("\n[INFO] Loading pipeline definition...")
    pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.yml")
    
    if not os.path.exists(pipeline_path):
        print(f"[ERROR] Pipeline file not found: {pipeline_path}")
        return
    
    pipeline_job = load_job(source=pipeline_path)
    
    # Submit pipeline
    print("\n[INFO] Submitting pipeline to Azure ML...")
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="wine-quality-pipeline"
    )
    
    print(f"\n[SUCCESS] Pipeline submitted!")
    print(f"  - Job Name: {submitted_job.name}")
    print(f"  - Experiment: {submitted_job.experiment_name}")
    print(f"  - Status: {submitted_job.status}")
    
    # Get portal URL
    portal_url = f"https://ml.azure.com/runs/{submitted_job.name}"
    print(f"\n[INFO] View pipeline run in Azure ML Studio:")
    print(f"  {portal_url}")
    
    # Stream logs (optional)
    print("\n[INFO] Streaming job logs... (Press Ctrl+C to stop)")
    try:
        ml_client.jobs.stream(submitted_job.name)
    except KeyboardInterrupt:
        print("\n[INFO] Log streaming stopped")
    
    print("="*70)


if __name__ == "__main__":
    main()
