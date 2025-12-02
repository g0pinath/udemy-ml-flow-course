"""
Cleanup all experiment runs in Azure ML workspace using MLflow
"""
import os
import mlflow
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from azure_config import AZURE_ML_CONFIG

def cleanup_all_runs():
    """Delete all runs from the wine quality experiment"""
    
    # Connect to workspace
    print("Connecting to Azure ML workspace...")
    auth = ServicePrincipalAuthentication(
        tenant_id=os.getenv('AZURE_TENANT_ID'),
        service_principal_id=os.getenv('AZURE_CLIENT_ID'),
        service_principal_password=os.getenv('AZURE_CLIENT_SECRET')
    )
    
    ws = Workspace(
        subscription_id=AZURE_ML_CONFIG['subscription_id'],
        resource_group=AZURE_ML_CONFIG['resource_group'],
        workspace_name=AZURE_ML_CONFIG['workspace_name'],
        auth=auth
    )
    
    print(f"✓ Connected to workspace: {ws.name}\n")
    
    # Set MLflow tracking URI
    tracking_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get MLflow client
    client = mlflow.tracking.MlflowClient()
    
    # List all experiments
    print("Listing all experiments...")
    experiments = client.search_experiments()
    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # Get the main experiment
    exp_name = 'wine-quality-comprehensive-optimization'
    target_exp = None
    for exp in experiments:
        if exp.name == exp_name:
            target_exp = exp
            break
    
    if target_exp:
        # Get all runs
        runs = client.search_runs(
            experiment_ids=[target_exp.experiment_id],
            max_results=1000
        )
        
        if len(runs) == 0:
            print(f"\n✓ Experiment '{exp_name}' has no runs to delete")
            return
        
        print(f"\nExperiment '{exp_name}' has {len(runs)} runs")
        
        # Ask for confirmation
        response = input(f"\nDelete all {len(runs)} runs? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            return
        
        print("\nDeleting all runs...")
        deleted = 0
        failed = 0
        
        for run in runs:
            try:
                client.delete_run(run.info.run_id)
                deleted += 1
                print(f"  ✓ Deleted run: {run.info.run_id}")
            except Exception as e:
                failed += 1
                print(f"  ✗ Failed to delete {run.info.run_id}: {e}")
        
        print(f"\n{'='*60}")
        print(f"CLEANUP COMPLETE")
        print(f"{'='*60}")
        print(f"Deleted: {deleted} runs")
        print(f"Failed: {failed} runs")
        print(f"{'='*60}")
        
    else:
        print(f"\nExperiment '{exp_name}' not found")

if __name__ == "__main__":
    cleanup_all_runs()
