"""
Cleanup MLflow Experiments in Azure ML Workspace

This script helps manage and cleanup experiments in your Azure ML workspace.
Supports:
- Deleting specific experiments by name
- Deleting all experiments
- Listing all experiments
- Deleting runs within an experiment

Usage:
    python cleanup_experiments.py --list                          # List all experiments
    python cleanup_experiments.py --delete-experiment "name"      # Delete specific experiment
    python cleanup_experiments.py --delete-all                    # Delete all experiments
    python cleanup_experiments.py --delete-runs "experiment_name" # Delete all runs in experiment
"""

import os
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import mlflow
from mlflow.tracking import MlflowClient

# Import Azure ML configuration
from azure_config import AZURE_ML_CONFIG


def get_workspace():
    """
    Connect to Azure ML workspace using service principal authentication
    """
    # Check for service principal environment variables
    tenant_id = os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    
    auth = None
    if tenant_id and client_id and client_secret:
        print("[OK] Using Service Principal authentication from environment variables")
        auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=client_id,
            service_principal_password=client_secret
        )
    else:
        print("[INFO] No service principal environment variables found, using interactive auth")
        print("  Set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET for non-interactive auth")
    
    try:
        ws = Workspace(
            subscription_id=AZURE_ML_CONFIG['subscription_id'],
            resource_group=AZURE_ML_CONFIG['resource_group'],
            workspace_name=AZURE_ML_CONFIG['workspace_name'],
            auth=auth
        )
        print(f"[OK] Connected to workspace: {ws.name}")
        return ws
    except Exception as e:
        print(f"[ERROR] Failed to connect to Azure ML workspace: {e}")
        return None


def setup_mlflow(workspace):
    """
    Configure MLflow to track to Azure ML workspace
    """
    tracking_uri = workspace.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[OK] MLflow tracking URI: {tracking_uri}")
    return MlflowClient()


def list_experiments(client):
    """
    List all experiments in the workspace
    """
    print("\n" + "="*70)
    print("EXPERIMENTS IN WORKSPACE")
    print("="*70)
    
    experiments = client.search_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        print(f"\nExperiment: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Lifecycle Stage: {exp.lifecycle_stage}")
        print(f"  Number of Runs: {len(runs)}")
        
        if runs:
            print(f"  Runs:")
            for run in runs[:5]:  # Show first 5 runs
                print(f"    - {run.info.run_name} ({run.info.run_id}) - {run.info.status}")
            if len(runs) > 5:
                print(f"    ... and {len(runs) - 5} more runs")
    
    print("\n" + "="*70)


def delete_experiment_by_name(client, experiment_name, force=False):
    """
    Delete a specific experiment by name
    """
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"[ERROR] Experiment '{experiment_name}' not found.")
            return False
        
        # Confirm deletion
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"\n[WARNING] About to delete experiment: {experiment_name}")
        print(f"  Experiment ID: {experiment.experiment_id}")
        print(f"  Number of runs: {len(runs)}")
        
        if not force:
            confirm = input("Are you sure you want to delete this experiment? (yes/no): ")
            
            if confirm.lower() != 'yes':
                print("[INFO] Deletion cancelled.")
                return False
        
        # Delete the experiment
        client.delete_experiment(experiment.experiment_id)
        print(f"[OK] Experiment '{experiment_name}' deleted successfully.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to delete experiment: {e}")
        return False


def delete_all_experiments(client, force=False):
    """
    Delete all experiments in the workspace
    """
    experiments = client.search_experiments()
    
    if not experiments:
        print("[INFO] No experiments found to delete.")
        return
    
    print(f"\n[WARNING] About to delete {len(experiments)} experiments:")
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  - {exp.name} ({len(runs)} runs)")
    
    if not force:
        confirm = input("\nAre you sure you want to delete ALL experiments? (yes/no): ")
        
        if confirm.lower() != 'yes':
            print("[INFO] Deletion cancelled.")
            return
    
    deleted_count = 0
    for exp in experiments:
        try:
            client.delete_experiment(exp.experiment_id)
            print(f"[OK] Deleted: {exp.name}")
            deleted_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to delete {exp.name}: {e}")
    
    print(f"\n[OK] Deleted {deleted_count} of {len(experiments)} experiments.")


def delete_runs_in_experiment(client, experiment_name, force=False):
    """
    Delete all runs within a specific experiment
    """
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"[ERROR] Experiment '{experiment_name}' not found.")
            return False
        
        # Search for ALL runs including deleted ones
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=3  # ViewType.ALL (0=ACTIVE_ONLY, 1=DELETED_ONLY, 3=ALL)
        )
        
        if not runs:
            print(f"[INFO] No runs found in experiment '{experiment_name}'.")
            return True
        
        print(f"\n[WARNING] About to delete {len(runs)} runs from experiment: {experiment_name}")
        
        if not force:
            confirm = input("Are you sure you want to delete all runs? (yes/no): ")
            
            if confirm.lower() != 'yes':
                print("[INFO] Deletion cancelled.")
                return False
        
        deleted_count = 0
        for run in runs:
            try:
                # Only delete if not already deleted
                if run.info.lifecycle_stage != 'deleted':
                    client.delete_run(run.info.run_id)
                    deleted_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete run {run.info.run_id}: {e}")
        
        print(f"[OK] Deleted {deleted_count} of {len(runs)} runs.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to delete runs: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup MLflow experiments in Azure ML workspace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all experiments:
    python cleanup_experiments.py --list

  Delete specific experiment:
    python cleanup_experiments.py --delete-experiment "wine-quality-comprehensive-optimization"

  Delete all runs in an experiment:
    python cleanup_experiments.py --delete-runs "wine-quality-comprehensive-optimization"

  Delete all experiments:
    python cleanup_experiments.py --delete-all

Note: Set environment variables for non-interactive authentication:
  $env:AZURE_TENANT_ID = "your-tenant-id"
  $env:AZURE_CLIENT_ID = "your-client-id"
  $env:AZURE_CLIENT_SECRET = "your-secret"
        """
    )
    
    parser.add_argument('--list', action='store_true',
                        help='List all experiments in the workspace')
    parser.add_argument('--delete-experiment', type=str, metavar='NAME',
                        help='Delete a specific experiment by name')
    parser.add_argument('--delete-all', action='store_true',
                        help='Delete all experiments in the workspace')
    parser.add_argument('--delete-runs', type=str, metavar='EXPERIMENT_NAME',
                        help='Delete all runs within a specific experiment')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompts (use with caution)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Connect to Azure ML workspace
    workspace = get_workspace()
    if workspace is None:
        return
    
    # Setup MLflow client
    client = setup_mlflow(workspace)
    
    # Execute requested operation
    if args.list:
        list_experiments(client)
    
    if args.delete_experiment:
        delete_experiment_by_name(client, args.delete_experiment, force=args.force)
    
    if args.delete_runs:
        delete_runs_in_experiment(client, args.delete_runs, force=args.force)
    
    if args.delete_all:
        delete_all_experiments(client, force=args.force)


if __name__ == "__main__":
    main()
