"""
Run Wine Quality Pipeline using Azure ML Python SDK v2

This script builds and submits the pipeline programmatically without relying on YAML parsing.
"""

import os
from pathlib import Path
from azure.ai.ml import MLClient, Input, Output, dsl, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential


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
    
    # Try multiple credential methods
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential()
    )
    
    ml_client = MLClient(
        credential=credential,
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


def build_simple_pipeline(ml_client, data_path):
    """Build simple training pipeline (no hyperparameter sweep)"""
    
    # Get current directory
    base_dir = Path(__file__).parent
    
    # Load components
    print("[INFO] Loading pipeline components...")
    prep_component = load_component(source=str(base_dir / "components/prep-data/component.yml"))
    train_component = load_component(source=str(base_dir / "components/train-model/component.yml"))
    register_component = load_component(source=str(base_dir / "components/register-model/component.yml"))
    
    print("[OK] Components loaded successfully")
    
    # Define pipeline using Python DSL
    @dsl.pipeline(
        name="wine_quality_simple_pipeline",
        description="Simple wine quality training pipeline with fixed hyperparameters",
        default_compute="cpu-cluster"
    )
    def wine_quality_pipeline(
        raw_data: Input,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        model_name: str = "wine-quality-elasticnet"
    ):
        # Step 1: Prepare data
        prep_job = prep_component(
            input_data=raw_data,
            test_split_ratio=0.25,
            random_state=42
        )
        
        # Step 2: Train model
        train_job = train_component(
            train_data=prep_job.outputs.train_data,
            test_data=prep_job.outputs.test_data,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
            random_state=42
        )
        
        # Step 3: Register model
        register_job = register_component(
            model_input_path=train_job.outputs.model_output,
            model_name=model_name
        )
        
        return {
            "train_data": prep_job.outputs.train_data,
            "test_data": prep_job.outputs.test_data,
            "model": train_job.outputs.model_output
        }
    
    # Create pipeline instance
    pipeline_job = wine_quality_pipeline(
        raw_data=Input(type=AssetTypes.URI_FILE, path=data_path),
        alpha=0.1,
        l1_ratio=0.5,
        model_name="wine-quality-elasticnet"
    )
    
    return pipeline_job


def build_sweep_pipeline(ml_client, data_path):
    """Build training pipeline with hyperparameter sweep"""
    from azure.ai.ml.sweep import Choice, Uniform
    
    base_dir = Path(__file__).parent
    
    # Load components
    print("[INFO] Loading pipeline components for sweep...")
    prep_component = load_component(source=str(base_dir / "components/prep-data/component.yml"))
    train_component = load_component(source=str(base_dir / "components/train-model/component.yml"))
    register_component = load_component(source=str(base_dir / "components/register-model/component.yml"))
    
    print("[OK] Components loaded successfully")
    
    @dsl.pipeline(
        name="wine_quality_sweep_pipeline",
        description="Wine quality training pipeline with hyperparameter sweep",
        default_compute="cpu-cluster"
    )
    def wine_quality_sweep_pipeline(
        raw_data: Input,
        model_name: str = "wine-quality-elasticnet"
    ):
        # Step 1: Prepare data
        prep_job = prep_component(
            input_data=raw_data,
            test_split_ratio=0.25,
            random_state=42
        )
        
        # Step 2: Train with sweep
        train_job = train_component(
            train_data=prep_job.outputs.train_data,
            test_data=prep_job.outputs.test_data,
            max_iter=10000,
            random_state=42
        )
        
        # Configure sweep on train_job
        sweep_job = train_job.sweep(
            primary_metric="test_rmse",
            goal="minimize",
            sampling_algorithm="grid",
            compute="cpu-cluster"
        )
        
        # Define search space
        sweep_job.set_limits(
            max_total_trials=99,
            max_concurrent_trials=5,
            timeout=7200
        )
        
        # Set hyperparameter search space
        train_job.inputs.alpha = Choice([0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
        train_job.inputs.l1_ratio = Choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Step 3: Register best model
        register_job = register_component(
            model_input_path=sweep_job.outputs.model_output,
            model_name=model_name
        )
        
        return {
            "best_model": sweep_job.outputs.model_output
        }
    
    # Create pipeline instance
    pipeline_job = wine_quality_sweep_pipeline(
        raw_data=Input(type=AssetTypes.URI_FILE, path=data_path),
        model_name="wine-quality-elasticnet"
    )
    
    return pipeline_job


def main(use_sweep=False):
    print("="*70)
    print("WINE QUALITY PIPELINE - PYTHON SDK SUBMISSION")
    print("="*70)
    
    # Get ML client
    print("\n[INFO] Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    print(f"[OK] Connected to workspace: {ml_client.workspace_name}")
    
    # Create compute if needed
    print("\n[INFO] Checking compute cluster...")
    create_compute_if_not_exists(ml_client)
    
    # Define data path
    # Update this path to where your data is in the workspace
    data_path = "azureml://datastores/workspaceblobstore/paths/data/red-wine-quality.csv"
    
    # Build pipeline
    print(f"\n[INFO] Building {'sweep' if use_sweep else 'simple'} pipeline...")
    
    if use_sweep:
        pipeline_job = build_sweep_pipeline(ml_client, data_path)
        experiment_name = "wine-quality-sweep-pipeline"
    else:
        pipeline_job = build_simple_pipeline(ml_client, data_path)
        experiment_name = "wine-quality-simple-pipeline"
    
    # Submit pipeline
    print(f"\n[INFO] Submitting pipeline to Azure ML...")
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name=experiment_name
    )
    
    print(f"\n[SUCCESS] Pipeline submitted!")
    print(f"  - Job Name: {submitted_job.name}")
    print(f"  - Experiment: {submitted_job.experiment_name}")
    print(f"  - Status: {submitted_job.status}")
    
    # Get portal URL
    portal_url = f"https://ml.azure.com/runs/{submitted_job.name}?wsid=/subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{ml_client.workspace_name}"
    
    print(f"\n[INFO] View pipeline run in Azure ML Studio:")
    print(f"  {portal_url}")
    
    # Stream logs
    print("\n[INFO] Streaming job logs... (Press Ctrl+C to stop)")
    try:
        ml_client.jobs.stream(submitted_job.name)
    except KeyboardInterrupt:
        print("\n[INFO] Log streaming stopped")
    except Exception as e:
        print(f"\n[WARNING] Could not stream logs: {e}")
        print(f"[INFO] Check portal for job status: {portal_url}")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Check for --sweep flag
    use_sweep = "--sweep" in sys.argv
    
    if use_sweep:
        print("\n>>> Running with HYPERPARAMETER SWEEP (99 trials)")
    else:
        print("\n>>> Running SIMPLE pipeline (single training run)")
        print(">>> Use --sweep flag for hyperparameter tuning")
    
    main(use_sweep=use_sweep)
