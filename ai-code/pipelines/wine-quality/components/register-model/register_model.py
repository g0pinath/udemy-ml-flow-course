"""
Model Registration Component for Wine Quality Pipeline

Registers the best model from a sweep job to Azure ML Model Registry.
"""

import argparse
import os
import mlflow
from azureml.core import Run


def parse_args():
    parser = argparse.ArgumentParser("register_model")
    parser.add_argument("--model_input_path", type=str, help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="wine-quality-elasticnet", 
                       help="Name for registered model")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("MODEL REGISTRATION COMPONENT")
    print("="*70)
    
    # Get current Azure ML run context
    current_run = Run.get_context()
    ws = current_run.experiment.workspace
    
    # Set up MLflow tracking
    tracking_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_run.experiment.name)
    
    print(f"\n[INFO] Workspace: {ws.name}")
    print(f"[INFO] MLflow Tracking URI: {tracking_uri}")
    
    # Extract run ID from model path
    print(f"\n[INFO] Model input path: {args.model_input_path}")
    
    # Read MLmodel file to get run_id
    mlmodel_path = os.path.join(args.model_input_path, "MLmodel")
    run_id = ""
    
    if os.path.exists(mlmodel_path):
        with open(mlmodel_path, "r") as f:
            for line in f:
                if "run_id" in line:
                    run_id = line.split(":")[1].strip()
                    break
    
    if run_id:
        model_uri = f"runs:/{run_id}/model_output"
        print(f"[OK] Extracted Run ID: {run_id}")
        print(f"[OK] Model URI: {model_uri}")
    else:
        # Fallback: use model path directly
        model_uri = args.model_input_path
        print(f"[WARNING] Could not extract run_id, using path directly")
        print(f"[OK] Model URI: {model_uri}")
    
    # Register model
    print(f"\n[INFO] Registering model as: {args.model_name}")
    
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )
    
    print(f"[SUCCESS] Model registered!")
    print(f"  - Name: {registered_model.name}")
    print(f"  - Version: {registered_model.version}")
    print(f"  - Source: {registered_model.source}")
    
    # Add tags to the registered model version
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        name=args.model_name,
        version=registered_model.version,
        key="dataset",
        value="red_wine_quality"
    )
    client.set_model_version_tag(
        name=args.model_name,
        version=registered_model.version,
        key="model_type",
        value="ElasticNet"
    )
    
    print(f"[OK] Added metadata tags to model version")
    print("="*70)


if __name__ == "__main__":
    main()
