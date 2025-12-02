"""
Test Azure ML Workspace Connection
Quick test script to verify Azure ML SDK installation and workspace connection
"""

import sys

# Test 1: Check if Azure ML SDK is installed
print("="*70)
print("Testing Azure ML SDK Installation")
print("="*70)

try:
    from azureml.core import Workspace
    print("✓ Azure ML SDK imported successfully")
    print(f"✓ SDK Version: {Workspace.__module__}")
except ImportError as e:
    print(f"✗ Failed to import Azure ML SDK: {e}")
    sys.exit(1)

# Test 2: Check if azure_config.py exists and is configured
print("\n" + "="*70)
print("Checking Azure ML Configuration")
print("="*70)

try:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
    from azure_config import AZURE_ML_CONFIG
    print("✓ azure_config.py imported successfully")
    
    # Check if configuration is populated
    required_keys = ['subscription_id', 'resource_group', 'workspace_name', 'location']
    missing_keys = []
    
    for key in required_keys:
        if key not in AZURE_ML_CONFIG:
            missing_keys.append(key)
        elif AZURE_ML_CONFIG[key] == "YOUR_SUBSCRIPTION_ID_HERE" or \
             AZURE_ML_CONFIG[key] == "YOUR_RESOURCE_GROUP_HERE" or \
             AZURE_ML_CONFIG[key] == "YOUR_WORKSPACE_NAME_HERE" or \
             AZURE_ML_CONFIG[key] == "eastus":
            print(f"⚠ {key}: {AZURE_ML_CONFIG[key]} (needs to be configured)")
        else:
            print(f"✓ {key}: {AZURE_ML_CONFIG[key][:30]}..." if len(str(AZURE_ML_CONFIG[key])) > 30 else f"✓ {key}: {AZURE_ML_CONFIG[key]}")
    
    if missing_keys:
        print(f"\n⚠ Missing configuration keys: {', '.join(missing_keys)}")
        print("\nPlease edit azure_config.py and add your Azure ML workspace details")
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Failed to import azure_config.py: {e}")
    print("\nPlease create azure_config.py with your Azure ML workspace details")
    sys.exit(1)

# Test 3: Try to connect to workspace (optional - only if configured)
print("\n" + "="*70)
print("Testing Azure ML Workspace Connection")
print("="*70)

try:
    # Try loading from config.json first
    try:
        ws = Workspace.from_config()
        print(f"✓ Connected to workspace using config.json: {ws.name}")
    except:
        # Try using azure_config.py
        ws = Workspace(
            subscription_id=AZURE_ML_CONFIG['subscription_id'],
            resource_group=AZURE_ML_CONFIG['resource_group'],
            workspace_name=AZURE_ML_CONFIG['workspace_name']
        )
        print(f"✓ Connected to workspace: {ws.name}")
    
    print(f"✓ Subscription: {ws.subscription_id}")
    print(f"✓ Resource Group: {ws.resource_group}")
    print(f"✓ Location: {ws.location}")
    
    # Get MLflow tracking URI
    tracking_uri = ws.get_mlflow_tracking_uri()
    print(f"✓ MLflow Tracking URI: {tracking_uri}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - Ready to use Azure ML for MLflow tracking!")
    print("="*70)
    
except Exception as e:
    print(f"⚠ Could not connect to Azure ML workspace: {e}")
    print("\nPossible reasons:")
    print("1. Azure ML workspace doesn't exist yet")
    print("2. Authentication not configured (run 'az login' or use service principal)")
    print("3. Incorrect workspace details in azure_config.py")
    print("\nYou can still use local MLflow tracking by setting USE_AZURE_ML = False")
    print("="*70)
