# MLflow Azure ML Integration

Wine quality prediction experiments using MLflow with Azure ML workspace tracking.

## Project Structure

```
ai-code/
├── experiments/          # Main experiment scripts
│   ├── wine_quality_comprehensive.py   # Comprehensive optimization suite
│   └── train_and_register_best.py      # Train & register best model
├── scripts/              # Utility scripts
│   ├── load-env.ps1                    # Environment setup (local/pipeline)
│   ├── cleanup_experiments.py          # Manage MLflow experiments
│   ├── test_azure_connection.py        # Test Azure ML connectivity
│   └── start_mlflow_server.sh          # Start MLflow tracking server
├── config/               # Configuration files
│   ├── azure_config.py                 # Azure ML workspace config
│   ├── config.json                     # General configuration
│   └── requirements.txt                # Python dependencies
└── docs/                 # Documentation
    ├── AUTHENTICATION.md               # Azure authentication guide
    ├── AUTH_SETUP.md                   # Setup instructions
    ├── SECURITY.md                     # Security & credential management
    └── SUCCESS_SUMMARY.md              # Project achievements
```

## Security

⚠️ **Important**: Configuration files with credentials are git-ignored.
- See [docs/SECURITY.md](docs/SECURITY.md) for setup instructions
- Use `*.example` template files to create your local config
- Never commit actual credentials to version control

## Quick Start

### 1. Setup Environment

**First-time setup:**
```powershell
# Copy template files
Copy-Item scripts\load-env.ps1.example scripts\load-env.ps1
Copy-Item config\azure_config.py.example config\azure_config.py
Copy-Item config\config.json.example config\config.json

# Edit these files with your Azure credentials (see docs/SECURITY.md)
```

**Load credentials:**
```powershell
# Load Azure credentials (from ai-code root)
.\scripts\load-env.ps1

# Install dependencies
pip install -r config\requirements.txt
```

### 2. Run Experiments
```powershell
# Run comprehensive optimization (from ai-code root)
python experiments\wine_quality_comprehensive.py

# Train and register best model (from ai-code root)
python experiments\train_and_register_best.py
```

### 3. Manage Experiments
```powershell
# List all experiments
python scripts\cleanup_experiments.py --list

# Delete specific experiment
python scripts\cleanup_experiments.py --delete-experiment <name>

# Test Azure connection
python scripts\test_azure_connection.py
```

## Results

- **Best Model**: ElasticNet (alpha=0.01, l1_ratio=0.9)
- **Test RMSE**: 0.6258 (20.51% improvement over baseline)
- **Registered Model**: `my_ver_1` in Azure ML Model Registry

## View in Azure ML

🔗 [Azure ML Studio](https://ml.azure.com?tid=20f4aea2-36b7-45bf-bb52-d91200496ae8&wsid=/subscriptions/813f951c-481f-491f-a03c-236e68d61659/resourcegroups/rg-gopi-ai900-dev/providers/Microsoft.MachineLearningServices/workspaces/aml-ws-hopeful-mayfly)

## Documentation

- [Authentication Setup](docs/AUTHENTICATION.md) - Azure service principal setup
- [Auth Configuration](docs/AUTH_SETUP.md) - Detailed auth instructions
- [Success Summary](docs/SUCCESS_SUMMARY.md) - Project achievements
