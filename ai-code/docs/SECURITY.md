# Security & Configuration Setup

## Sensitive Files Protected

The following files contain credentials and are excluded from version control:

### 🔒 Ignored Files:
- `scripts/load-env.ps1` - Azure service principal credentials
- `config/azure_config.py` - Azure ML workspace configuration
- `config/config.json` - Application configuration (may contain paths/URLs)
- `*.log` files - May contain sensitive output
- `best_model_temp/` - Temporary model artifacts
- `.env*` files - Environment variables

### 📝 Template Files (Commit These):
- `scripts/load-env.ps1.example` - Template for load-env.ps1
- `config/azure_config.py.example` - Template for azure_config.py
- `config/config.json.example` - Template for config.json

## Setup Instructions

### 1. Create Your Configuration Files

```powershell
# Copy template files
Copy-Item scripts\load-env.ps1.example scripts\load-env.ps1
Copy-Item config\azure_config.py.example config\azure_config.py
Copy-Item config\config.json.example config\config.json
```

### 2. Update With Your Credentials

Edit `scripts/load-env.ps1`:
```powershell
$env:AZURE_TENANT_ID = "YOUR_TENANT_ID_HERE"
$env:AZURE_CLIENT_ID = "YOUR_CLIENT_ID_HERE"
$env:AZURE_CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
$env:AZURE_SUBSCRIPTION_ID = "YOUR_SUBSCRIPTION_ID_HERE"
```

Edit `config/azure_config.py`:
```python
AZURE_ML_CONFIG = {
    'subscription_id': 'YOUR_SUBSCRIPTION_ID',
    'resource_group': 'YOUR_RESOURCE_GROUP',
    'workspace_name': 'YOUR_WORKSPACE_NAME',
    'location': 'YOUR_REGION'  # e.g., 'australiaeast'
}
```

### 3. Verify Git Ignore

```powershell
# These should NOT appear (protected by .gitignore)
git status | Select-String "load-env.ps1|azure_config.py|config.json"

# These SHOULD appear (templates are tracked)
git status | Select-String "\.example"
```

## ⚠️ Important Security Notes

1. **Never commit actual credentials** to version control
2. **Always use template files** (*.example) for sharing configuration structure
3. **Rotate credentials** if accidentally committed
4. **Use Azure Key Vault** for production deployments
5. **Use Azure Pipelines library groups** for CI/CD (see AZURE_PIPELINES_SETUP.md)

## For New Team Members

1. Get credentials from team lead or Azure portal
2. Copy template files to actual config files
3. Fill in credentials
4. Run `.\scripts\load-env.ps1` to test
5. Run `python scripts\test_azure_connection.py` to verify

## Azure Pipelines

Pipeline credentials are stored in the **mlflow** library group, not in code.
See `AZURE_PIPELINES_SETUP.md` for configuration instructions.
