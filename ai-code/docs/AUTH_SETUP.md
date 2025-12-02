# Azure ML Service Principal Authentication Setup

This directory contains scripts for authenticating to Azure ML using environment variables instead of interactive browser authentication.

## Quick Start

### Option 1: Using PowerShell (Windows)

1. **Create a Service Principal** (one-time setup):
   ```powershell
   az ad sp create-for-rbac --name "mlflow-experiment-sp" --role Contributor `
     --scopes /subscriptions/813f951c-481f-491f-a03c-236e68d61659/resourceGroups/rg-gopi-ai900-dev
   ```

2. **Copy and configure `.env` file**:
   ```powershell
   Copy-Item .env.example .env
   # Edit .env and replace placeholder values with output from step 1
   ```

3. **Load environment variables and run experiments**:
   ```powershell
   .\load-env.ps1
   python wine_quality_comprehensive.py
   ```

### Option 2: Set Environment Variables Directly

**PowerShell:**
```powershell
$env:AZURE_TENANT_ID = "20f4aea2-36b7-45bf-bb52-d91200496ae8"
$env:AZURE_CLIENT_ID = "your-client-id"
$env:AZURE_CLIENT_SECRET = "your-client-secret"
python wine_quality_comprehensive.py
```

**Command Prompt:**
```cmd
set AZURE_TENANT_ID=20f4aea2-36b7-45bf-bb52-d91200496ae8
set AZURE_CLIENT_ID=your-client-id
set AZURE_CLIENT_SECRET=your-client-secret
python wine_quality_comprehensive.py
```

**Linux/Mac:**
```bash
export AZURE_TENANT_ID="20f4aea2-36b7-45bf-bb52-d91200496ae8"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
python wine_quality_comprehensive.py
```

## Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_TENANT_ID` | Azure AD Tenant ID | `20f4aea2-36b7-45bf-bb52-d91200496ae8` |
| `AZURE_CLIENT_ID` | Service Principal Application ID | `12345678-1234-1234-1234-123456789012` |
| `AZURE_CLIENT_SECRET` | Service Principal Secret | `your-secret-value` |

## Creating a Service Principal

### Using Azure CLI:

```bash
az ad sp create-for-rbac \
  --name "mlflow-experiment-sp" \
  --role Contributor \
  --scopes /subscriptions/813f951c-481f-491f-a03c-236e68d61659/resourceGroups/rg-gopi-ai900-dev
```

This will output:
```json
{
  "appId": "12345678-1234-1234-1234-123456789012",
  "displayName": "mlflow-experiment-sp",
  "password": "your-secret-here",
  "tenant": "20f4aea2-36b7-45bf-bb52-d91200496ae8"
}
```

Map these values to environment variables:
- `appId` → `AZURE_CLIENT_ID`
- `password` → `AZURE_CLIENT_SECRET`
- `tenant` → `AZURE_TENANT_ID`

### Using Azure Portal:

1. Go to **Azure Active Directory** → **App registrations** → **New registration**
2. Note the **Application (client) ID** → `AZURE_CLIENT_ID`
3. Note the **Directory (tenant) ID** → `AZURE_TENANT_ID`
4. Go to **Certificates & secrets** → **New client secret**
5. Copy the secret value → `AZURE_CLIENT_SECRET`
6. Assign **Contributor** role to the service principal on your resource group

## Azure DevOps Pipeline Integration

Add these variables to your Azure DevOps pipeline:

```yaml
variables:
  AZURE_TENANT_ID: '20f4aea2-36b7-45bf-bb52-d91200496ae8'
  AZURE_CLIENT_ID: '$(azureClientId)'  # Store in pipeline variables
  AZURE_CLIENT_SECRET: '$(azureClientSecret)'  # Store as secret variable

steps:
- script: |
    export AZURE_TENANT_ID=$(AZURE_TENANT_ID)
    export AZURE_CLIENT_ID=$(AZURE_CLIENT_ID)
    export AZURE_CLIENT_SECRET=$(AZURE_CLIENT_SECRET)
    python wine_quality_comprehensive.py
  displayName: 'Run MLflow Experiments with Service Principal Auth'
```

## Troubleshooting

### "No service principal environment variables found"
- The script will fall back to interactive authentication
- Set the required environment variables before running

### "Failed to connect to Azure ML workspace"
- Verify service principal has Contributor access to the resource group
- Check that environment variables are set correctly
- Ensure service principal credentials haven't expired

### "Interactive authentication required"
- Service principal authentication failed or not configured
- Check credentials and permissions
- Try interactive auth first to verify workspace access

## Security Best Practices

1. **Never commit `.env` files** - Already in `.gitignore`
2. **Use Azure Key Vault** for production secrets
3. **Rotate secrets regularly** (every 90 days recommended)
4. **Use managed identities** when running on Azure (VMs, Container Instances, etc.)
5. **Limit service principal permissions** to minimum required scope

## Files

- `.env.example` - Template for environment variables
- `load-env.ps1` - PowerShell script to load .env file
- `azure_config.py` - Azure ML workspace configuration
- `wine_quality_comprehensive.py` - Main experiment script (supports both auth methods)
