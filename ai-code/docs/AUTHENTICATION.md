# Azure ML Authentication - Quick Reference

## ✅ Configuration Complete

The wine quality comprehensive experiment now supports **two authentication methods**:

### 1. **Service Principal (Non-Interactive)** - Recommended for CI/CD
- Uses environment variables
- No browser popup required
- Perfect for Azure DevOps pipelines

### 2. **Interactive Authentication** - For local development
- Automatic fallback if environment variables not set
- Browser-based Azure AD login

---

## 📁 Files Added

| File | Purpose |
|------|---------|
| `.env.example` | Template for service principal credentials |
| `load-env.ps1` | PowerShell script to load environment variables |
| `AUTH_SETUP.md` | Comprehensive authentication setup guide |
| `check-env.py` | Verify environment variable configuration |

---

## 🚀 Quick Start

### For Local Development (Interactive Auth)
```powershell
cd ai-code/g0pinath/udemy-ml-flow-course/ai-code
python wine_quality_comprehensive.py
# Browser will open for Azure AD login
```

### For CI/CD or Non-Interactive (Service Principal)

**Step 1: Create Service Principal**
```bash
az ad sp create-for-rbac --name "mlflow-experiment-sp" --role Contributor \
  --scopes /subscriptions/813f951c-481f-491f-a03c-236e68d61659/resourceGroups/rg-gopi-ai900-dev
```

**Step 2: Configure Environment**
```powershell
# Copy template
Copy-Item .env.example .env

# Edit .env with your service principal credentials
# Then load:
.\load-env.ps1
```

**Step 3: Run Experiments**
```powershell
python wine_quality_comprehensive.py
```

---

## 🔍 Verify Setup

```powershell
python check-env.py
```

Expected output when configured:
```
Environment Variables Status:
==================================================
AZURE_TENANT_ID:     ✓ Set
AZURE_CLIENT_ID:     ✓ Set
AZURE_CLIENT_SECRET: ✓ Set
==================================================

✓ Service Principal authentication configured
  Script will use non-interactive authentication
```

---

## 🔐 Environment Variables

| Variable | Description | Value |
|----------|-------------|-------|
| `AZURE_TENANT_ID` | Azure AD Tenant ID | `20f4aea2-36b7-45bf-bb52-d91200496ae8` |
| `AZURE_CLIENT_ID` | Service Principal App ID | From `az ad sp create-for-rbac` output |
| `AZURE_CLIENT_SECRET` | Service Principal Secret | From `az ad sp create-for-rbac` output |

---

## 📊 What's Tracked in Azure ML

When you run experiments, the following are logged to Azure ML workspace:

✅ **Parameters**: `alpha`, `l1_ratio`, grid search configs  
✅ **Metrics**: `test_rmse`, `test_mae`, `test_r2`, `cv_best_score`  
✅ **Tags**: `model_type`, `experiment_phase`, `strategy`  
✅ **Artifacts**: Model files (when `LOG_MODELS=True`)

**View experiments at:**  
https://ml.azure.com?tid=20f4aea2-36b7-45bf-bb52-d91200496ae8&wsid=/subscriptions/813f951c-481f-491f-a03c-236e68d61659/resourcegroups/rg-gopi-ai900-dev/providers/Microsoft.MachineLearningServices/workspaces/aml-ws-hopeful-mayfly

---

## 🛠️ Troubleshooting

### Issue: "Interactive authentication required"
**Solution:** Environment variables not set. Run `.\load-env.ps1` or set them manually

### Issue: "Failed to connect to Azure ML workspace"
**Solution:** Check service principal has Contributor access to resource group

### Issue: "No module named 'azureml'"
**Solution:** Install dependencies:
```powershell
pip install azureml-core azureml-mlflow setuptools
```

---

## 📝 Security Notes

- ✅ `.env` files are in `.gitignore` - never committed
- ✅ Use Azure Key Vault for production secrets
- ✅ Rotate service principal secrets every 90 days
- ✅ Limit permissions to required scope only

---

## 🔄 Next Steps

1. **Test locally** with interactive auth first
2. **Create service principal** for automated runs
3. **Configure `.env`** file with credentials
4. **Run experiments** with environment variables
5. **View results** in Azure ML Studio

For detailed instructions, see **AUTH_SETUP.md**
