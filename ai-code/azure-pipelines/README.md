# Azure DevOps Pipelines for MLOps

This directory contains **Azure DevOps YAML pipelines** for implementing MLOps workflows for the wine quality prediction model.

## 📋 Pipeline Overview

### **1. CI Pipeline** (`ci-pipeline.yml`)
**Trigger:** Automatic on PR/push to main  
**Purpose:** Code quality, linting, and unit testing

**Stages:**
- ✅ Code Quality & Linting (Black, Flake8, Pylint)
- ✅ Unit Tests (Pytest with coverage)
- ✅ Validate ML Experiments (Syntax checks, MLflow validation)

**When to use:** Every code commit or pull request

---

### **2. Training Pipeline** (`training-pipeline.yml`)
**Trigger:** Manual or scheduled (weekly)  
**Purpose:** Train and register ML models

**Stages:**
- ✅ Setup Environment (Verify Azure ML workspace)
- ✅ Train Model (Run `wine_quality_comprehensive.py`)
- ✅ Register Model (Save best model to registry)
- ✅ Notifications (Success/failure alerts)

**When to use:** 
- Weekly scheduled runs (Sunday 2 AM UTC)
- Manual trigger for ad-hoc retraining
- After data updates

---

### **3. Deployment Pipeline** (`deployment-pipeline.yml`)
**Trigger:** Manual  
**Purpose:** Deploy trained models to staging/production

**Stages:**
- ✅ Validate Model (Check model registry)
- ✅ Deploy to Staging (Test deployment)
- ✅ Test Staging (Smoke tests)
- ✅ Manual Approval Gate
- ✅ Deploy to Production

**When to use:** After training pipeline completes successfully

---

## 🚀 Quick Start

### **Prerequisites**

1. **Azure DevOps Organization** with a project
2. **Azure ML Workspace** provisioned
3. **Service Connection** configured in Azure DevOps:
   - Go to Project Settings → Service Connections
   - Create "Azure Resource Manager" connection
   - Name it: `azure-ml-connection`
   - Grant access to your Azure ML workspace

### **Setup Steps**

#### **Step 1: Add Pipeline to Azure DevOps**

```bash
# Option A: Via Azure DevOps UI
1. Go to Azure DevOps → Pipelines → New Pipeline
2. Select "Azure Repos Git" (or your source)
3. Select your repository
4. Choose "Existing Azure Pipelines YAML file"
5. Select path: /azure-pipelines/ci-pipeline.yml
6. Click "Run"

# Option B: Via Azure CLI
az pipelines create \
  --name "Wine-Quality-CI" \
  --repository <your-repo> \
  --branch main \
  --yml-path azure-pipelines/ci-pipeline.yml
```

#### **Step 2: Configure Pipeline Variables**

Set these variables in Azure DevOps pipeline settings:

```yaml
# Library Group: azure-ml-credentials
AZURE_TENANT_ID: <your-tenant-id>
AZURE_CLIENT_ID: <service-principal-client-id>
AZURE_CLIENT_SECRET: <service-principal-secret>  # Mark as secret
AZURE_SUBSCRIPTION_ID: <your-subscription-id>
AZURE_RESOURCE_GROUP: <your-resource-group>
AZURE_WORKSPACE_NAME: <your-workspace-name>
```

**To create variable group:**
```bash
# In Azure DevOps
Project Settings → Pipelines → Library → + Variable Group
Name: "azure-ml-credentials"
Add variables above (mark secrets appropriately)
```

#### **Step 3: Link Variable Group to Pipelines**

Edit each pipeline YAML and add at the top:

```yaml
variables:
  - group: azure-ml-credentials
```

---

## 📊 Pipeline Workflows

### **Complete MLOps Flow**

```
┌─────────────────────────────────────────────────────────────┐
│  Developer commits code                                     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  CI Pipeline (Automatic)                                    │
│  ├─ Code Quality Checks                                     │
│  ├─ Unit Tests                                              │
│  └─ Experiment Validation                                   │
└─────────────┬───────────────────────────────────────────────┘
              │ (Pass)
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Training Pipeline (Scheduled/Manual)                       │
│  ├─ Setup Azure ML Environment                             │
│  ├─ Run wine_quality_comprehensive.py                       │
│  ├─ Track experiments in MLflow                             │
│  └─ Register best model                                     │
└─────────────┬───────────────────────────────────────────────┘
              │ (Model ready)
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Deployment Pipeline (Manual)                               │
│  ├─ Validate model exists                                   │
│  ├─ Deploy to Staging                                       │
│  ├─ Run smoke tests                                         │
│  ├─ Manual approval gate                                    │
│  └─ Deploy to Production                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Customization

### **Change Training Schedule**

Edit `training-pipeline.yml`:

```yaml
schedules:
- cron: "0 2 * * 0"  # Sunday 2 AM
  # Change to:
  # "0 2 * * *"  = Daily at 2 AM
  # "0 2 * * 1"  = Monday at 2 AM
  # "0 */6 * * *" = Every 6 hours
```

### **Add Data Validation Stage**

Add to `training-pipeline.yml`:

```yaml
- stage: ValidateData
  displayName: 'Validate Training Data'
  dependsOn: Setup
  jobs:
  - job: DataQuality
    steps:
    - script: |
        python scripts/validate_data.py
      displayName: 'Run data quality checks'
```

### **Add Model Performance Tests**

Add to `deployment-pipeline.yml`:

```yaml
- stage: PerformanceTest
  displayName: 'Model Performance Testing'
  dependsOn: DeployToStaging
  jobs:
  - job: LoadTest
    steps:
    - script: |
        python tests/load_test_endpoint.py
      displayName: 'Run load tests'
```

---

## 📈 Monitoring & Alerts

### **Pipeline Notifications**

Configure in Azure DevOps:

1. **Project Settings → Notifications**
2. Create subscription:
   - **Build completes:** Notify on failure
   - **Release deployment approval pending:** Notify approvers
   - **Release deployment completed:** Notify team

### **Integrate with Slack/Teams**

Add to pipeline:

```yaml
- script: |
    curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"Training pipeline completed!"}' \
    $SLACK_WEBHOOK_URL
  env:
    SLACK_WEBHOOK_URL: $(SLACK_WEBHOOK)
  displayName: 'Send Slack notification'
```

---

## 🎯 Best Practices

### **1. Environment Separation**

```yaml
# Use different variable groups per environment
variables:
  - ${{ if eq(variables['Build.SourceBranchName'], 'main') }}:
    - group: prod-credentials
  - ${{ else }}:
    - group: dev-credentials
```

### **2. Artifact Management**

```yaml
# Publish training artifacts
- task: PublishPipelineArtifact@1
  inputs:
    targetPath: 'outputs/models'
    artifactName: 'trained-models'
```

### **3. Approval Gates**

```yaml
# Add manual intervention
- task: ManualValidation@0
  inputs:
    notifyUsers: 'ml-team@company.com'
    instructions: 'Review model metrics before production'
    onTimeout: 'reject'
```

### **4. Rollback Strategy**

```yaml
# Keep previous deployment for rollback
- script: |
    az ml online-endpoint update \
      --name wine-quality-prod \
      --traffic "previous=100"
  displayName: 'Rollback to previous version'
```

---

## 🔍 Troubleshooting

### **Pipeline Fails at Authentication**

**Solution:** Verify service connection permissions:
```bash
# Grant ML workspace contributor role
az role assignment create \
  --assignee <service-principal-id> \
  --role "AzureML Workspace Contributor" \
  --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws-name>
```

### **Training Script Fails**

**Solution:** Check environment variables are set:
```yaml
- script: |
    echo "Tenant: $AZURE_TENANT_ID"
    echo "Subscription: $AZURE_SUBSCRIPTION_ID"
    echo "Resource Group: $AZURE_RESOURCE_GROUP"
  displayName: 'Debug environment variables'
```

### **Model Not Found**

**Solution:** Verify model registration:
```bash
az ml model list \
  --resource-group <rg> \
  --workspace-name <ws> \
  --name wine-quality-elasticnet
```

---

## 📚 Additional Resources

- [Azure DevOps YAML Schema](https://docs.microsoft.com/en-us/azure/devops/pipelines/yaml-schema)
- [Azure ML CLI v2](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)
- [MLOps with Azure DevOps](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-devops-machine-learning)
- [Pipeline Templates](https://docs.microsoft.com/en-us/azure/devops/pipelines/process/templates)

---

## 🎬 Getting Started Checklist

- [ ] Create Azure DevOps project
- [ ] Set up service connection to Azure
- [ ] Create variable group with credentials
- [ ] Import CI pipeline
- [ ] Import training pipeline
- [ ] Import deployment pipeline
- [ ] Test CI pipeline with a PR
- [ ] Run training pipeline manually
- [ ] Verify model registration in Azure ML
- [ ] Deploy to staging
- [ ] Approve and deploy to production

---

**Created:** December 2025  
**Pipeline Type:** Azure DevOps CI/CD for MLOps  
**Target:** Wine Quality ML Model
