# Getting Started with MLflow Experiments Pipeline

This guide walks you through setting up and running the Azure DevOps pipeline for MLflow experiments.

## Quick Start

### 1. Setup Repository

```bash
# Clone or navigate to your repository
cd /path/to/ml-flow-course

# Verify required files exist
ls ai-code/optimize_elasticnet.py
ls ai-code/experiment_series.py
ls data/red-wine-quality.csv
```

### 2. Create Azure DevOps Pipeline

**Option A: Using Azure DevOps UI**

1. Go to Azure DevOps → Your Project → Pipelines
2. Click **New Pipeline**
3. Select **Azure Repos Git** (or your Git provider)
4. Select your repository
5. Choose **Existing Azure Pipelines YAML file**
6. Path: `azure-pipelines/mlflow-experiments.yml`
7. Click **Run**

**Option B: Using Azure CLI**

```bash
# Login to Azure DevOps
az login
az devops configure --defaults organization=https://dev.azure.com/yourorg project=YourProject

# Create pipeline
az pipelines create \
  --name "MLflow Experiments" \
  --repository ml-flow-course \
  --branch main \
  --yml-path azure-pipelines/mlflow-experiments.yml
```

### 3. Run the Pipeline

**First Run:**

```bash
# Using Azure CLI
az pipelines run --name "MLflow Experiments"

# Or click "Run pipeline" in Azure DevOps UI
```

**Expected Duration:** 5-15 minutes depending on agent availability

### 4. View Results

1. **In Azure DevOps:**
   - Go to pipeline run page
   - Click **Summary** tab
   - Scroll to **Tests** section to see experiment metrics
   - Click **Artifacts** to download results

2. **Download and View Locally:**

```bash
# Download the archive artifact
# Extract it locally
unzip mlflow-experiments-*.zip

# Start MLflow UI
cd mlflow-elasticnet-results
mlflow ui --backend-store-uri file:./mlruns

# Open browser: http://localhost:5000
```

## Pipeline Outputs Explained

### Artifacts

#### 1. `mlflow-elasticnet-results`
Contains MLflow tracking data from ElasticNet hyperparameter optimization:
- Grid search results (11 alpha × 9 l1_ratio = 99 combinations)
- Best model artifacts (model.pkl, requirements.txt, etc.)
- Baseline comparison metrics

#### 2. `mlflow-experiment-series-results`
Contains results from 5 different optimization strategies:
- Coarse grid search
- Fine-tuned grid search
- Random search
- Domain-specific optimization
- Ultra-fine grid search

#### 3. `experiment-summary-reports`
Human-readable reports:
- **elasticnet-summary.md** - Best ElasticNet run and top 5 comparison
- **experiment-series-summary.md** - Best strategy and comparison
- **combined-summary.md** - Overall best across all experiments
- **\*.xml** - Machine-readable JUnit format

#### 4. `mlflow-experiments-archive`
ZIP file containing all above artifacts for easy download

### Test Results Integration

The pipeline publishes metrics as test results:

| Test Case | Metrics Published |
|-----------|------------------|
| ElasticNet Optimization | RMSE, MAE, R², CV Score |
| Experiment Series | RMSE, MAE, R² per strategy |

**Viewing:**
1. Go to pipeline run
2. Click **Tests** tab
3. See all experiment runs as test cases
4. Click on individual tests to see metrics

## Common Workflows

### Workflow 1: Regular Experiment Runs

```bash
# 1. Make changes to experiment scripts
vim ai-code/experiment_series.py

# 2. Commit and push
git add ai-code/
git commit -m "Updated experiment parameters"
git push origin main

# 3. Pipeline auto-triggers
# Monitor in Azure DevOps

# 4. Download and analyze results
```

### Workflow 2: Test Before Commit

```bash
# 1. Run quick test pipeline manually
az pipelines run --name "MLflow Quick Test"

# 2. Check if it passes
# 3. If successful, commit changes

git add .
git commit -m "Tested experiment changes"
git push
```

### Workflow 3: Experiment Comparison

```bash
# 1. Run pipeline on feature branch
git checkout -b experiment/new-strategy
# Make changes
git push origin experiment/new-strategy

# 2. Pipeline runs on PR
# Create PR to main

# 3. Compare results in PR
# Download artifacts from both runs
# Compare metrics

# 4. Merge if better results
```

### Workflow 4: Local Development → Pipeline

```bash
# 1. Develop locally
cd ai-code
python experiment_series.py

# 2. Verify results locally
mlflow ui

# 3. Push to trigger pipeline
git add .
git commit -m "New experiment ready"
git push

# 4. Verify pipeline produces same results
```

## Environment Variables

Set these in Azure DevOps pipeline variables:

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_VERSION` | Python version | 3.11 |
| `USE_AZURE_ML` | Use Azure ML workspace | False |
| `MLFLOW_TRACKING_URI` | Custom tracking URI | file:./mlruns |

### For Azure ML Integration

| Variable | Description |
|----------|-------------|
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | Resource group name |
| `AZURE_WORKSPACE_NAME` | ML workspace name |

**Setting Variables:**

```bash
# Via Azure CLI
az pipelines variable create \
  --pipeline-name "MLflow Experiments" \
  --name USE_AZURE_ML \
  --value True

# Or in Azure DevOps UI:
# Pipeline → Edit → Variables → New variable
```

## Troubleshooting

### Issue: Pipeline fails at dependency installation

**Solution:**
```bash
# Update requirements.txt with compatible versions
# Test locally first:
pip install -r ai-code/requirements.txt
```

### Issue: No artifacts published

**Cause:** Experiments failed or didn't create mlruns directory

**Debug:**
1. Check pipeline logs in "Run Experiments" stage
2. Look for Python errors
3. Verify data file exists: `data/red-wine-quality.csv`

### Issue: Reports not generated

**Cause:** MLflow runs completed but report script failed

**Debug:**
```bash
# Download mlflow results artifact
# Run report generator locally:
python azure-pipelines/scripts/generate_summary_report.py
```

### Issue: Different results locally vs pipeline

**Possible Causes:**
- Different Python versions
- Different library versions
- Different random seeds
- Data file differences

**Solution:**
```bash
# Match pipeline environment locally
pip install -r ai-code/requirements.txt
python --version  # Should match pipeline
```

## Advanced Usage

### Custom Experiment Integration

Add your own experiment to the pipeline:

```yaml
# In mlflow-experiments.yml, add to RunExperiments stage:

- job: MyCustomExperiment
  displayName: 'Run My Custom Experiment'
  steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '$(pythonVersion)'
  
  - script: |
      pip install -r ai-code/requirements.txt
    displayName: 'Install Dependencies'
  
  - script: |
      cd ai-code
      python my_custom_experiment.py
    displayName: 'Run Custom Experiment'
  
  - task: PublishPipelineArtifact@1
    inputs:
      targetPath: '$(workingDirectory)/ai-code/mlruns'
      artifact: 'my-custom-results'
```

### Scheduled Runs

Add schedule trigger to run experiments automatically:

```yaml
# At top of mlflow-experiments.yml
schedules:
- cron: "0 2 * * *"  # Daily at 2 AM UTC
  displayName: Daily experiment run
  branches:
    include:
    - main
  always: true
```

### Notifications

Setup notifications for pipeline results:

1. Azure DevOps → Project Settings → Notifications
2. Create new subscription
3. Event: **Build completes**
4. Filter: Pipeline = "MLflow Experiments"
5. Recipients: Your team

## Best Practices

### 1. Version Control Everything
```bash
git add ai-code/ data/ azure-pipelines/
git commit -m "Experiment configuration and pipeline"
```

### 2. Use Branches for Experiments
```bash
git checkout -b experiment/optimizer-comparison
# Make changes
# Push and create PR
# Pipeline runs automatically
```

### 3. Tag Successful Runs
```bash
git tag -a v1.0-best-rmse -m "Best RMSE: 0.6233"
git push origin v1.0-best-rmse
```

### 4. Document Experiment Changes
```bash
git commit -m "
Experiment: Fine-tuned alpha range
- Narrowed alpha to [0.001, 0.1]
- Increased l1_ratio granularity
- Expected RMSE improvement: 2-5%
"
```

### 5. Archive Baseline Results
```bash
# Download baseline results before major changes
# Store in docs/baselines/ for comparison
```

## Next Steps

1. ✅ Run the pipeline successfully
2. ✅ Download and view MLflow results
3. ✅ Review summary reports
4. 🔄 Modify experiment parameters
5. 🔄 Compare new results with baseline
6. 🚀 Integrate into CI/CD workflow
7. 🚀 Set up scheduled runs
8. 🚀 Add more experiments

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Azure Pipelines YAML Reference](https://docs.microsoft.com/en-us/azure/devops/pipelines/yaml-schema)
- [Scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

## Support

For help:
1. Check pipeline logs in Azure DevOps
2. Review this guide
3. Check MLflow experiment tracking
4. Verify environment matches pipeline

Happy Experimenting! 🚀
