# Azure DevOps Pipeline for MLflow Experiments

This directory contains Azure DevOps pipeline configurations for running MLflow wine quality prediction experiments.

## Pipeline Overview

### `mlflow-experiments.yml`

Main pipeline that runs wine quality prediction experiments using MLflow for tracking and publishes results.

**Stages:**

1. **Setup** - Validate and setup Python environment
2. **RunExperiments** - Execute ML experiments
   - ElasticNet Optimization (Grid Search)
   - Experiment Series (5 optimization strategies)
3. **PublishResults** - Generate reports and publish artifacts
4. **CleanupAndArchive** - Archive all results

## Pipeline Features

✅ **Automated Experiment Execution**
- Runs `optimize_elasticnet.py` for hyperparameter optimization
- Runs `experiment_series.py` for comprehensive strategy comparison

✅ **MLflow Integration**
- Automatic experiment tracking with MLflow
- Local tracking (no Azure ML workspace required)
- Model artifacts preservation

✅ **Result Publication**
- MLflow artifacts published as pipeline artifacts
- Summary reports in Markdown format
- JUnit XML format for test results integration
- Archived results as downloadable ZIP

✅ **Multi-Stage Architecture**
- Parallel job execution where possible
- Dependency management between stages
- Continues on error for reporting stage

## Using the Pipeline

### Prerequisites

1. Azure DevOps project with pipeline permissions
2. Git repository with the code
3. Python 3.11+ available in pipeline agent

### Setup in Azure DevOps

1. Navigate to your Azure DevOps project
2. Go to **Pipelines** → **New Pipeline**
3. Select your repository
4. Choose **Existing Azure Pipelines YAML file**
5. Select `azure-pipelines/mlflow-experiments.yml`
6. Click **Run**

### Pipeline Variables

Configure these variables if needed:

- `pythonVersion`: Python version to use (default: 3.11)
- `USE_AZURE_ML`: Set to 'True' to use Azure ML workspace (default: False)

### Trigger Configuration

**Automatic Triggers:**
- Push to `main` or `develop` branches
- Changes to `ai-code/*` or `data/*` directories
- Pull requests to `main` or `develop`

**Manual Trigger:**
- Click **Run pipeline** in Azure DevOps

## Artifacts Published

### 1. MLflow Results
- **Artifact:** `mlflow-elasticnet-results`
- **Content:** MLflow tracking data from ElasticNet optimization
- **Location:** `mlruns/` directory

### 2. Experiment Series Results
- **Artifact:** `mlflow-experiment-series-results`
- **Content:** MLflow tracking data from 5 optimization strategies
- **Location:** `mlruns/` directory

### 3. Summary Reports
- **Artifact:** `experiment-summary-reports`
- **Content:** 
  - `elasticnet-summary.md` - ElasticNet optimization summary
  - `experiment-series-summary.md` - Experiment series summary
  - `combined-summary.md` - Combined results from all experiments
  - `*.xml` - JUnit format test results

### 4. Archived Results
- **Artifact:** `mlflow-experiments-archive`
- **Content:** ZIP file containing all MLflow results
- **Format:** `mlflow-experiments-{BuildNumber}.zip`

## Viewing Results

### In Azure DevOps

1. **Pipeline Run View:**
   - Go to pipeline run
   - Check **Summary** tab for test results
   - View **Tests** tab for detailed metrics

2. **Artifacts:**
   - Click on **Artifacts** button
   - Download specific artifacts or archive
   - View published reports

### Locally with MLflow UI

1. Download `mlflow-elasticnet-results` artifact
2. Extract to local directory
3. Run: `mlflow ui --backend-store-uri file:./mlruns`
4. Open browser: `http://localhost:5000`

## Report Generator Script

### `scripts/generate_summary_report.py`

This script processes MLflow experiment results and generates:

- **Markdown Reports:** Human-readable summaries with best runs and comparisons
- **JUnit XML:** Machine-readable format for Azure DevOps test integration

**Features:**
- Automatically finds best performing models
- Compares top 5 runs for each experiment
- Generates combined summary across all experiments
- Exports metrics and parameters

**Usage:**
```bash
python azure-pipelines/scripts/generate_summary_report.py
```

**Output:**
- `reports/elasticnet-summary.md`
- `reports/experiment-series-summary.md`
- `reports/combined-summary.md`
- `reports/*.xml` (JUnit format)

## Customization

### Adding New Experiments

1. Create your experiment script in `ai-code/`
2. Add a new job in the `RunExperiments` stage:

```yaml
- job: YourExperiment
  displayName: 'Run Your Experiment'
  steps:
  - task: UsePythonVersion@0
    displayName: 'Use Python $(pythonVersion)'
    inputs:
      versionSpec: '$(pythonVersion)'
  
  - script: |
      python -m pip install --upgrade pip
      pip install -r ai-code/requirements.txt
    displayName: 'Install Dependencies'
  
  - script: |
      cd ai-code
      python your_experiment.py
    displayName: 'Run Your Experiment'
  
  - task: PublishPipelineArtifact@1
    displayName: 'Publish Your Experiment Results'
    inputs:
      targetPath: '$(workingDirectory)/ai-code/mlruns'
      artifact: 'mlflow-your-experiment-results'
```

3. Update `generate_summary_report.py` to include your experiment

### Using Azure ML Workspace

To enable Azure ML workspace tracking:

1. Set pipeline variable: `USE_AZURE_ML: 'True'`
2. Add Azure ML workspace configuration in pipeline:

```yaml
- script: |
    # Create config.json for Azure ML workspace
    cat > ai-code/config.json << EOF
    {
      "subscription_id": "$(AZURE_SUBSCRIPTION_ID)",
      "resource_group": "$(AZURE_RESOURCE_GROUP)",
      "workspace_name": "$(AZURE_WORKSPACE_NAME)"
    }
    EOF
  displayName: 'Configure Azure ML Workspace'
```

3. Add Azure service connection to pipeline
4. Install Azure ML SDK in requirements.txt

## Troubleshooting

### Pipeline Fails on Python Dependencies

**Solution:** Update `requirements.txt` with compatible versions for Python 3.11

### MLflow Artifacts Not Found

**Solution:** Check that experiments ran successfully and created `mlruns/` directory

### Reports Not Generated

**Solution:** 
- Verify MLflow tracking data exists
- Check `generate_summary_report.py` logs
- Ensure experiments completed successfully

### Test Results Not Showing

**Solution:**
- Verify JUnit XML files are generated in `reports/` directory
- Check `PublishTestResults@2` task in pipeline logs

## Dependencies

See `ai-code/requirements.txt` for complete list:

- mlflow >= 2.8.0
- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Support

For issues or questions:
1. Check pipeline logs in Azure DevOps
2. Review MLflow experiment tracking data
3. Verify Python environment and dependencies
4. Check experiment scripts for errors

## Version History

- **v1.0** - Initial pipeline with ElasticNet and Experiment Series support
- Multi-stage architecture with artifact publishing
- JUnit XML report generation for test integration
