# Azure Pipelines for MLflow Experiments - File Index

This document lists all pipeline-related files created for running MLflow experiments in Azure DevOps.

## 📁 Directory Structure

```
azure-pipelines/
├── README.md                          # Main pipeline documentation
├── GETTING_STARTED.md                 # Step-by-step setup guide
├── mlflow-experiments.yml             # Main production pipeline
├── quick-test.yml                     # Quick test pipeline (manual)
├── pr-validation.yml                  # Pull request validation
└── scripts/
    ├── generate_summary_report.py     # Generate markdown/JUnit reports
    └── create_visualizations.py       # Create comparison charts

ai-code/
└── requirements.txt                   # Updated with all dependencies
```

## 📄 File Descriptions

### Pipeline Configurations

#### 1. `mlflow-experiments.yml` (Main Pipeline)
**Purpose:** Production pipeline for running experiments and publishing results

**Triggers:**
- Push to `main` or `develop` branches
- Changes to `ai-code/*` or `data/*`
- Pull requests to `main` or `develop`

**Stages:**
1. Setup - Environment validation
2. RunExperiments - Execute ML experiments
3. PublishResults - Generate and publish reports
4. CleanupAndArchive - Archive all results

**Artifacts Published:**
- `mlflow-elasticnet-results` - ElasticNet optimization results
- `mlflow-experiment-series-results` - 5-strategy comparison results
- `experiment-summary-reports` - Markdown and JUnit reports
- `mlflow-experiments-archive` - Complete ZIP archive

**Duration:** 5-15 minutes

---

#### 2. `quick-test.yml` (Quick Test Pipeline)
**Purpose:** Fast validation pipeline for testing changes

**Triggers:** Manual only

**Features:**
- Simplified single-job execution
- 15-minute timeout
- Continues on error
- Quick feedback loop

**Use Cases:**
- Testing pipeline changes
- Validating experiment scripts
- Quick sanity checks

**Duration:** 3-5 minutes

---

#### 3. `pr-validation.yml` (PR Validation Pipeline)
**Purpose:** Automated validation for pull requests

**Triggers:** PRs to `main` or `develop`

**Validation Steps:**
1. Code linting (flake8, pylint)
2. Quick experiment test (300s timeout)
3. MLflow output validation

**Benefits:**
- Catches issues before merge
- Ensures code quality
- Validates experiment compatibility

**Duration:** 2-5 minutes

---

### Support Scripts

#### 4. `scripts/generate_summary_report.py`
**Purpose:** Generate experiment summary reports

**Inputs:** MLflow tracking data from experiments

**Outputs:**
- `reports/elasticnet-summary.md` - ElasticNet summary
- `reports/experiment-series-summary.md` - Series summary
- `reports/combined-summary.md` - Combined summary
- `reports/*.xml` - JUnit XML format

**Features:**
- Identifies best performing runs
- Compares top 5 runs
- Exports metrics and parameters
- Azure DevOps test results integration

**Usage:**
```bash
python azure-pipelines/scripts/generate_summary_report.py
```

---

#### 5. `scripts/create_visualizations.py`
**Purpose:** Generate visual comparisons of experiment results

**Outputs:**
- `reports/visualizations/rmse_comparison.png` - Top 10 runs bar chart
- `reports/visualizations/parameter_heatmap.png` - Alpha vs L1 ratio heatmap
- `reports/visualizations/metrics_scatter.png` - RMSE vs MAE scatter
- `reports/visualizations/r2_distribution.png` - R² score distribution
- `reports/visualizations/strategy_comparison.png` - Strategy comparison

**Features:**
- Professional matplotlib/seaborn visualizations
- Highlights best performing models
- Parameter relationship analysis
- Strategy effectiveness comparison

**Usage:**
```bash
python azure-pipelines/scripts/create_visualizations.py
```

---

### Documentation

#### 6. `README.md`
**Purpose:** Comprehensive pipeline documentation

**Contents:**
- Pipeline overview and features
- Artifact descriptions
- Setup instructions
- Customization guide
- Troubleshooting
- Dependencies

**Audience:** DevOps engineers, Data scientists

---

#### 7. `GETTING_STARTED.md`
**Purpose:** Step-by-step guide for first-time users

**Contents:**
- Quick start guide
- Azure DevOps setup
- Running pipelines
- Viewing results
- Common workflows
- Best practices
- Troubleshooting

**Audience:** New team members, Beginners

---

#### 8. `ai-code/requirements.txt` (Updated)
**Purpose:** Python dependencies for experiments

**Updated Dependencies:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
mlflow>=2.8.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

**Notes:**
- Compatible with Python 3.11+
- Azure ML SDK commented out (optional)
- All versions tested and validated

---

## 🚀 Quick Reference

### Running Pipelines

**Main Pipeline (Auto):**
```bash
# Triggered automatically on push to main/develop
git push origin main
```

**Quick Test (Manual):**
```bash
az pipelines run --name "MLflow Quick Test"
```

**PR Validation (Auto):**
```bash
# Triggered automatically on PR creation
git push origin feature/my-branch
# Create PR in Azure DevOps
```

### Viewing Results

**Download Artifacts:**
1. Go to pipeline run in Azure DevOps
2. Click **Artifacts** button
3. Download desired artifact
4. Extract and view locally

**View MLflow UI:**
```bash
# Extract downloaded artifact
cd mlflow-elasticnet-results
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

**View Reports:**
```bash
# Extract experiment-summary-reports artifact
cd reports
cat elasticnet-summary.md
cat combined-summary.md
```

**View Visualizations:**
```bash
# Extract experiment-summary-reports artifact
cd reports/visualizations
# Open PNG files in image viewer
```

### Common Tasks

**Add New Experiment:**
1. Create experiment script in `ai-code/`
2. Add job to `mlflow-experiments.yml`
3. Update `generate_summary_report.py`
4. Push changes

**Change Python Version:**
1. Update `pythonVersion` variable in pipeline
2. Test locally with same version
3. Push changes

**Enable Azure ML Tracking:**
1. Set `USE_AZURE_ML: 'True'` in pipeline
2. Add workspace configuration
3. Add Azure service connection
4. Push changes

## 📊 Pipeline Outputs Summary

| Output Type | Location | Format | Purpose |
|-------------|----------|--------|---------|
| MLflow Results | Artifacts | Directory | Experiment tracking data |
| Summary Reports | Artifacts | Markdown | Human-readable summaries |
| Test Results | Tests tab | JUnit XML | Azure DevOps integration |
| Visualizations | Artifacts | PNG | Visual comparisons |
| Archive | Artifacts | ZIP | Complete downloadable package |

## 🔧 Maintenance

### Regular Tasks

1. **Update Dependencies:** Review and update `requirements.txt` quarterly
2. **Review Triggers:** Adjust trigger paths as experiments evolve
3. **Archive Old Results:** Download and archive artifacts from old runs
4. **Update Documentation:** Keep README and GETTING_STARTED current

### Monitoring

1. **Pipeline Success Rate:** Track in Azure DevOps analytics
2. **Experiment Quality:** Review test results trends
3. **Performance:** Monitor pipeline duration
4. **Artifact Size:** Watch for excessive artifact growth

## 📝 Notes

- All pipelines use Ubuntu latest agent
- Python 3.11 recommended for best compatibility
- MLflow results are preserved as pipeline artifacts
- Reports are regenerated on each run
- Visualizations require matplotlib and seaborn

## 🆘 Support

**Issues with Pipeline:**
1. Check pipeline logs in Azure DevOps
2. Review this index for file purposes
3. Check GETTING_STARTED.md for troubleshooting
4. Verify Python environment matches pipeline

**Issues with Experiments:**
1. Test locally first
2. Check MLflow tracking output
3. Review experiment script logs
4. Verify data files exist

**Issues with Reports:**
1. Check if MLflow runs completed
2. Verify tracking data exists
3. Run report scripts locally
4. Check script logs

## 🎯 Next Steps

1. ✅ Review all files in this index
2. ✅ Read GETTING_STARTED.md
3. ✅ Set up pipeline in Azure DevOps
4. ✅ Run quick-test pipeline
5. ✅ Run main pipeline
6. ✅ Download and review results
7. 🚀 Customize for your needs
8. 🚀 Add more experiments

---

**Version:** 1.0  
**Last Updated:** {{ current_date }}  
**Maintained By:** Data Science Team
