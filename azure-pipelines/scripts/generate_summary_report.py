"""
Generate Summary Report from MLflow Experiments
This script reads MLflow experiment results and generates summary reports
for Azure DevOps pipeline publication
"""

import os
import json
import mlflow
from pathlib import Path
import pandas as pd
from datetime import datetime

def get_experiment_runs(tracking_uri, experiment_name):
    """Get all runs from an MLflow experiment"""
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_rmse ASC"]
        )
        return runs
    except Exception as e:
        print(f"Error getting experiment runs: {e}")
        return []

def generate_summary_markdown(runs_df, experiment_name, output_file):
    """Generate a markdown summary report"""
    
    with open(output_file, 'w') as f:
        f.write(f"# MLflow Experiment Results Summary\n\n")
        f.write(f"**Experiment:** {experiment_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Runs:** {len(runs_df)}\n\n")
        
        if len(runs_df) == 0:
            f.write("No runs found.\n")
            return
        
        # Best run summary
        f.write("## Best Run\n\n")
        best_run = runs_df.iloc[0]
        f.write(f"- **Run ID:** {best_run.get('run_id', 'N/A')}\n")
        f.write(f"- **Run Name:** {best_run.get('tags.mlflow.runName', 'N/A')}\n")
        
        # Metrics
        metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
        if metric_cols:
            f.write("\n### Metrics\n\n")
            for col in metric_cols:
                metric_name = col.replace('metrics.', '')
                value = best_run.get(col, 'N/A')
                if value != 'N/A':
                    f.write(f"- **{metric_name}:** {value:.6f}\n")
        
        # Parameters
        param_cols = [col for col in runs_df.columns if col.startswith('params.')]
        if param_cols:
            f.write("\n### Parameters\n\n")
            for col in param_cols:
                param_name = col.replace('params.', '')
                value = best_run.get(col, 'N/A')
                f.write(f"- **{param_name}:** {value}\n")
        
        # Top 5 runs comparison
        if len(runs_df) > 1:
            f.write("\n## Top 5 Runs Comparison\n\n")
            f.write("| Rank | Run Name | RMSE | MAE | R² |\n")
            f.write("|------|----------|------|-----|----|\n")
            
            for idx, row in runs_df.head(5).iterrows():
                rank = idx + 1
                run_name = row.get('tags.mlflow.runName', 'N/A')
                rmse = row.get('metrics.test_rmse', 'N/A')
                mae = row.get('metrics.test_mae', 'N/A')
                r2 = row.get('metrics.test_r2', 'N/A')
                
                rmse_str = f"{rmse:.4f}" if rmse != 'N/A' else 'N/A'
                mae_str = f"{mae:.4f}" if mae != 'N/A' else 'N/A'
                r2_str = f"{r2:.4f}" if r2 != 'N/A' else 'N/A'
                
                f.write(f"| {rank} | {run_name} | {rmse_str} | {mae_str} | {r2_str} |\n")
        
        print(f"Summary report generated: {output_file}")

def generate_junit_xml(runs_df, experiment_name, output_file):
    """Generate JUnit XML format for Azure DevOps test results integration"""
    
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    
    testsuites = Element('testsuites')
    testsuite = SubElement(testsuites, 'testsuite', {
        'name': experiment_name,
        'tests': str(len(runs_df)),
        'failures': '0',
        'errors': '0',
        'timestamp': datetime.now().isoformat()
    })
    
    for idx, row in runs_df.iterrows():
        run_name = row.get('tags.mlflow.runName', f'run_{idx}')
        rmse = row.get('metrics.test_rmse', None)
        
        testcase = SubElement(testsuite, 'testcase', {
            'name': run_name,
            'classname': experiment_name,
            'time': '0'
        })
        
        # Add metrics as properties
        properties = SubElement(testcase, 'properties')
        
        metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
        for col in metric_cols:
            metric_name = col.replace('metrics.', '')
            value = row.get(col)
            if value is not None and pd.notna(value):
                SubElement(properties, 'property', {
                    'name': metric_name,
                    'value': f'{value:.6f}'
                })
        
        param_cols = [col for col in runs_df.columns if col.startswith('params.')]
        for col in param_cols:
            param_name = col.replace('params.', '')
            value = row.get(col)
            if value is not None and pd.notna(value):
                SubElement(properties, 'property', {
                    'name': param_name,
                    'value': str(value)
                })
    
    # Pretty print XML
    xml_str = minidom.parseString(tostring(testsuites)).toprettyxml(indent="  ")
    
    with open(output_file, 'w') as f:
        f.write(xml_str)
    
    print(f"JUnit XML report generated: {output_file}")

def main():
    """Main execution function"""
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Generating MLflow Experiment Summary Reports")
    print("="*70)
    
    # Process ElasticNet results
    print("\nProcessing ElasticNet Optimization results...")
    elasticnet_uri = "file:./results/elasticnet/mlruns"
    elasticnet_runs = get_experiment_runs(elasticnet_uri, "wine-quality-elasticnet-optimization")
    
    if len(elasticnet_runs) > 0:
        generate_summary_markdown(
            elasticnet_runs,
            "ElasticNet Optimization",
            reports_dir / "elasticnet-summary.md"
        )
        generate_junit_xml(
            elasticnet_runs,
            "ElasticNet_Optimization",
            reports_dir / "elasticnet-results.xml"
        )
    else:
        print("No ElasticNet runs found")
    
    # Process Experiment Series results
    print("\nProcessing Experiment Series results...")
    series_uri = "file:./results/experiment-series/mlruns"
    series_runs = get_experiment_runs(series_uri, "wine-quality-rmse-optimization-series")
    
    if len(series_runs) > 0:
        generate_summary_markdown(
            series_runs,
            "Experiment Series (RMSE Optimization)",
            reports_dir / "experiment-series-summary.md"
        )
        generate_junit_xml(
            series_runs,
            "Experiment_Series_RMSE_Optimization",
            reports_dir / "experiment-series-results.xml"
        )
    else:
        print("No Experiment Series runs found")
    
    # Generate combined summary
    print("\nGenerating combined summary...")
    all_runs = []
    
    if len(elasticnet_runs) > 0:
        elasticnet_runs['source'] = 'ElasticNet Optimization'
        all_runs.append(elasticnet_runs)
    
    if len(series_runs) > 0:
        series_runs['source'] = 'Experiment Series'
        all_runs.append(series_runs)
    
    if all_runs:
        combined_runs = pd.concat(all_runs, ignore_index=True)
        combined_runs = combined_runs.sort_values(
            by='metrics.test_rmse',
            ascending=True
        ).reset_index(drop=True)
        
        generate_summary_markdown(
            combined_runs,
            "All MLflow Experiments",
            reports_dir / "combined-summary.md"
        )
        generate_junit_xml(
            combined_runs,
            "All_MLflow_Experiments",
            reports_dir / "experiment-results.xml"
        )
    
    print("\n" + "="*70)
    print("✓ All summary reports generated successfully!")
    print(f"✓ Reports saved to: {reports_dir.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()
