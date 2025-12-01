"""
Create Visual Comparisons of MLflow Experiment Results
Generates charts comparing different experiment runs
"""

import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def create_metric_comparison_chart(runs_df, output_dir):
    """Create bar chart comparing RMSE across all runs"""
    
    if len(runs_df) == 0:
        print("No runs to visualize")
        return
    
    # Get top 10 runs
    top_runs = runs_df.head(10).copy()
    
    # Prepare data
    run_names = top_runs.get('tags.mlflow.runName', range(len(top_runs)))
    rmse_values = top_runs.get('metrics.test_rmse', [])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    bars = ax.barh(range(len(top_runs)), rmse_values, color='steelblue')
    
    # Highlight best run
    bars[0].set_color('green')
    
    # Customize
    ax.set_yticks(range(len(top_runs)))
    ax.set_yticklabels(run_names)
    ax.set_xlabel('RMSE (Lower is Better)', fontsize=12)
    ax.set_title('Top 10 Experiment Runs - RMSE Comparison', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, rmse_values)):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / 'rmse_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_file}")
    plt.close()

def create_parameter_heatmap(runs_df, output_dir):
    """Create heatmap showing parameter vs RMSE relationship"""
    
    if len(runs_df) == 0:
        print("No runs to visualize")
        return
    
    # Check if we have alpha and l1_ratio parameters
    if 'params.alpha' not in runs_df.columns or 'params.l1_ratio' not in runs_df.columns:
        print("No alpha/l1_ratio parameters found for heatmap")
        return
    
    # Prepare data
    plot_data = runs_df[['params.alpha', 'params.l1_ratio', 'metrics.test_rmse']].copy()
    plot_data = plot_data.dropna()
    
    if len(plot_data) == 0:
        print("No valid data for heatmap")
        return
    
    # Convert to numeric
    plot_data['params.alpha'] = pd.to_numeric(plot_data['params.alpha'], errors='coerce')
    plot_data['params.l1_ratio'] = pd.to_numeric(plot_data['params.l1_ratio'], errors='coerce')
    plot_data = plot_data.dropna()
    
    if len(plot_data) < 5:
        print("Not enough data points for meaningful heatmap")
        return
    
    # Create pivot table
    pivot_data = plot_data.pivot_table(
        values='metrics.test_rmse',
        index='params.l1_ratio',
        columns='params.alpha',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'RMSE'},
        ax=ax
    )
    
    ax.set_title('Parameter Heatmap: Alpha vs L1 Ratio', fontsize=14, fontweight='bold')
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('L1 Ratio', fontsize=12)
    
    plt.tight_layout()
    
    output_file = output_dir / 'parameter_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_file}")
    plt.close()

def create_metrics_scatter(runs_df, output_dir):
    """Create scatter plot of RMSE vs MAE"""
    
    if len(runs_df) == 0:
        print("No runs to visualize")
        return
    
    if 'metrics.test_rmse' not in runs_df.columns or 'metrics.test_mae' not in runs_df.columns:
        print("Missing RMSE or MAE metrics for scatter plot")
        return
    
    # Prepare data
    plot_data = runs_df[['metrics.test_rmse', 'metrics.test_mae']].copy()
    plot_data = plot_data.dropna()
    
    if len(plot_data) == 0:
        print("No valid data for scatter plot")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(
        plot_data['metrics.test_rmse'],
        plot_data['metrics.test_mae'],
        alpha=0.6,
        s=100,
        c='steelblue',
        edgecolors='black'
    )
    
    # Highlight best RMSE point
    best_idx = plot_data['metrics.test_rmse'].idxmin()
    ax.scatter(
        plot_data.loc[best_idx, 'metrics.test_rmse'],
        plot_data.loc[best_idx, 'metrics.test_mae'],
        color='green',
        s=200,
        edgecolors='black',
        linewidths=2,
        label='Best RMSE',
        zorder=5
    )
    
    ax.set_xlabel('RMSE', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('RMSE vs MAE Across All Runs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'metrics_scatter.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_file}")
    plt.close()

def create_r2_distribution(runs_df, output_dir):
    """Create histogram of R² scores"""
    
    if len(runs_df) == 0:
        print("No runs to visualize")
        return
    
    if 'metrics.test_r2' not in runs_df.columns:
        print("No R² metrics found")
        return
    
    r2_values = runs_df['metrics.test_r2'].dropna()
    
    if len(r2_values) == 0:
        print("No valid R² values")
        return
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(r2_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add mean line
    mean_r2 = r2_values.mean()
    ax.axvline(mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r2:.4f}')
    
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of R² Scores Across Experiments', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / 'r2_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {output_file}")
    plt.close()

def main():
    """Main execution function"""
    
    # Create visualizations directory
    viz_dir = Path("reports/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Generating MLflow Experiment Visualizations")
    print("="*70)
    
    # Process ElasticNet results
    print("\nProcessing ElasticNet results...")
    elasticnet_uri = "file:./results/elasticnet/mlruns"
    
    try:
        mlflow.set_tracking_uri(elasticnet_uri)
        experiment = mlflow.get_experiment_by_name("wine-quality-elasticnet-optimization")
        
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.test_rmse ASC"]
            )
            
            if len(runs) > 0:
                print(f"Found {len(runs)} ElasticNet runs")
                create_metric_comparison_chart(runs, viz_dir)
                create_parameter_heatmap(runs, viz_dir)
                create_metrics_scatter(runs, viz_dir)
                create_r2_distribution(runs, viz_dir)
        else:
            print("ElasticNet experiment not found")
    except Exception as e:
        print(f"Error processing ElasticNet results: {e}")
    
    # Process Experiment Series results
    print("\nProcessing Experiment Series results...")
    series_uri = "file:./results/experiment-series/mlruns"
    
    try:
        mlflow.set_tracking_uri(series_uri)
        experiment = mlflow.get_experiment_by_name("wine-quality-rmse-optimization-series")
        
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.test_rmse ASC"]
            )
            
            if len(runs) > 0:
                print(f"Found {len(runs)} Experiment Series runs")
                
                # Create strategy comparison chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by experiment type
                if 'tags.experiment_type' in runs.columns:
                    strategy_data = runs.groupby('tags.experiment_type')['metrics.test_rmse'].min()
                    
                    bars = ax.bar(range(len(strategy_data)), strategy_data.values, color='steelblue')
                    bars[strategy_data.argmin()].set_color('green')
                    
                    ax.set_xticks(range(len(strategy_data)))
                    ax.set_xticklabels(strategy_data.index, rotation=45, ha='right')
                    ax.set_ylabel('Best RMSE', fontsize=12)
                    ax.set_title('Strategy Comparison: Best RMSE per Approach', fontsize=14, fontweight='bold')
                    
                    for i, value in enumerate(strategy_data.values):
                        ax.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
                    
                    plt.tight_layout()
                    
                    output_file = viz_dir / 'strategy_comparison.png'
                    plt.savefig(output_file, dpi=150, bbox_inches='tight')
                    print(f"✓ Chart saved: {output_file}")
                    plt.close()
        else:
            print("Experiment Series experiment not found")
    except Exception as e:
        print(f"Error processing Experiment Series results: {e}")
    
    print("\n" + "="*70)
    print("✓ Visualization generation complete!")
    print(f"✓ Charts saved to: {viz_dir.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()
