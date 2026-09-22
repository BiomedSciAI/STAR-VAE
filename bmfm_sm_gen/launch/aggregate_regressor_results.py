import os
import json
import pandas as pd
import argparse
from pathlib import Path


def aggregate_results(base_dir, output_dir):
    """
    Aggregate all regressor results from subdirectories.
    
    Args:
        base_dir: Base directory containing all regressor result subdirectories
        output_dir: Directory to save aggregated results
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lists to store aggregated data
    all_metrics = []
    all_train_metrics = []
    all_test_metrics = []
    
    # Find all subdirectories with results
    subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('regressor_')]
    
    if not subdirs:
        print(f"No regressor result directories found in {base_dir}")
        return
    
    print(f"Found {len(subdirs)} result directories")
    
    for subdir in sorted(subdirs):
        dataset_name = subdir.name.replace('regressor_', '')
        print(f"Processing {dataset_name}...")
        
        # Load metrics.json
        metrics_file = subdir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                metrics['dataset_name'] = dataset_name
                all_metrics.append(metrics)
        else:
            print(f"  Warning: metrics.json not found in {subdir}")
        
        # Load train_metrics.csv
        train_metrics_file = subdir / 'train_metrics.csv'
        if train_metrics_file.exists():
            train_df = pd.read_csv(train_metrics_file)
            train_df['dataset_name'] = dataset_name
            all_train_metrics.append(train_df)
        else:
            print(f"  Warning: train_metrics.csv not found in {subdir}")
        
        # Load test_metrics.csv
        test_metrics_file = subdir / 'test_metrics.csv'
        if test_metrics_file.exists():
            test_df = pd.read_csv(test_metrics_file)
            test_df['dataset_name'] = dataset_name
            all_test_metrics.append(test_df)
        else:
            print(f"  Warning: test_metrics.csv not found in {subdir}")
    
    # Save aggregated metrics as JSON
    if all_metrics:
        aggregated_json_path = output_path / 'aggregated_metrics.json'
        with open(aggregated_json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAggregated metrics JSON saved to {aggregated_json_path}")
    
    # Save aggregated train metrics as CSV
    if all_train_metrics:
        train_df_combined = pd.concat(all_train_metrics, ignore_index=True)
        # Reorder columns to put dataset_name first
        cols = ['dataset_name'] + [col for col in train_df_combined.columns if col != 'dataset_name']
        train_df_combined = train_df_combined[cols]
        
        aggregated_train_csv_path = output_path / 'aggregated_train_metrics.csv'
        train_df_combined.to_csv(aggregated_train_csv_path, index=False)
        print(f"Aggregated train metrics CSV saved to {aggregated_train_csv_path}")
    
    # Save aggregated test metrics as CSV
    if all_test_metrics:
        test_df_combined = pd.concat(all_test_metrics, ignore_index=True)
        # Reorder columns to put dataset_name first
        cols = ['dataset_name'] + [col for col in test_df_combined.columns if col != 'dataset_name']
        test_df_combined = test_df_combined[cols]
        
        aggregated_test_csv_path = output_path / 'aggregated_test_metrics.csv'
        test_df_combined.to_csv(aggregated_test_csv_path, index=False)
        print(f"Aggregated test metrics CSV saved to {aggregated_test_csv_path}")
    
    # Print summary statistics
    if all_test_metrics:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS (Test Set)")
        print("="*60)
        print(f"Number of datasets: {len(test_df_combined)}")
        print("\nAverage metrics across all datasets:")
        for metric in ['rmse', 'mae', 'r2', 'pearson_corr', 'spearman_corr']:
            if metric in test_df_combined.columns:
                mean_val = test_df_combined[metric].mean()
                std_val = test_df_combined[metric].std()
                print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate regressor results from multiple datasets"
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        required=True,
        help='Base directory containing all regressor result subdirectories'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save aggregated results (default: base_dir/aggregated)'
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else os.path.join(args.base_dir, 'aggregated')
    
    aggregate_results(args.base_dir, output_dir)


if __name__ == "__main__":
    main()
