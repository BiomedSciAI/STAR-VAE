#!/usr/bin/env python3
"""
Combine rediscovery results from multiple experiment folders with dataset-specific analysis.

This script:
1. Iterates over multiple folders containing rediscovery results
2. Loads all CSV files matching '*-scaffolds-redis.csv'
3. Extracts dataset information from filenames (ABL1, BRD4, GRL, hiv1protease)
4. Adds 'model', 'num_samples', and 'dataset' columns
5. Combines results from all folders, keeping only common columns
6. Exports to a single combined CSV file with dataset column
7. Creates dataset-specific visualization charts (non-clustering and clustering separately)
8. Creates comparison tables in LaTeX format for rediscovery metrics
"""

import pandas as pd
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import re


def _load_altair():
    try:
        import altair as alt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chart generation requires the optional 'altair' package. Install "
            "it in the project environment before requesting chart outputs."
        ) from exc
    alt.data_transformers.disable_max_rows()
    return alt


def sanitize_column_name(column_name: str) -> str:
    """
    Convert a source column name into a Vega-Lite-safe field name.

    Args:
        column_name: Original DataFrame column name

    Returns:
        Sanitized field name safe for Altair encodings
    """
    sanitized = re.sub(r'[^0-9a-zA-Z_]+', '_', column_name).strip('_')
    if not sanitized:
        sanitized = 'metric'
    if sanitized[0].isdigit():
        sanitized = f'field_{sanitized}'
    return sanitized


def prepare_metric_chart_df(
    dataset_df: pd.DataFrame,
    measure: str
) -> Tuple[pd.DataFrame, str]:
    """
    Prepare an aggregated chart DataFrame with a sanitized metric column name.

    Args:
        dataset_df: Dataset-specific DataFrame
        measure: Original metric column name

    Returns:
        Tuple of (aggregated DataFrame, sanitized metric field name)
    """
    measure_df = dataset_df[['model', 'num_samples', measure]].copy()
    measure_df[measure] = pd.to_numeric(measure_df[measure], errors='coerce')
    measure_df['num_samples'] = pd.to_numeric(measure_df['num_samples'], errors='coerce')
    measure_df['model'] = measure_df['model'].astype(str)
    measure_df = measure_df.dropna(subset=[measure, 'num_samples', 'model'])

    if measure_df.empty:
        return pd.DataFrame(), sanitize_column_name(measure)

    agg_df = measure_df.groupby(['model', 'num_samples'], as_index=False)[measure].mean()
    safe_measure = sanitize_column_name(measure)
    agg_df = agg_df.rename(columns={measure: safe_measure})

    return agg_df, safe_measure


def extract_dataset_from_filename(filename: str) -> str:
    """
    Extract dataset name from filename.
    
    Args:
        filename: Name of the CSV file (e.g., 'ABL1_cvae_generated-scaffolds-redis.csv')
        
    Returns:
        Dataset name (ABL1, BRD4, GRL, or hiv1protease), or None if not found
    """
    # Known datasets
    datasets = ['ABL1', 'BRD4', 'GRL', 'hiv1protease']
    
    filename_upper = filename.upper()
    for dataset in datasets:
        if filename_upper.startswith(dataset.upper()):
            return dataset
    
    return None


def find_redis_csv_files(folder_path: str) -> List[str]:
    """
    Find all CSV files containing '-scaffolds-redis.csv' in the filename.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of full paths to matching CSV files
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return []
    
    # Find all files matching the pattern
    csv_files = list(folder.glob('*-scaffolds-redis.csv'))
    csv_files = [str(f) for f in csv_files]
    
    print(f"Found {len(csv_files)} redis CSV files in {folder_path}")
    return csv_files


def extract_num_samples_from_path(folder_path: str) -> int:
    """
    Extract num_samples from folder path by finding the last numeric part after underscore.
    
    Args:
        folder_path: Path to the folder (e.g., '.../vae_cluster_10000')
        
    Returns:
        Number of samples as integer, or None if not found
    """
    # Get the folder name (last part of path)
    folder_name = os.path.basename(folder_path.rstrip('/'))
    
    # Split by underscore and find the last part that's numeric
    parts = folder_name.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    
    print(f"Warning: Could not extract num_samples from folder path: {folder_path}")
    return None


def load_and_combine_folder(folder_path: str, model_name: str) -> pd.DataFrame:
    """
    Load all redis CSV files from a folder and combine them with model, num_samples, and dataset columns.
    
    Args:
        folder_path: Path to the folder containing CSV files
        model_name: Name to assign to the 'model' column
        
    Returns:
        Combined DataFrame with 'model', 'num_samples', and 'dataset' columns added
    """
    csv_files = find_redis_csv_files(folder_path)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return pd.DataFrame()
    
    # Extract num_samples from folder path
    num_samples = extract_num_samples_from_path(folder_path)
    
    dfs = []
    for csv_file in csv_files:
        try:
            # Extract dataset from filename
            filename = os.path.basename(csv_file)
            dataset = extract_dataset_from_filename(filename)
            
            if dataset is None:
                print(f"  Warning: Could not extract dataset from {filename}, skipping")
                continue
            
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['model'] = model_name
            df['num_samples'] = num_samples
            df['dataset'] = dataset
            
            print(f"  Loaded {filename}: {len(df)} rows, dataset={dataset}")
            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
            continue
    
    if not dfs:
        print(f"No valid CSV files loaded from {folder_path}")
        return pd.DataFrame()
    
    # Combine all DataFrames from this folder
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Combined {len(dfs)} files from {folder_path}: {len(combined_df)} total rows (num_samples={num_samples})")
    return combined_df


def get_common_columns(dfs: List[pd.DataFrame]) -> List[str]:
    """
    Get columns that appear in all DataFrames.
    
    Args:
        dfs: List of DataFrames
        
    Returns:
        List of column names common to all DataFrames
    """
    if not dfs:
        return []
    
    # Start with columns from first DataFrame
    common_cols = set(dfs[0].columns)
    
    # Intersect with columns from all other DataFrames
    for df in dfs[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    # Convert to list and maintain some order (metadata columns last)
    common_cols_list = [col for col in dfs[0].columns if col in common_cols]
    
    # Ensure metadata columns are included and at the end
    metadata_cols = ['num_samples', 'model', 'dataset']
    for col in metadata_cols:
        if col in common_cols_list:
            common_cols_list.remove(col)
    
    common_cols_list.extend(metadata_cols)
    
    return common_cols_list


def combine_all_folders(
    folders: List[str], 
    model_names: List[str]
) -> pd.DataFrame:
    """
    Combine redis results from all folders.
    
    Args:
        folders: List of folder paths
        model_names: List of model names corresponding to each folder
        
    Returns:
        Combined DataFrame with only common columns
    """
    all_dfs = []
    
    for folder, model_name in zip(folders, model_names):
        print(f"\nProcessing folder: {folder}")
        print(f"Model name: {model_name}")
        
        df = load_and_combine_folder(folder, model_name)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("\nError: No data loaded from any folder")
        return pd.DataFrame()
    
    # Get common columns across all folders
    common_cols = get_common_columns(all_dfs)
    print(f"\nCommon columns across all folders ({len(common_cols)}): {common_cols}")
    
    # Select only common columns from each DataFrame
    filtered_dfs = [df[common_cols] for df in all_dfs]
    
    # Combine all DataFrames
    final_df = pd.concat(filtered_dfs, ignore_index=True)
    
    print(f"\nCombined DataFrame before filtering: {len(final_df)} rows, {len(final_df.columns)} columns")
    
    # Filter out rows with "_train" in the Method column
    if 'Method' in final_df.columns:
        before_filter = len(final_df)
        final_df = final_df[~final_df['Method'].str.contains('_train', case=False, na=False)]
        after_filter = len(final_df)
        removed = before_filter - after_filter
        print(f"Removed {removed} rows containing '_train' in Method column")
    else:
        print("Warning: 'Method' column not found, skipping _train filter")
    
    print(f"\nFinal combined DataFrame: {len(final_df)} rows, {len(final_df.columns)} columns")
    print(f"Models included: {final_df['model'].unique().tolist()}")
    print(f"Datasets included: {final_df['dataset'].unique().tolist()}")
    
    return final_df


def create_clustering_visualizations(df: pd.DataFrame, output_prefix: str = 'redis_combined_clustering'):
    """
    Create separate visualization PDFs for clustering metrics for each dataset.
    
    Args:
        df: Combined DataFrame with clustering metrics
        output_prefix: Prefix for output PDF files
    """
    alt = _load_altair()
    print("\n" + "=" * 80)
    print("Creating Clustering Metrics Visualizations")
    print("=" * 80)
    
    datasets = df['dataset'].unique()
    
    # Only include clustering metrics
    clustering_prefixes = ('Clustering',)
    excluded_columns = {'Method', 'model', 'num_samples', 'dataset'}
    
    clustering_columns = [
        col for col in df.columns
        if col not in excluded_columns and any(col.startswith(prefix) for prefix in clustering_prefixes)
    ]
    
    if not clustering_columns:
        print("Warning: No clustering metrics found in DataFrame")
        return
    
    print(f"Found {len(clustering_columns)} clustering metrics: {clustering_columns}")
    
    for dataset in datasets:
        print(f"\nCreating clustering visualization for dataset: {dataset}")
        dataset_df = df[df['dataset'] == dataset].copy()
        
        if dataset_df.empty:
            print(f"  Warning: No data for dataset {dataset}")
            continue
        
        charts = []
        
        for measure in clustering_columns:
            agg_df, safe_measure = prepare_metric_chart_df(dataset_df, measure)

            if agg_df.empty:
                continue

            # Shorten the title for better display
            short_title = measure.replace('Clustering Metrics: ', '')

            chart = alt.Chart(agg_df).mark_line(point=True).encode(
                x=alt.X(
                    'num_samples:Q',
                    title='num_samples',
                    axis=alt.Axis(format=',.0f')
                ),
                y=alt.Y(
                    f'{safe_measure}:Q',
                    title=short_title,
                    scale=alt.Scale(zero=False, nice=True)
                ),
                color=alt.Color(
                    'model:N',
                    title='model',
                    legend=alt.Legend(orient='bottom')
                ),
                tooltip=[
                    alt.Tooltip('model:N', title='model'),
                    alt.Tooltip('num_samples:Q', title='num_samples', format=',.0f'),
                    alt.Tooltip(f'{safe_measure}:Q', title=short_title, format='.4f'),
                ]
            ).properties(
                width=300,
                height=250,
                title=short_title
            )

            charts.append(chart)
        
        if not charts:
            print(f"  Warning: No clustering charts created for dataset {dataset}")
            continue
        
        # Arrange charts in a grid (2 columns)
        if len(charts) == 1:
            final_chart = charts[0]
        elif len(charts) == 2:
            final_chart = alt.hconcat(*charts)
        else:
            # Create rows of 2 charts each
            rows = []
            for i in range(0, len(charts), 2):
                if i + 1 < len(charts):
                    rows.append(alt.hconcat(charts[i], charts[i+1]))
                else:
                    rows.append(charts[i])
            final_chart = alt.vconcat(*rows)
        
        final_chart = final_chart.resolve_scale(
            color='shared'
        ).properties(
            title=alt.TitleParams(
                text=f'Clustering Metrics vs num_samples - {dataset}',
                fontSize=16,
                anchor='middle'
            )
        )
        
        output_pdf = f"{output_prefix}_{dataset}.pdf"
        try:
            final_chart.save(output_pdf)
            print(f"  ✓ Successfully saved clustering visualization to: {output_pdf}")
        except Exception as e:
            print(f"  Error saving clustering visualization: {e}")


def create_dataset_specific_visualizations(df: pd.DataFrame, output_prefix: str = 'redis_combined'):
    """
    Create separate visualization PDFs for each dataset (excluding clustering metrics).
    
    Args:
        df: Combined DataFrame with rediscovery metrics
        output_prefix: Prefix for output PDF files
    """
    alt = _load_altair()
    print("\n" + "=" * 80)
    print("Creating Dataset-Specific Visualizations (Non-Clustering Metrics)")
    print("=" * 80)
    
    datasets = df['dataset'].unique()
    
    excluded_prefixes = ('Clustering',)
    excluded_columns = {'Method', 'model', 'num_samples', 'dataset'}
    
    measure_columns = [
        col for col in df.columns
        if col not in excluded_columns and not any(col.startswith(prefix) for prefix in excluded_prefixes)
    ]
    
    if not measure_columns:
        print("Error: No non-clustering measure columns found.")
        return
    
    for dataset in datasets:
        print(f"\nCreating visualization for dataset: {dataset}")
        dataset_df = df[df['dataset'] == dataset].copy()
        
        if dataset_df.empty:
            print(f"  Warning: No data for dataset {dataset}")
            continue
        
        charts = []
        
        for measure in measure_columns:
            agg_df, safe_measure = prepare_metric_chart_df(dataset_df, measure)

            if agg_df.empty:
                continue

            chart = alt.Chart(agg_df).mark_line(point=True).encode(
                x=alt.X(
                    'num_samples:Q',
                    title='num_samples',
                    axis=alt.Axis(format=',.0f')
                ),
                y=alt.Y(
                    f'{safe_measure}:Q',
                    title=measure,
                    scale=alt.Scale(zero=False, nice=True)
                ),
                color=alt.Color(
                    'model:N',
                    title='model',
                    legend=alt.Legend(orient='bottom')
                ),
                tooltip=[
                    alt.Tooltip('model:N', title='model'),
                    alt.Tooltip('num_samples:Q', title='num_samples', format=',.0f'),
                    alt.Tooltip(f'{safe_measure}:Q', title=measure, format='.4f'),
                ]
            ).properties(
                width=220,
                height=220,
                title=measure
            )

            charts.append(chart)
        
        if not charts:
            print(f"  Warning: No charts created for dataset {dataset}")
            continue
        
        final_chart = alt.hconcat(*charts).resolve_scale(
            color='shared'
        ).properties(
            title=alt.TitleParams(
                text=f'Rediscovery Metrics vs num_samples - {dataset}',
                fontSize=16,
                anchor='middle'
            )
        )
        
        output_pdf = f"{output_prefix}_{dataset}.pdf"
        try:
            final_chart.save(output_pdf)
            print(f"  ✓ Successfully saved visualization to: {output_pdf}")
        except Exception as e:
            print(f"  Error saving visualization: {e}")


def create_rediscovery_comparison_table(df: pd.DataFrame, output_tex: str = 'redis_rediscovery.tex'):
    """
    Create a comparison table for rediscovery metrics across models and datasets.
    Export as LaTeX table format.
    
    Args:
        df: Combined DataFrame with rediscovery metrics
        output_tex: Path to save the LaTeX table file
    """
    print("\n" + "=" * 80)
    print("Creating Rediscovery Comparison Table")
    print("=" * 80)
    
    # Select key rediscovery metrics
    rediscovery_metrics = [
        'Scaffold Rediscovery | Scaff(Generated) ∩ Scaff(Heldout) / Scaff(Heldout)',
        'Scaffold Rediscovery | Scaff(Generated) ∩ Scaff(Heldout) / Scaff(Generated)',
        'Nearest Neighbor Discovery | Mean Sim Score'
    ]
    
    # Check which metrics are available
    available_metrics = [m for m in rediscovery_metrics if m in df.columns]
    
    if not available_metrics:
        print("Error: No rediscovery metrics found in DataFrame")
        return
    
    print(f"Using metrics: {available_metrics}")
    
    # Create summary table
    results = []
    
    for dataset in sorted(df['dataset'].unique()):
        for model in sorted(df['model'].unique()):
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
            
            if subset.empty:
                continue
            
            row = {
                'Dataset': dataset,
                'Model': model,
                'Num Samples': subset['num_samples'].max()
            }
            
            for metric in available_metrics:
                # Use shorter names for table
                short_name = metric.split('|')[-1].strip()
                if 'Heldout) / Scaff(Heldout)' in metric:
                    short_name = 'Recall'
                elif 'Heldout) / Scaff(Generated)' in metric:
                    short_name = 'Precision'
                elif 'Mean Sim Score' in metric:
                    short_name = 'Mean Sim'
                
                # Convert to numeric and compute mean
                values = pd.to_numeric(subset[metric], errors='coerce')
                mean_val = values.mean()
                row[short_name] = mean_val
            
            results.append(row)
    
    if not results:
        print("Error: No results to create table")
        return
    
    result_df = pd.DataFrame(results)
    
    # Sort by dataset and model
    result_df = result_df.sort_values(['Dataset', 'Model'])
    
    print("\nComparison Table:")
    print(result_df.to_string(index=False))
    
    # Export to LaTeX
    try:
        # Create a nicely formatted LaTeX table
        latex_str = result_df.to_latex(
            index=False,
            float_format='%.4f',
            caption='Rediscovery Metrics Comparison Across Models and Datasets',
            label='tab:rediscovery_comparison',
            column_format='l' * len(result_df.columns),
            escape=False
        )
        
        with open(output_tex, 'w') as f:
            f.write(latex_str)
        
        print(f"\n✓ Successfully saved LaTeX table to: {output_tex}")
    except Exception as e:
        print(f"\nError saving LaTeX table: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Combine redis results from multiple experiment folders with dataset-specific analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default folders and names
  python bmfm-combine-redis-results-fixed.py
  
  # Custom folders and names
  python bmfm-combine-redis-results-fixed.py \\
    --folders /path/to/folder1 /path/to/folder2 \\
    --names Model1 Model2 \\
    --output my_results.csv
        """
    )
    
    _output_dir = os.environ.get('OUTPUT_DIR', 'outputs')
    parser.add_argument(
        '--folders',
        nargs='+',
        default=[
            f"{_output_dir}/chembl_comp_validation_pubchem_samples_10000",
            f"{_output_dir}/chembl_comp_validation_pubchem_samples_20000",
            f"{_output_dir}/chembl_comp_validation_pubchem_samples_30000",
            f"{_output_dir}/chembl_comp_validation_pubchem_samples_40000",
            f"{_output_dir}/chembl_comp_validation_pubchem_samples_50000",
            f"{_output_dir}/chembl_comp_validation_cvae_decoder_only_10000",
            f"{_output_dir}/chembl_comp_validation_cvae_decoder_only_20000",
            f"{_output_dir}/chembl_comp_validation_cvae_decoder_only_30000",
            f"{_output_dir}/chembl_comp_validation_cvae_decoder_only_40000",
            f"{_output_dir}/chembl_comp_validation_cvae_decoder_only_50000",
            f"{_output_dir}/chembl_comp_validation_vae_10000",
            f"{_output_dir}/chembl_comp_validation_vae_20000",
            f"{_output_dir}/chembl_comp_validation_vae_30000",
            f"{_output_dir}/chembl_comp_validation_vae_40000",
            f"{_output_dir}/chembl_comp_validation_vae_50000",
        ],
        help='List of folder paths containing redis CSV files'
    )
    
    parser.add_argument(
        '--names',
        nargs='+',
        default=[
            'PUBCHEM', 'PUBCHEM', 'PUBCHEM', 'PUBCHEM', 'PUBCHEM',
            'CVAE', 'CVAE', 'CVAE', 'CVAE', 'CVAE',
            'VAE', 'VAE', 'VAE', 'VAE', 'VAE'
        ],
        help='List of model names corresponding to each folder'
    )
    
    parser.add_argument(
        '--output',
        default='redis_combined.csv',
        help='Output CSV file path (default: redis_combined.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.folders) != len(args.names):
        print(f"Error: Number of folders ({len(args.folders)}) must match number of names ({len(args.names)})")
        sys.exit(1)
    
    print("=" * 80)
    print("Redis Results Combiner with Dataset-Specific Analysis (FIXED VERSION)")
    print("=" * 80)
    print(f"\nFolders to process: {len(args.folders)}")
    for folder, name in zip(args.folders, args.names):
        print(f"  - {name}: {folder}")
    print(f"\nOutput file: {args.output}")
    print("=" * 80)
    
    # Combine all folders
    combined_df = combine_all_folders(args.folders, args.names)
    
    if combined_df.empty:
        print("\nNo data to export. Exiting.")
        sys.exit(1)
    
    # Export to CSV
    try:
        combined_df.to_csv(args.output, index=False)
        print(f"\n✓ Successfully exported to: {args.output}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total columns: {len(combined_df.columns)}")
        
        # Print summary statistics
        print("\nSummary by model:")
        model_counts = combined_df['model'].value_counts()
        for model, count in model_counts.items():
            print(f"  {model}: {count} rows")
        
        print("\nSummary by dataset:")
        dataset_counts = combined_df['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} rows")
        
    except Exception as e:
        print(f"\nError exporting to CSV: {e}")
        sys.exit(1)
    
    # Create dataset-specific visualizations (non-clustering metrics)
    create_dataset_specific_visualizations(combined_df, 'redis_combined')
    
    # Create clustering metrics visualizations
    create_clustering_visualizations(combined_df, 'redis_combined_clustering')
    
    # Create comparison table
    create_rediscovery_comparison_table(combined_df, 'redis_rediscovery.tex')


if __name__ == "__main__":
    main()
