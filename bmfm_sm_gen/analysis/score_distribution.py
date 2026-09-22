#!/usr/bin/env python3
"""
Generate score distribution visualizations for ChEMBL datasets.
Creates a 4x4 grid of histograms comparing original and silver-label datasets.
"""

import pandas as pd
import altair as alt
import os
from pathlib import Path
import numpy as np

# Dataset names
DATASETS = ['ABL1', 'BRD4', 'GRL', 'hiv1protease']

# Paths
ORIGINAL_PATH = os.environ.get('CHEMBL_DATA_DIR', 'data/chembl/cluster_split')
SILVER_PATH = os.environ.get('CHEMBL_SPLIT_DIR', 'data/chembl/silver_label_100k_cluster_split')

# Output
OUTPUT_FILE = 'label_score_dist.pdf'


def load_and_prepare_data(base_path, dataset_name, split, score_column):
    """
    Load dataset and prepare for visualization.
    
    Args:
        base_path: Base directory path
        dataset_name: Name of the dataset (e.g., 'ABL1')
        split: 'train' or 'test'
        score_column: Column name for the score
    
    Returns:
        DataFrame with score data
    """
    # Find matching file
    pattern = f"*{dataset_name}*_{split}.csv"
    files = list(Path(base_path).glob(pattern))
    
    if not files:
        print(f"Warning: No file found for {dataset_name} {split} in {base_path}")
        return None
    
    filepath = files[0]
    print(f"Loading: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        if score_column not in df.columns:
            print(f"Warning: Column '{score_column}' not found in {filepath}")
            return None
        
        # Extract scores and add metadata
        scores = df[score_column].dropna()
        result_df = pd.DataFrame({
            'score': scores,
            'dataset': dataset_name,
            'split': split
        })
        
        return result_df
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_histogram(data, title, bins, y_max):
    """
    Create a histogram chart with consistent styling.
    
    Args:
        data: DataFrame with 'score' column
        title: Chart title
        bins: Bin configuration for histogram
        y_max: Maximum y-axis value
    
    Returns:
        Altair chart
    """
    chart = alt.Chart(data).mark_bar(opacity=0.7).encode(
        x=alt.X('score:Q',
                bin=bins,
                title='Label Score',
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('count():Q',
                title='Count',
                scale=alt.Scale(domain=[0, y_max])),
        tooltip=[
            alt.Tooltip('score:Q', bin=bins, title='Score Range'),
            alt.Tooltip('count():Q', title='Count')
        ]
    ).properties(
        title=title,
        width=200,
        height=150
    )
    
    return chart


def main():
    print("Starting score distribution visualization generation...")
    
    # Collect all data
    all_data = []
    
    # Row labels
    row_labels = [
        'Original Train',
        'Original Test',
        'Silver-Label Train',
        'Silver-Label Test'
    ]
    
    # Load original train data
    print("\n=== Loading Original Train Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_data(ORIGINAL_PATH, dataset, 'train', 'y_normalized_minmax')
        if df is not None:
            df['row_label'] = 'Original Train'
            all_data.append(df)
    
    # Load original test data
    print("\n=== Loading Original Test Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_data(ORIGINAL_PATH, dataset, 'test', 'y_normalized_minmax')
        if df is not None:
            df['row_label'] = 'Original Test'
            all_data.append(df)
    
    # Load silver-label train data
    print("\n=== Loading Silver-Label Train Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_data(SILVER_PATH, dataset, 'train', 'predicted_y_normalized_minmax')
        if df is not None:
            df['row_label'] = 'Silver-Label Train'
            all_data.append(df)
    
    # Load silver-label test data
    print("\n=== Loading Silver-Label Test Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_data(SILVER_PATH, dataset, 'test', 'predicted_y_normalized_minmax')
        if df is not None:
            df['row_label'] = 'Silver-Label Test'
            all_data.append(df)
    
    if not all_data:
        print("Error: No data loaded successfully!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined_df)}")
    print(f"Datasets: {combined_df['dataset'].unique()}")
    print(f"Row labels: {combined_df['row_label'].unique()}")
    
    # Calculate consistent bins and y-axis max
    # Use 20 bins from 0 to 1
    bins = alt.Bin(maxbins=20, extent=[0, 1])
    
    # Calculate max count across all histograms for consistent y-axis
    # Create temporary histograms to find max count
    max_count = 0
    for row_label in row_labels:
        for dataset in DATASETS:
            subset = combined_df[
                (combined_df['row_label'] == row_label) & 
                (combined_df['dataset'] == dataset)
            ]
            if len(subset) > 0:
                # Calculate histogram counts
                hist, _ = np.histogram(subset['score'], bins=20, range=(0, 1))
                max_count = max(max_count, hist.max())
    
    # Add some padding to y_max
    y_max = int(max_count * 1.1)
    print(f"\nUsing y_max: {y_max}")
    
    # Create grid of charts
    charts = []
    
    for row_label in row_labels:
        row_charts = []
        for dataset in DATASETS:
            # Filter data for this specific chart
            subset = combined_df[
                (combined_df['row_label'] == row_label) & 
                (combined_df['dataset'] == dataset)
            ]
            
            if len(subset) == 0:
                # Create empty chart with message
                empty_chart = alt.Chart(pd.DataFrame({'x': [0.5], 'y': [y_max/2]})).mark_text(
                    text='No Data',
                    size=14,
                    color='gray'
                ).encode(
                    x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]), title='Label Score'),
                    y=alt.Y('y:Q', scale=alt.Scale(domain=[0, y_max]), title='Count')
                ).properties(
                    title=f"{dataset} - {row_label}",
                    width=200,
                    height=150
                )
                row_charts.append(empty_chart)
            else:
                # Create histogram
                chart = create_histogram(
                    subset,
                    f"{dataset} - {row_label}",
                    bins,
                    y_max
                )
                row_charts.append(chart)
        
        # Concatenate charts in this row
        row_chart = alt.hconcat(*row_charts)
        charts.append(row_chart)
    
    # Concatenate all rows vertically
    final_chart = alt.vconcat(*charts).configure_axis(
        labelFontSize=10,
        titleFontSize=11
    ).configure_title(
        fontSize=12
    ).properties(
        title=alt.TitleParams(
            text='Label Score Distribution: Original vs Silver-Label Datasets',
            fontSize=16,
            anchor='middle'
        )
    )
    
    # Save to PDF
    print(f"\nSaving chart to {OUTPUT_FILE}...")
    final_chart.save(OUTPUT_FILE)
    print(f"Successfully saved to {OUTPUT_FILE}")
    
    # # Also save as HTML for interactive viewing
    # html_file = OUTPUT_FILE.replace('.pdf', '.html')
    # final_chart.save(html_file)
    # print(f"Also saved interactive version to {html_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for row_label in row_labels:
        print(f"\n{row_label}:")
        for dataset in DATASETS:
            subset = combined_df[
                (combined_df['row_label'] == row_label) & 
                (combined_df['dataset'] == dataset)
            ]
            if len(subset) > 0:
                print(f"  {dataset}: n={len(subset)}, "
                      f"mean={subset['score'].mean():.3f}, "
                      f"std={subset['score'].std():.3f}, "
                      f"min={subset['score'].min():.3f}, "
                      f"max={subset['score'].max():.3f}")


if __name__ == '__main__':
    main()