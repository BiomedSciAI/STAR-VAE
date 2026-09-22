#!/usr/bin/env python3
"""
Generate cluster membership distribution visualizations for ChEMBL datasets.
Creates a 4x4 grid of bar charts showing cluster sizes (number of members per cluster).
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
OUTPUT_FILE = 'cluster_membership_dist.png'
top_n=50

def load_and_prepare_cluster_data(base_path, dataset_name, split):
    """
    Load dataset and prepare cluster membership data.
    
    Args:
        base_path: Base directory path
        dataset_name: Name of the dataset (e.g., 'ABL1')
        split: 'train' or 'test'
    
    Returns:
        DataFrame with cluster membership counts
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
        
        if 'cluster_index' not in df.columns:
            print(f"Warning: Column 'cluster_index' not found in {filepath}")
            return None
        
        # Count members per cluster
        cluster_counts = df['cluster_index'].value_counts().reset_index()
        cluster_counts.columns = ['cluster_index', 'member_count']
        
        # Sort by member count in descending order
        cluster_counts = cluster_counts.sort_values('member_count', ascending=False).reset_index(drop=True)
        
        # Add rank for x-axis (1-based)
        cluster_counts['rank'] = range(1, len(cluster_counts) + 1)
        
        # Keep only top 30 clusters by rank
        total_clusters = len(cluster_counts)
        cluster_counts = cluster_counts.head(top_n)
        
        # Add metadata
        cluster_counts['dataset'] = dataset_name
        cluster_counts['split'] = split
        
        print(f"  Found {total_clusters} clusters (showing top 30), "
              f"largest: {cluster_counts['member_count'].max()}, "
              f"smallest: {cluster_counts['member_count'].min()}")
        
        return cluster_counts
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_cluster_chart(data, title, y_max):
    """
    Create a bar chart showing cluster membership distribution.
    
    Args:
        data: DataFrame with 'rank' and 'member_count' columns
        title: Chart title
        y_max: Maximum y-axis value
    
    Returns:
        Altair chart
    """
    chart = alt.Chart(data).mark_line(opacity=0.7, size=2).encode(
        x=alt.X('rank:Q',
                title='Cluster Rank (by size)',
                scale=alt.Scale(domain=[0, top_n+1])),
        y=alt.Y('member_count:Q',
                title='Number of Members',
                scale=alt.Scale(domain=[0, y_max])),
        tooltip=[
            alt.Tooltip('cluster_index:N', title='Cluster ID'),
            alt.Tooltip('rank:Q', title='Rank'),
            alt.Tooltip('member_count:Q', title='Members')
        ]
    ).properties(
        title=title,
        width=200,
        height=150
    )
    
    return chart


def main():
    print("Starting cluster membership distribution visualization generation...")
    
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
        df = load_and_prepare_cluster_data(ORIGINAL_PATH, dataset, 'train')
        if df is not None:
            df['row_label'] = 'Original Train'
            all_data.append(df)
    
    # Load original test data
    print("\n=== Loading Original Test Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_cluster_data(ORIGINAL_PATH, dataset, 'test')
        if df is not None:
            df['row_label'] = 'Original Test'
            all_data.append(df)
    
    # Load silver-label train data
    print("\n=== Loading Silver-Label Train Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_cluster_data(SILVER_PATH, dataset, 'train')
        if df is not None:
            df['row_label'] = 'Silver-Label Train'
            all_data.append(df)
    
    # Load silver-label test data
    print("\n=== Loading Silver-Label Test Data ===")
    for dataset in DATASETS:
        df = load_and_prepare_cluster_data(SILVER_PATH, dataset, 'test')
        if df is not None:
            df['row_label'] = 'Silver-Label Test'
            all_data.append(df)
    
    if not all_data:
        print("Error: No data loaded successfully!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal cluster records loaded: {len(combined_df)}")
    print(f"Datasets: {combined_df['dataset'].unique()}")
    print(f"Row labels: {combined_df['row_label'].unique()}")
    
    # Calculate y-axis max per dataset per source (original/silver)
    y_max_dict = {}
    
    print("\nCalculating y-axis maximums per dataset per source:")
    for dataset in DATASETS:
        # Original datasets
        original_data = combined_df[
            (combined_df['dataset'] == dataset) &
            (combined_df['row_label'].str.contains('Original'))
        ]
        if len(original_data) > 0:
            y_max_dict[(dataset, 'Original')] = int(original_data['member_count'].max() * 1.1)
            print(f"  {dataset} (Original): {y_max_dict[(dataset, 'Original')]}")
        else:
            y_max_dict[(dataset, 'Original')] = 100
        
        # Silver datasets
        silver_data = combined_df[
            (combined_df['dataset'] == dataset) &
            (combined_df['row_label'].str.contains('Silver'))
        ]
        if len(silver_data) > 0:
            y_max_dict[(dataset, 'Silver')] = int(silver_data['member_count'].max() * 1.1)
            print(f"  {dataset} (Silver): {y_max_dict[(dataset, 'Silver')]}")
        else:
            y_max_dict[(dataset, 'Silver')] = 100
    
    print(f"\nShowing top 30 clusters by rank for each dataset")
    
    # Create grid of charts
    charts = []
    
    for row_label in row_labels:
        row_charts = []
        for dataset in DATASETS:
            # Determine which y_max to use based on dataset and row label
            source = 'Original' if 'Original' in row_label else 'Silver'
            y_max = y_max_dict.get((dataset, source), 100)
            
            # Filter data for this specific chart
            subset = combined_df[
                (combined_df['row_label'] == row_label) &
                (combined_df['dataset'] == dataset)
            ]
            
            if len(subset) == 0:
                # Create empty chart with message
                empty_chart = alt.Chart(pd.DataFrame({'x': [15], 'y': [y_max/2]})).mark_text(
                    text='No Data',
                    size=14,
                    color='gray'
                ).encode(
                    x=alt.X('x:Q', scale=alt.Scale(domain=[0, top_n+1]), title='Cluster Rank (by size)'),
                    y=alt.Y('y:Q', scale=alt.Scale(domain=[0, y_max]), title='Number of Members')
                ).properties(
                    title=f"{dataset} - {row_label}",
                    width=200,
                    height=150
                )
                row_charts.append(empty_chart)
            else:
                # Create bar chart
                chart = create_cluster_chart(
                    subset,
                    f"{dataset} - {row_label}",
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
            text='Cluster Membership Distribution (Top 30): Original vs Silver-Label Datasets',
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
                print(f"  {dataset}: "
                      f"n_clusters={len(subset)}, "
                      f"max_members={subset['member_count'].max()}, "
                      f"min_members={subset['member_count'].min()}, "
                      f"mean_members={subset['member_count'].mean():.1f}, "
                      f"median_members={subset['member_count'].median():.1f}")


if __name__ == '__main__':
    main()