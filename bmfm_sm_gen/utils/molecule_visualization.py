"""
Molecule visualization utilities for clustering analysis.
"""

import pandas as pd
import numpy as np
import json
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import io


def visualize_molecules_by_cluster(
    df: pd.DataFrame,
    cluster_col: str = "Cluster Index",
    smiles_col: str = "Smiles",
    fp_col: str = "morgan_fp",
    img_size: Tuple[int, int] = (400, 400),
    molecules_per_cluster: int = 3,
    save_path: Optional[str] = None
) -> SVG:
    """
    Visualize representative molecules from each cluster in a grid layout.
    Grid layout: columns = clusters, rows = molecules per cluster
    
    For each cluster, displays:
    1. The molecule closest to the cluster centroid
    2. Two molecules furthest from the centroid
    
    Args:
        df: DataFrame containing molecules with cluster assignments
        cluster_col: Name of column containing cluster indices
        smiles_col: Name of column containing SMILES strings
        fp_col: Name of column containing fingerprints (as JSON strings)
        img_size: Size of each molecule image (width, height) in pixels
        molecules_per_cluster: Number of molecules to show per cluster (default: 3)
        save_path: Optional path to save the SVG output
        
    Returns:
        SVG object that can be displayed in Jupyter notebooks
    """
    
    # Get unique clusters (excluding invalid cluster index -1)
    clusters = sorted([c for c in df[cluster_col].unique() if c >= 0])
    n_clusters = len(clusters)
    
    if n_clusters == 0:
        raise ValueError("No valid clusters found in the DataFrame")
    
    print(f"Visualizing {n_clusters} clusters with {molecules_per_cluster} molecules each...")
    
    # Prepare grid layout: columns = clusters, rows = molecules per cluster
    n_cols = n_clusters
    n_rows = molecules_per_cluster
    
    # Calculate total canvas size
    canvas_width = img_size[0] * n_cols
    canvas_height = img_size[1] * n_rows
    
    # Create SVG drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(canvas_width, canvas_height, img_size[0], img_size[1])
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().addStereoAnnotation = True
    
    # First, collect molecules for each cluster
    cluster_molecules = {}  # cluster_id -> list of (mol, label) tuples
    
    for cluster_idx, cluster_id in enumerate(clusters):
        # Get molecules in this cluster
        cluster_df = df[df[cluster_col] == cluster_id].copy()
        
        if len(cluster_df) == 0:
            continue
        
        # Parse fingerprints
        fps = []
        indices = []
        for idx, row in cluster_df.iterrows():
            try:
                fp_json = row[fp_col]
                if pd.notna(fp_json) and fp_json != "":
                    fp_array = np.array(json.loads(fp_json), dtype=np.float32)
                    fps.append(fp_array)
                    indices.append(idx)
            except Exception as e:
                print(f"Warning: Could not parse fingerprint at index {idx}: {e}")
                continue
        
        if len(fps) == 0:
            print(f"Warning: No valid fingerprints for cluster {cluster_id}")
            continue
        
        fps_array = np.array(fps)
        
        # Calculate cluster centroid
        centroid = fps_array.mean(axis=0)
        
        # Calculate distances to centroid (Euclidean distance)
        distances = np.linalg.norm(fps_array - centroid, axis=1)
        
        # Find closest molecule to centroid
        closest_idx = np.argmin(distances)
        
        # Find furthest molecules from centroid
        furthest_indices = np.argsort(distances)[-2:][::-1]  # Get top 2 furthest
        
        # Select molecules to display
        selected_indices = [closest_idx]
        if len(furthest_indices) >= 2:
            selected_indices.extend(furthest_indices[:2])
        elif len(furthest_indices) == 1:
            selected_indices.append(furthest_indices[0])
        
        # Ensure we have exactly molecules_per_cluster molecules
        while len(selected_indices) < molecules_per_cluster and len(selected_indices) < len(indices):
            # Add random molecules if needed
            remaining = [i for i in range(len(indices)) if i not in selected_indices]
            if remaining:
                selected_indices.append(remaining[0])
            else:
                break
        
        # Store molecules for this cluster
        cluster_molecules[cluster_id] = []
        for mol_idx, fp_idx in enumerate(selected_indices[:molecules_per_cluster]):
            df_idx = indices[fp_idx]
            smiles = cluster_df.loc[df_idx, smiles_col]
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Create legend
                    dist = distances[fp_idx]
                    if mol_idx == 0:
                        label = f"Cluster {cluster_id}\nCentroid (d={dist:.3f})"
                    else:
                        label = f"Cluster {cluster_id}\nFar {mol_idx} (d={dist:.3f})"
                    cluster_molecules[cluster_id].append((mol, label))
                else:
                    print(f"Warning: Invalid SMILES at index {df_idx}: {smiles}")
            except Exception as e:
                print(f"Warning: Error processing SMILES at index {df_idx}: {e}")
    
    # Now arrange molecules in grid: rows = molecules, columns = clusters
    # This means we iterate by molecule position first, then by cluster
    mols_to_draw = []
    legends = []
    
    for mol_idx in range(molecules_per_cluster):
        for cluster_id in clusters:
            if cluster_id in cluster_molecules and mol_idx < len(cluster_molecules[cluster_id]):
                mol, label = cluster_molecules[cluster_id][mol_idx]
                mols_to_draw.append(mol)
                legends.append(label)
    
    if len(mols_to_draw) == 0:
        raise ValueError("No valid molecules to draw")
    
    # Draw all molecules in grid
    drawer.DrawMolecules(mols_to_draw, legends=legends)
    drawer.FinishDrawing()
    
    # Get SVG
    svg = drawer.GetDrawingText()
    
    # Save if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(svg)
        print(f"Saved visualization to {save_path}")
    
    return SVG(svg)


def visualize_cluster_diversity(
    df: pd.DataFrame,
    cluster_col: str = "Cluster Index",
    smiles_col: str = "Smiles",
    fp_col: str = "morgan_fp",
    n_samples: int = 5,
    img_size: Tuple[int, int] = (300, 300),
    save_path: Optional[str] = None
) -> SVG:
    """
    Visualize random samples from each cluster to show diversity.
    
    Args:
        df: DataFrame containing molecules with cluster assignments
        cluster_col: Name of column containing cluster indices
        smiles_col: Name of column containing SMILES strings
        fp_col: Name of column containing fingerprints (as JSON strings)
        n_samples: Number of random samples per cluster
        img_size: Size of each molecule image (width, height) in pixels
        save_path: Optional path to save the SVG output
        
    Returns:
        SVG object that can be displayed in Jupyter notebooks
    """
    
    # Get unique clusters (excluding invalid cluster index -1)
    clusters = sorted([c for c in df[cluster_col].unique() if c >= 0])
    n_clusters = len(clusters)
    
    if n_clusters == 0:
        raise ValueError("No valid clusters found in the DataFrame")
    
    print(f"Visualizing {n_clusters} clusters with {n_samples} random samples each...")
    
    # Prepare grid layout
    n_cols = n_samples
    n_rows = n_clusters
    
    # Calculate total canvas size
    canvas_width = img_size[0] * n_cols
    canvas_height = img_size[1] * n_rows
    
    # Create SVG drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(canvas_width, canvas_height, img_size[0], img_size[1])
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().addStereoAnnotation = True
    
    # Collect molecules and their positions
    mols_to_draw = []
    legends = []
    
    for cluster_id in clusters:
        # Get molecules in this cluster
        cluster_df = df[df[cluster_col] == cluster_id].copy()
        
        if len(cluster_df) == 0:
            continue
        
        # Sample random molecules
        sample_size = min(n_samples, len(cluster_df))
        sampled_df = cluster_df.sample(n=sample_size, random_state=42)
        
        # Add molecules to drawing list
        for idx, row in sampled_df.iterrows():
            smiles = row[smiles_col]
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols_to_draw.append(mol)
                    legends.append(f"Cluster {cluster_id}")
                else:
                    print(f"Warning: Invalid SMILES at index {idx}: {smiles}")
            except Exception as e:
                print(f"Warning: Error processing SMILES at index {idx}: {e}")
    
    if len(mols_to_draw) == 0:
        raise ValueError("No valid molecules to draw")
    
    # Draw all molecules in grid
    drawer.DrawMolecules(mols_to_draw, legends=legends)
    drawer.FinishDrawing()
    
    # Get SVG
    svg = drawer.GetDrawingText()
    
    # Save if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(svg)
        print(f"Saved visualization to {save_path}")
    
    return SVG(svg)