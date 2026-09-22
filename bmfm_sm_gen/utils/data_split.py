import json
import math
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

def tanimoto_distance_matrix(fp_list):
    dissimilarity_matrix = []
    for i in range(1, len(fp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def split_by_year(df: pd.DataFrame,
                  year_col: str = "Year",
                  seed: int = 42,
                  train_frac: float = 0.7,
                  valid_frac: float = 0.1,
                  test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:

    n_rows = len(df)
    
    # Handle empty DataFrame or DataFrame with no valid year values
    if n_rows == 0:
        raise ValueError("Cannot split an empty DataFrame")
    
    # Drop rows with NaN/null values in year column and check if any rows remain
    df_with_years = df.dropna(subset=[year_col])
    if len(df_with_years) == 0:
        raise ValueError(f"No valid (non-null) values found in column '{year_col}'")
    
    sorted_years = df_with_years[year_col].sort_values().reset_index(drop=True)
    target_idx = math.floor(len(df_with_years) * (train_frac + valid_frac))

    if target_idx == 0 or len(sorted_years) == 0:
        # If target_idx is 0, we have very few rows, use the first year as cutoff
        lower_year = middle_year = upper_year = sorted_years.iloc[0]
        cutoff_year = lower_year
    else:
        lower_year = sorted_years.iloc[target_idx] - 1
        middle_year = sorted_years.iloc[target_idx]
        upper_year = sorted_years.iloc[target_idx] + 1

        lower_cum = (df[year_col] > lower_year).mean()
        middle_cum = (df[year_col] > middle_year).mean()
        upper_cum = (df[year_col] > upper_year).mean()
        
        cutoff_cum = min((lower_cum, middle_cum, upper_cum), key=lambda x: abs(x - test_frac))
        if cutoff_cum == lower_cum:
            cutoff_year = lower_year
        elif cutoff_cum == middle_cum:
            cutoff_year = middle_year
        else:  # cutoff_cum == upper_cum
            cutoff_year = upper_year

    train_valid_df = df[df[year_col] <= cutoff_year].copy()
    test_df = df[df[year_col] > cutoff_year].copy()
    
    df_shuffled = train_valid_df.sample(frac=1, random_state=seed)
    train_size = math.floor(len(df_shuffled) * train_frac / (train_frac + valid_frac))
    train_df = df_shuffled.iloc[:train_size]
    valid_df = df_shuffled.iloc[train_size:]

    return train_df, valid_df, test_df, cutoff_year

def split_by_cluster(
    df,
    cluster_index_col='cluster_index',
    score_col='y_normalized_minmax',
    strata_bin_num=10,
    random_state=1,
    train_frac=0.7,
    valid_frac=0.1,
    test_frac=0.2,
):
    """
    Assign each complete cluster to exactly one split.

    ``score_col`` and ``strata_bin_num`` remain accepted for call-site
    compatibility; whole-cluster assignment cannot guarantee row-level strata.
    """
    del score_col, strata_bin_num
    fractions = np.array([train_frac, valid_frac, test_frac], dtype=float)
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")
    if cluster_index_col not in df.columns:
        raise ValueError(f"Column '{cluster_index_col}' not found in DataFrame")
    if df[cluster_index_col].isna().any():
        raise ValueError(f"Column '{cluster_index_col}' contains missing values")
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train_frac, valid_frac, and test_frac must be non-negative and sum to 1")

    cluster_sizes = df.groupby(cluster_index_col, sort=False).size()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(len(cluster_sizes))
    clusters = list(zip(cluster_sizes.index[shuffled], cluster_sizes.iloc[shuffled]))
    clusters.sort(key=lambda item: item[1], reverse=True)

    targets = fractions * len(df)
    counts = np.zeros(3, dtype=int)
    assigned = [set(), set(), set()]
    for cluster_id, size in clusters:
        eligible = np.flatnonzero(fractions > 0)
        deficits = (targets[eligible] - counts[eligible]) / fractions[eligible]
        split_idx = eligible[int(np.argmax(deficits))]
        assigned[split_idx].add(cluster_id)
        counts[split_idx] += size

    splits = tuple(
        df[df[cluster_index_col].isin(cluster_ids)].copy()
        for cluster_ids in assigned
    )
    split_cluster_sets = [set(part[cluster_index_col].unique()) for part in splits]
    assert split_cluster_sets[0].isdisjoint(split_cluster_sets[1])
    assert split_cluster_sets[0].isdisjoint(split_cluster_sets[2])
    assert split_cluster_sets[1].isdisjoint(split_cluster_sets[2])
    assert sum(len(part) for part in splits) == len(df)
    assert set().union(*split_cluster_sets) == set(cluster_sizes.index)
    return splits


def split_by_cluster_without_cluster_index(df: pd.DataFrame,
                     smiles_col: str = "Smiles",
                     fp_col: Optional[str] = None,
                     clustering_method: str = "agglomerative",
                     n_clusters_range: Optional[tuple] = (5, 40),
                     distance_threshold_range: Optional[list] = None,
                     train_frac: float = 0.7,
                     valid_frac: float = 0.1,
                     test_frac: float = 0.2,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform clustering on molecules based on fingerprints,
    then split into train/valid/test sets based on cluster assignments.
    
    Supports two clustering methods:
    - 'agglomerative': Agglomerative hierarchical clustering
    - 'butina': Butina clustering algorithm from RDKit
    
    Test set is created by selecting entire clusters (cluster-based holdout).
    Train and valid sets are randomly split from remaining clusters.
    
    Args:
        df: Input DataFrame containing SMILES strings and/or fingerprints
        smiles_col: Name of column containing SMILES strings (default: 'Smiles')
        fp_col: Name of column containing pre-computed fingerprints as JSON strings (optional).
                If provided, uses these instead of generating new fingerprints from SMILES.
        clustering_method: 'agglomerative' or 'butina' (default: 'agglomerative')
        n_clusters_range: Range of cluster numbers to try (min, max) for agglomerative.
                         If None and distance_threshold_range is provided, uses distance thresholds instead.
        distance_threshold_range: List of distance thresholds to try.
                                 For agglomerative: used with n_clusters=None
                                 For butina: required, used as distThresh parameter
        train_frac: Fraction of data for training (default: 0.7)
        valid_frac: Fraction of data for validation (default: 0.1)
        test_frac: Fraction of data for testing (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, valid_df, test_df, clustering_stats_df, df_valid)
        where:
        - train_df, valid_df, test_df: Split datasets
        - clustering_stats_df: Statistics for all clustering attempts
        - df_valid: Full dataframe with all clustering results as columns
    """

    
    if len(df) == 0:
        raise ValueError("Cannot cluster an empty DataFrame")
    
    print(f"Processing {len(df)} molecules for clustering...")
    
    # Step 1: Get or generate fingerprints
    fingerprints = []
    valid_indices = []
    
    if fp_col is not None and fp_col in df.columns:
        # Use pre-computed fingerprints from column
        print(f"Using pre-computed fingerprints from column '{fp_col}'...")
        for idx, fp_json in enumerate(df[fp_col]):
            try:
                if pd.isna(fp_json) or fp_json == "":
                    print(f"Warning: Empty fingerprint at index {idx}")
                    continue
                # Parse JSON string to numpy array
                fp_array = np.array(json.loads(fp_json), dtype=np.uint8)
                # Convert numpy array to RDKit ExplicitBitVect for Butina clustering
                fp = DataStructs.ExplicitBitVect(len(fp_array))
                for bit_idx, bit_val in enumerate(fp_array):
                    if bit_val:
                        fp.SetBit(bit_idx)
                fingerprints.append(fp)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Error parsing fingerprint at index {idx}: {e}")
                continue
    else:
        # Generate fingerprints from SMILES
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in DataFrame")
        
        print("Generating ECFP4 fingerprints (radius=2, 2048 bits)...")
        for idx, smiles in enumerate(df[smiles_col]):
            try:
                # Canonicalize SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Warning: Invalid SMILES at index {idx}: {smiles}")
                    continue
                
                canonical_smiles = Chem.MolToSmiles(mol)
                
                # Generate ECFP4 (Morgan fingerprint with radius 2)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                # Store RDKit fingerprint object directly (needed for Butina clustering)
                fingerprints.append(fp)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Warning: Error processing SMILES at index {idx}: {e}")
                continue
    
    if len(fingerprints) == 0:
        raise ValueError("No valid SMILES found for clustering")
    
    print(f"Successfully generated {len(fingerprints)} fingerprints")
    
    # Step 2: Calculate distance matrix (Tanimoto distance = 1 - Tanimoto similarity)
    # For Agglomerative clustering, we need numpy array and precomputed distance matrix
    # For Butina clustering, we use RDKit fingerprint objects directly
    print("Calculating distance matrix...")
    
    # Convert fingerprints to numpy array for distance calculation
    fp_array = np.array([np.array(fp) for fp in fingerprints])
    
    # For binary fingerprints, we can use Jaccard distance which equals Tanimoto distance
    distances = pdist(fp_array, metric='jaccard')
    distance_matrix = squareform(distances)
    
    # Initialize result dataframe early to store all clustering results
    df_result = df.copy()
    
    # Step 3: Perform clustering based on method
    best_score = -1
    best_labels = None
    best_n_clusters = None
    best_param = None
    best_column_name = None
    clustering_stats = []
    all_clustering_results = {}  # Store all clustering results: {column_name: labels}
    
    if clustering_method.lower() == "butina":
        # Butina clustering
        if distance_threshold_range is None:
            raise ValueError("distance_threshold_range must be provided for Butina clustering")
        
        print(f"Performing Butina clustering with distance thresholds: {distance_threshold_range}")
        
        for dist_thresh in distance_threshold_range:
            try:
                # Butina clustering expects upper triangle of distance matrix as list
                # dists_list = []
                # n_mols = len(fingerprints)
                # for i in range(n_mols):
                #     for j in range(i+1, n_mols):
                #         dists_list.append(distance_matrix[i, j])
                
                dists_list = tanimoto_distance_matrix(fingerprints)
                n_mols = len(fingerprints)
                
                # Perform Butina clustering
                clusters = Butina.ClusterData(dists_list, n_mols, dist_thresh, isDistData=True)
                
                # Convert cluster tuples to labels array
                labels = np.full(n_mols, -1, dtype=int)
                for cluster_id, cluster_members in enumerate(clusters):
                    for member_idx in cluster_members:
                        labels[member_idx] = cluster_id
                
                n_clusters = len(clusters)
                
                # Calculate silhouette score
                if n_clusters > 1:
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                else:
                    score = -1.0
                
                # Calculate cluster member distribution
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                member_distribution = '-'.join(map(str, sorted(counts)))
                
                # Store clustering result
                column_name = f'butina-{dist_thresh}'
                all_clustering_results[column_name] = labels.copy()
                
                # Record statistics
                clustering_stats.append({
                    'clustering_method': 'butina',
                    'distance_threshold': dist_thresh,
                    'n_clusters': n_clusters,
                    'member_distribution': member_distribution,
                    'silhouette_score': score,
                    'selected_as_best': 0,
                    'column_name': column_name
                })
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
                    best_param = dist_thresh
                    best_column_name = column_name
                
                print(f"  dist_thresh={dist_thresh}: n_clusters={n_clusters}, silhouette_score={score:.4f}")
                
            except Exception as e:
                print(f"  dist_thresh={dist_thresh}: Error - {e}")
                continue
    
    elif clustering_method.lower() == "agglomerative":
        # Agglomerative clustering
        
        if n_clusters_range is not None and distance_threshold_range is None:
            # Use n_clusters parameter
            print(f"Performing Agglomerative clustering with n_clusters range: {n_clusters_range}")
            min_clusters, max_clusters = n_clusters_range
            max_clusters = min(max_clusters, len(fingerprints) - 1)
            
            if max_clusters < min_clusters:
                print(f"Warning: Not enough samples for clustering range. Using {len(fingerprints)} clusters")
                max_clusters = min_clusters = max(2, len(fingerprints) // 2)
            
            for n_clusters in range(min_clusters, max_clusters + 1):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='precomputed',
                        linkage='complete'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    
                    # Calculate silhouette score
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    
                    # Calculate cluster member distribution
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    member_distribution = '-'.join(map(str, sorted(counts)))
                    
                    # Store clustering result
                    column_name = f'agglomerative-{n_clusters}'
                    all_clustering_results[column_name] = labels.copy()
                    
                    # Record statistics
                    clustering_stats.append({
                        'clustering_method': 'agglomerative',
                        'n_clusters': n_clusters,
                        'distance_threshold': None,
                        'member_distribution': member_distribution,
                        'silhouette_score': score,
                        'selected_as_best': 0,
                        'column_name': column_name
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        best_labels = labels
                        best_param = n_clusters
                        best_column_name = column_name
                    
                    print(f"  n_clusters={n_clusters}: silhouette_score={score:.4f}")
                        
                except Exception as e:
                    print(f"  n_clusters={n_clusters}: Error - {e}")
                    continue
        
        elif distance_threshold_range is not None:
            # Use distance_threshold parameter
            print(f"Performing Agglomerative clustering with distance thresholds: {distance_threshold_range}")
            
            for dist_thresh in distance_threshold_range:
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=dist_thresh,
                        metric='precomputed',
                        linkage='complete'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    
                    n_clusters = len(np.unique(labels))
                    
                    # Calculate silhouette score
                    if n_clusters > 1:
                        score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    else:
                        score = -1.0
                    
                    # Calculate cluster member distribution
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    member_distribution = '-'.join(map(str, sorted(counts)))
                    
                    # Store clustering result
                    column_name = f'agglomerative-{dist_thresh}'
                    all_clustering_results[column_name] = labels.copy()
                    
                    # Record statistics
                    clustering_stats.append({
                        'clustering_method': 'agglomerative',
                        'n_clusters': n_clusters,
                        'distance_threshold': dist_thresh,
                        'member_distribution': member_distribution,
                        'silhouette_score': score,
                        'selected_as_best': 0,
                        'column_name': column_name
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        best_labels = labels
                        best_param = dist_thresh
                        best_column_name = column_name
                    
                    print(f"  dist_thresh={dist_thresh}: n_clusters={n_clusters}, silhouette_score={score:.4f}")
                        
                except Exception as e:
                    print(f"  dist_thresh={dist_thresh}: Error - {e}")
                    continue
        else:
            raise ValueError("Either n_clusters_range or distance_threshold_range must be provided for agglomerative clustering")
    
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}. Use 'agglomerative' or 'butina'")
    
    # Update the 'selected_as_best' flag for the actual best clustering
    for stat in clustering_stats:
        if stat['silhouette_score'] == best_score:
            stat['selected_as_best'] = 1
    
    if best_labels is None:
        raise ValueError("Clustering failed: could not find valid clustering solution")
    
    print(f"\nOptimal clustering: {best_n_clusters} clusters (silhouette score: {best_score:.4f})")
    if clustering_method.lower() == "butina":
        print(f"Best distance threshold: {best_param}")
    elif distance_threshold_range is not None:
        print(f"Best distance threshold: {best_param}")
    else:
        print(f"Best n_clusters: {best_param}")
    
    # Step 4: Assign ALL cluster indices to DataFrame as separate columns
    # df_result was already initialized earlier
    
    # First, assign all clustering results as separate columns
    for column_name, labels in all_clustering_results.items():
        # Initialize column with -1 for invalid SMILES
        df_result[column_name] = -1
        # Assign cluster labels to valid indices
        for i, idx in enumerate(valid_indices):
            df_result.loc[idx, column_name] = int(labels[i])
    
    # Assign best clustering result to 'Cluster Index' column
    df_result['Cluster Index'] = -1  # Initialize with -1 for invalid SMILES
    for i, idx in enumerate(valid_indices):
        df_result.loc[idx, 'Cluster Index'] = int(best_labels[i])
    
    # Create df_valid with only valid molecules (for splitting)
    df_valid = df_result[df_result['Cluster Index'] >= 0].copy()
    
    if len(df_valid) == 0:
        raise ValueError("No valid molecules after clustering")
    
    # Report clustering statistics
    cluster_counts = df_valid['Cluster Index'].value_counts().sort_index()
    print("\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} molecules ({count/len(df_valid)*100:.2f}%)")
    
    invalid_count = (df_result['Cluster Index'] == -1).sum()
    if invalid_count > 0:
        print(f"\nWarning: {invalid_count} molecules could not be clustered (invalid SMILES) - excluded from splits")
    
    # Step 5: Split data based on clusters
    print(f"\nSplitting data (train: {train_frac:.1%}, valid: {valid_frac:.1%}, test: {test_frac:.1%})...")
    
    # Set random seed
    np.random.seed(seed)
    
    # Get unique cluster IDs and their sizes
    cluster_sizes = df_valid.groupby('Cluster Index').size().to_dict()
    cluster_ids = list(cluster_sizes.keys())
    total_samples = len(df_valid)
    
    # Randomly shuffle cluster IDs
    np.random.shuffle(cluster_ids)
    
    # Select clusters for test set to approximate test_frac
    test_clusters = []
    test_count = 0
    target_test_count = int(total_samples * test_frac)
    
    for cluster_id in cluster_ids:
        if test_count < target_test_count:
            test_clusters.append(cluster_id)
            test_count += cluster_sizes[cluster_id]
        else:
            break
    
    # Remaining clusters for train/valid split
    train_valid_clusters = [c for c in cluster_ids if c not in test_clusters]
    
    # Create test set (entire clusters)
    test_df = df_valid[df_valid['Cluster Index'].isin(test_clusters)].copy()
    train_valid_df = df_valid[df_valid['Cluster Index'].isin(train_valid_clusters)].copy()
    
    # Randomly split train_valid into train and valid
    train_valid_shuffled = train_valid_df.sample(frac=1, random_state=seed)
    
    # Calculate train size from remaining data
    train_size = int(len(train_valid_shuffled) * train_frac / (train_frac + valid_frac))
    
    train_df = train_valid_shuffled.iloc[:train_size].copy()
    valid_df = train_valid_shuffled.iloc[train_size:].copy()
    
    # Report split statistics
    print("\nSplit results:")
    print(f"  Train: {len(train_df)} molecules ({len(train_df)/total_samples*100:.2f}%)")
    print(f"  Valid: {len(valid_df)} molecules ({len(valid_df)/total_samples*100:.2f}%)")
    print(f"  Test:  {len(test_df)} molecules ({len(test_df)/total_samples*100:.2f}%)")
    print(f"  Test clusters: {sorted(test_clusters)}")
    print(f"  Train/Valid clusters: {sorted(train_valid_clusters)}")
    
    # Verify no cluster overlap between test and train/valid
    test_cluster_set = set(test_df['Cluster Index'].unique())
    train_cluster_set = set(train_df['Cluster Index'].unique())
    valid_cluster_set = set(valid_df['Cluster Index'].unique())
    
    assert len(test_cluster_set & train_cluster_set) == 0, "Test and train clusters overlap!"
    assert len(test_cluster_set & valid_cluster_set) == 0, "Test and valid clusters overlap!"
    
    print("\n✓ Cluster-based split successful: test clusters are completely held out")
    
    # Create clustering statistics DataFrame
    clustering_stats_df = pd.DataFrame(clustering_stats)
    
    # Return df_result as df_valid_full (contains all clustering columns)
    return train_df, valid_df, test_df, clustering_stats_df, df_result
