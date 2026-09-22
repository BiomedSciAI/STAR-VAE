from bmfm_sm_gen.utils.rediscovery_evaluation import *
from bmfm_sm_gen.utils.scaffold_utils import *
import pandas as pd
import argparse
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np

def compute_cluster_proportions_by_threshold(clustering_results, similarity_thresholds, num_train_total, num_heldout_total):
    train_proportions = {}
    heldout_proportions = {}
    train_best_clusters_by_threshold = {t: set() for t in similarity_thresholds}
    heldout_best_clusters_by_threshold = {t: set() for t in similarity_thresholds}

    for r in clustering_results:
        if r['cluster_origin'] is not None and r['best_cluster_id'] is not None:
            best_sim = float(r['best_similarity_score']) if r['best_similarity_score'] is not None else 0.0
            cluster_id = r['best_cluster_id']
            origin = r['cluster_origin']

            for threshold in similarity_thresholds:
                if best_sim >= threshold:
                    if origin == 'train':
                        train_best_clusters_by_threshold[threshold].add(cluster_id)
                    elif origin == 'heldout':
                        heldout_best_clusters_by_threshold[threshold].add(cluster_id)

    for threshold in similarity_thresholds:
        num_train_best = len(train_best_clusters_by_threshold[threshold])
        num_heldout_best = len(heldout_best_clusters_by_threshold[threshold])

        train_proportions[threshold] = (
            num_train_best / num_train_total if num_train_total > 0 else float('nan')
        )
        heldout_proportions[threshold] = (
            num_heldout_best / num_heldout_total if num_heldout_total > 0 else float('nan')
        )

    return train_proportions, heldout_proportions


def find_cluster_centroids(df, smiles_col, cluster_col='cluster_index'):
    """
    Find the centroid (medoid) for each cluster based on Tanimoto similarity.
    
    Args:
        df: DataFrame containing molecules
        smiles_col: Name of the column containing SMILES strings
        cluster_col: Name of the column containing cluster assignments
    
    Returns:
        DataFrame with added 'centroid' column where centroid items have their cluster_index
        and non-centroid items have -1
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize centroid column with -1
    df['centroid'] = -1
    
    # Get unique cluster IDs
    unique_clusters = df[cluster_col].unique()
    
    print(f"Finding centroids for {len(unique_clusters)} clusters...")
    
    for cluster_id in unique_clusters:
        # Get indices of molecules in this cluster
        cluster_mask = df[cluster_col] == cluster_id
        cluster_indices = df.index[cluster_mask].tolist()
        
        if len(cluster_indices) == 1:
            # Single molecule cluster - it is the centroid by default
            df.at[cluster_indices[0], 'centroid'] = cluster_id
            continue
        
        # Get SMILES for this cluster
        cluster_smiles = df.loc[cluster_indices, smiles_col].tolist()
        
        # Generate Morgan fingerprints for the cluster
        cluster_fps = []
        valid_indices = []
        
        for idx, smi in zip(cluster_indices, cluster_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                cluster_fps.append(fp)
                valid_indices.append(idx)
            else:
                print(f"Warning: Could not parse SMILES at index {idx}: {smi}")
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid molecules in cluster {cluster_id}")
            continue
        
        if len(valid_indices) == 1:
            # Only one valid molecule - it is the centroid
            df.at[valid_indices[0], 'centroid'] = cluster_id
            continue
        
        # Find the molecule with highest sum of similarities to all others
        max_sim_sum = -1
        centroid_idx = valid_indices[0]
        
        for i, fp_i in enumerate(cluster_fps):
            # Calculate Tanimoto similarity to all other molecules in cluster
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, cluster_fps)
            
            # Sum of similarities (subtract 1.0 because similarity to self is always 1.0)
            current_sim_sum = sum(sims) - 1.0
            
            if current_sim_sum > max_sim_sum:
                max_sim_sum = current_sim_sum
                centroid_idx = valid_indices[i]
        
        # Assign the cluster_id to the centroid
        df.at[centroid_idx, 'centroid'] = cluster_id
    
    print(f"Centroids found: {(df['centroid'] != -1).sum()} out of {len(unique_clusters)} clusters")
    
    return df


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--train_csv',
        default=os.environ.get("CHEMBL_DATA_DIR", "data/chembl"),
        type=str,
        help='Input file that contains generated smiles.'
    )
    
    parser.add_argument(
        '--test_csv',
        default=os.environ.get("CHEMBL_DATA_DIR", "data/chembl"),
        type=str,
        help='Output directory.'
    )
    
    parser.add_argument(
        '--generated_csv',
        default=os.environ.get("OUTPUT_DIR", "outputs"),
        type=str,
        help='Output directory.'
    )
    
    parser.add_argument(
        '--output_dir',
        default=os.environ.get("OUTPUT_DIR", "outputs"),
        type=str,
        help='Output directory.'
    )
    
    parser.add_argument(
        '--time_col',
        default='Document Year',
        type=str,
        help='Name of the time/year column for temporal analysis (e.g., "Document Year", "New Document Year"). If column does not exist, temporal metrics will be skipped.'
    )
    
    parser.add_argument(
        '--similarity_thresholds',
        default='0.1,0.2,0.3,0.4,0.5,0.6',
        type=str,
        help='Comma-separated list of similarity thresholds for clustering metrics (e.g., "0.1,0.2,0.3,0.4,0.5,0.6")'
    )
    
    parser.add_argument(
        '--parallel_calculation',
        action='store_true',
        help='Enable parallel calculation for computationally expensive rediscovery methods.'
    )
    
    args = parser.parse_args()
    
    # Parse similarity thresholds
    similarity_thresholds = [float(t.strip()) for t in args.similarity_thresholds.split(',')]
    num_workers = max(1, os.cpu_count() or 1)
    if args.parallel_calculation:
        print(f"Parallel calculation enabled. Using {num_workers} CPU cores.")
    else:
        print("Parallel calculation disabled. Using standard serial execution.")

    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)
    
    # Check if time column exists in both dataframes
    has_time_col = args.time_col in df_train.columns and args.time_col in df_test.columns
    
    if has_time_col:
        print(f"Time column '{args.time_col}' found in both train and test datasets. Temporal metrics will be calculated.")
    else:
        print(f"Warning: Time column '{args.time_col}' not found in train and/or test datasets.")
        print(f"  - In train: {args.time_col in df_train.columns}")
        print(f"  - In test: {args.time_col in df_test.columns}")
        print("Temporal metrics will be skipped.")
    
    train_interscaffsim = interscaffoldssimilarity(df_train['scaffold'])
    print(f'train_interscaffsim: {train_interscaffsim}')
    test_interscaffsim = interscaffoldssimilarity(df_test['scaffold'])
    print(f'test_interscaffsim: {test_interscaffsim}')
    intrascaffsim_train_test = crossscaffoldsimilarity(df_train['scaffold'], df_test['scaffold'])
    print(f'intrascaffsim_train_test: {intrascaffsim_train_test}')
    
    redis_results = []
    redis_time_results = []
    redis_best_similar = []
    
    
    for filepath in [args.train_csv, args.generated_csv]:
        df_ = pd.read_csv(filepath)
        colname = 'smiles'
        if "generated_smiles" in df_.columns:
            colname = 'generated_smiles'
        elif "smiles" in df_.columns:
            colname = 'smiles'
        elif "Smiles" in df_.columns:
            colname = 'Smiles'
        print(f'colname: {colname}; filepath: {filepath}')
        
        # Determine column name for test dataset separately
        test_colname = 'smiles'
        if "smiles" in df_test.columns:
            test_colname = 'smiles'
        elif "Smiles" in df_test.columns:
            test_colname = 'Smiles'
        elif "generated_smiles" in df_test.columns:
            test_colname = 'generated_smiles'
        print(f'test_colname: {test_colname}')
        
        base = os.path.basename(filepath)
        projname, _ = os.path.splitext(base)
        # projname, _ = os.path.splitext(args.generated_csv)
        # projname = '-'.join(projname.split('-')[:-1])
        
        # Initialize RediscoveryEvaluation with or without time column
        if has_time_col:
            redis = RediscoveryEvaluation(
                df_[colname],
                df_test[test_colname],
                df_test[args.time_col],
                name=projname,
                parallel_calculation=args.parallel_calculation,
                num_workers=num_workers,
            )
        else:
            # Pass empty array when time column doesn't exist
            redis = RediscoveryEvaluation(
                df_[colname],
                df_test[test_colname],
                [],
                name=projname,
                parallel_calculation=args.parallel_calculation,
                num_workers=num_workers,
            )
        
        common_smiles, n_common, precision, recall = redis.exact_rediscovery()
        max_pairs, mean_score, proportion_above, count_above = redis.near_neighbor_rediscovery()
        
        # Determine column name for train dataset separately
        train_colname = 'smiles'
        if "smiles" in df_train.columns:
            train_colname = 'smiles'
        elif "Smiles" in df_train.columns:
            train_colname = 'Smiles'
        elif "generated_smiles" in df_train.columns:
            train_colname = 'generated_smiles'
        
        # Calculate temporal metrics only if time column exists
        if has_time_col:
            enrichment_factor_future, p_future_gen, p_future_data = redis.temporal_enrichment_analysis(
                df_train[train_colname],
                df_train[args.time_col],
                int(min(df_train[args.time_col].max() + 1, df_test[args.time_col].min()))
            )
            best_yearg, bounds, counts, means, props = redis.time_potency_rediscovery_matrix(
                list(range(int(min(df_train[args.time_col].max() + 1, df_test[args.time_col].min())),
                          int(df_test[args.time_col].max()) + 1, 1)),
                .25
            )
        else:
            # Set temporal metrics to None/NaN when time column doesn't exist
            enrichment_factor_future = float('nan')
            p_future_gen = float('nan')
            p_future_data = float('nan')
            best_yearg = None
            bounds = []
            counts = []
            means = []
            props = []
        
        common_scaff_ratio, common_scaff_ratio_over_generated, common_scaff_count_g_h, scaff_count_g, scaff_count_h, common_scaff_list = redis.scaffold_rediscovery()
        cs_ht, cs_gt, h2g, cs_ht_list, cs_gt_list = redis.scaffold_ratio_against_train(df_train[train_colname])
        diversity_ratio, diversity_gen, diversity_h = redis.diversity()
        uniq_ratio, uniq_gen, uniq_heldout = redis.uniqueness()
        gen_interscaffsim = interscaffoldssimilarity(df_['scaffold'])
        print(f'train_interscaffsim: {gen_interscaffsim}')
        intrascaffsim_gen_train = crossscaffoldsimilarity(df_['scaffold'], df_train['scaffold'])
        print(f'train_interscaffsim: {intrascaffsim_gen_train}')
        intrascaffsim_gen_test = crossscaffoldsimilarity(df_['scaffold'], df_test['scaffold'])
        print(f'intrascaffsim_gen_test: {intrascaffsim_gen_test}')
        
        # Generate clustering metrics if cluster columns exist
        # Initialize dictionaries to store metrics for each threshold
        heldout_proportions = {}
        train_proportions = {}
        heldout_proportions_centroid = {}
        train_proportions_centroid = {}
        unique_train_clusters = float('nan')
        unique_heldout_clusters = float('nan')
        unique_generated_clusters = float('nan')
        
        if 'cluster_index' in df_train.columns and 'cluster_index' in df_test.columns:
            # Check if centroid column exists, if not create a dummy one with all -1 values
            has_centroid = 'centroid' in df_train.columns and 'centroid' in df_test.columns
            
            if not has_centroid:
                print(f'Warning: centroid column not found for {projname}, computing centroids using Tanimoto similarity...')
                # Find centroids for train set
                df_train = find_cluster_centroids(df_train, train_colname, cluster_col='cluster_index')
                # Find centroids for test set
                df_test = find_cluster_centroids(df_test, test_colname, cluster_col='cluster_index')
            
            print(f'Generating clustering metrics for {projname}...')
            unique_train_clusters, unique_heldout_clusters, unique_generated_clusters = redis.unique_cluster_counts(
                generated_cluster_index=df_['cluster_index'] if 'cluster_index' in df_.columns else [],
                heldout_cluster_index=df_test['cluster_index'],
                train_cluster_index=df_train['cluster_index'],
            )
            
            # Generate clustering results once with the highest threshold for saving
            clustering_results = redis.generate_clustering_metrics(
                train_smiles=df_train[train_colname],
                train_cluster_index=df_train['cluster_index'],
                train_centroid=df_train['centroid'],
                heldout_cluster_index=df_test['cluster_index'],
                heldout_centroid=df_test['centroid'],
                similarity_threshold=max(similarity_thresholds)
            )
            
            df_clustering = pd.DataFrame(clustering_results)
            df_clustering.to_csv(os.path.join(args.output_dir, f"{projname}-redis-best-clustering.csv"), index=False)
            
            # Get all unique cluster indices in train and heldout sets
            all_train_clusters = set(df_train['cluster_index'].dropna().unique())
            all_heldout_clusters = set(df_test['cluster_index'].dropna().unique())
            num_train_total = len(all_train_clusters)
            num_heldout_total = len(all_heldout_clusters)
            
            train_proportions, heldout_proportions = compute_cluster_proportions_by_threshold(
                clustering_results,
                similarity_thresholds,
                num_train_total,
                num_heldout_total,
            )

            for threshold in similarity_thresholds:
                num_train_best = int(round(train_proportions[threshold] * num_train_total)) if num_train_total > 0 and not pd.isna(train_proportions[threshold]) else 0
                num_heldout_best = int(round(heldout_proportions[threshold] * num_heldout_total)) if num_heldout_total > 0 and not pd.isna(heldout_proportions[threshold]) else 0

                if num_train_total > 0:
                    print(f'Train proportion (similarity >= {threshold}): {train_proportions[threshold]:.4f} ({num_train_best}/{num_train_total})')

                if num_heldout_total > 0:
                    print(f'Heldout proportion (similarity >= {threshold}): {heldout_proportions[threshold]:.4f} ({num_heldout_best}/{num_heldout_total})')
            
            # Generate centroid-only clustering metrics only if we have actual centroids
            if has_centroid:
                print(f'Generating centroid-only clustering metrics for {projname}...')
                clustering_results_centroid = redis.generate_clustering_metrics_for_centroid(
                    train_smiles=df_train[train_colname],
                    train_cluster_index=df_train['cluster_index'],
                    train_centroid=df_train['centroid'],
                    heldout_cluster_index=df_test['cluster_index'],
                    heldout_centroid=df_test['centroid'],
                    similarity_threshold=max(similarity_thresholds)
                )
                
                df_clustering_centroid = pd.DataFrame(clustering_results_centroid)
                df_clustering_centroid.to_csv(os.path.join(args.output_dir, f"{projname}-redis-best-clustering-centroid.csv"), index=False)
                
                # Get all unique cluster indices from centroid molecules only
                all_train_clusters_centroid = set(df_train[df_train['centroid'] >= 0]['cluster_index'].dropna().unique())
                all_heldout_clusters_centroid = set(df_test[df_test['centroid'] >= 0]['cluster_index'].dropna().unique())
                num_train_total_centroid = len(all_train_clusters_centroid)
                num_heldout_total_centroid = len(all_heldout_clusters_centroid)
                
                train_proportions_centroid, heldout_proportions_centroid = compute_cluster_proportions_by_threshold(
                    clustering_results_centroid,
                    similarity_thresholds,
                    num_train_total_centroid,
                    num_heldout_total_centroid,
                )

                for threshold in similarity_thresholds:
                    num_train_best_centroid = int(round(train_proportions_centroid[threshold] * num_train_total_centroid)) if num_train_total_centroid > 0 and not pd.isna(train_proportions_centroid[threshold]) else 0
                    num_heldout_best_centroid = int(round(heldout_proportions_centroid[threshold] * num_heldout_total_centroid)) if num_heldout_total_centroid > 0 and not pd.isna(heldout_proportions_centroid[threshold]) else 0

                    if num_train_total_centroid > 0:
                        print(f'Train proportion (centroid, similarity >= {threshold}): {train_proportions_centroid[threshold]:.4f} ({num_train_best_centroid}/{num_train_total_centroid})')

                    if num_heldout_total_centroid > 0:
                        print(f'Heldout proportion (centroid, similarity >= {threshold}): {heldout_proportions_centroid[threshold]:.4f} ({num_heldout_best_centroid}/{num_heldout_total_centroid})')
            else:
                print(f'Skipping centroid-only clustering metrics (no centroid column)')
                # Initialize empty dictionaries for centroid metrics
                for threshold in similarity_thresholds:
                    heldout_proportions_centroid[threshold] = float('nan')
                    train_proportions_centroid[threshold] = float('nan')
        else:
            print(f'Warning: cluster_index column not found in train or test data for {projname}')
            # Initialize empty dictionaries if cluster columns don't exist
            for threshold in similarity_thresholds:
                heldout_proportions[threshold] = float('nan')
                train_proportions[threshold] = float('nan')
                heldout_proportions_centroid[threshold] = float('nan')
                train_proportions_centroid[threshold] = float('nan')
        
        # Build results dictionary - conditionally include temporal metrics
        res = {
            'Method': redis.name,
            'Exact Discovery | Precision': precision,
            'Exact Discovery | Recall': recall,
            'Nearest Neighbor Discovery | Mean Sim Score': mean_score,
        }
        
        # Add temporal metrics only if time column exists
        if has_time_col:
            res.update({
                'Temporal Enrichment Analysis | Enrichment Factor Future': enrichment_factor_future,
                'Temporal Enrichment Analysis | P(Future, Generated)': p_future_gen,
                'Temporal Enrichment Analysis | P(Future, Data)': p_future_data,
            })
        
        # Add remaining metrics
        res.update({
            'Clusters | #Unique(Train)': unique_train_clusters,
            'Clusters | #Unique(Heldout)': unique_heldout_clusters,
            'Clusters | #Unique(Generated)': unique_generated_clusters,
            'Scaffold Rediscovery | Scaff(Generated) ∩ Scaff(Heldout) / Scaff(Heldout)': common_scaff_ratio,
            'Scaffold Rediscovery | Scaff(Generated) ∩ Scaff(Heldout) / Scaff(Generated)': common_scaff_ratio_over_generated,
            'Scaffold Rediscovery | Scaff(Generated) ∩ Scaff(Train) / Scaff(Train)': cs_gt,
            'Scaffold Rediscovery | Scaff(Heldout) ∩ Scaff(Train) / Scaff(Train)': cs_ht,
            'Scaffold Similarity | InterSim(Scaff(Generated))': gen_interscaffsim,
            'Scaffold Similarity | InterSim(Scaff(Train))': train_interscaffsim,
            'Scaffold Similarity | InterSim(Scaff(Heldout))': test_interscaffsim,
            'Scaffold Similarity | CrossSim(Scaff(Generated) & Scaff(Train))': intrascaffsim_gen_train,
            'Scaffold Similarity | CrossSim(Scaff(Generated) & Scaff(Heldout))': intrascaffsim_gen_test,
            'Scaffold Similarity | CrossSim(Scaff(Train) & Scaff(Heldout))': intrascaffsim_train_test,
            'Diversity | D(Gen) / D(Heldout)': diversity_ratio,
            'Diversity | D(Gen)': diversity_gen,
            'Diversity | D(Heldout)': diversity_h,
            'Uniqueness | U(Gen) / U(Heldout)': uniq_ratio,
            'Uniqueness | U(Gen)': uniq_gen,
            'Uniqueness | U(Heldout)': uniq_heldout,
        })
        
        # Add clustering metrics for all thresholds
        for threshold in similarity_thresholds:
            heldout_prop = heldout_proportions.get(threshold, float('nan'))
            train_prop = train_proportions.get(threshold, float('nan'))
            heldout_prop_centroid = heldout_proportions_centroid.get(threshold, float('nan'))
            train_prop_centroid = train_proportions_centroid.get(threshold, float('nan'))
            
            res[f'Clustering | Heldout Proportion (sim >= {threshold})'] = f'{heldout_prop:.3f}' if not pd.isna(heldout_prop) else 'N/A'
            res[f'Clustering | Train Proportion (sim >= {threshold})'] = f'{train_prop:.3f}' if not pd.isna(train_prop) else 'N/A'
            res[f'Clustering Centroid | Heldout Proportion (sim >= {threshold})'] = f'{heldout_prop_centroid:.3f}' if not pd.isna(heldout_prop_centroid) else 'N/A'
            res[f'Clustering Centroid | Train Proportion (sim >= {threshold})'] = f'{train_prop_centroid:.3f}' if not pd.isna(train_prop_centroid) else 'N/A'
        
        redis_results.append(res)

        # Add time potency results only if time column exists
        if has_time_col:
            for bound, count in zip(bounds, counts):
                res_time_potency = {'Method': redis.name, 'Time Range': f'{bound[0]}-{bound[1]},', 'No. Molecules': count}
                redis_time_results.append(res_time_potency)
            
        redis_best_similar = redis_best_similar + redis.find_max_similar_heldout_all(df_test)

    # Save results after processing all files
    df_redis_results = pd.DataFrame(redis_results)
    df_redis_best_similar = pd.DataFrame(redis_best_similar)
    
    df_redis_results.to_csv(os.path.join(args.output_dir, f"{projname}-redis.csv"), index=False)
    df_redis_best_similar.to_csv(os.path.join(args.output_dir, f"{projname}-redis-best-similar.csv"), index=False)
    
    # Save time results only if time column exists and we have time results
    if has_time_col and redis_time_results:
        df_redis_time_results = pd.DataFrame(redis_time_results)
        df_redis_time_results.to_csv(os.path.join(args.output_dir, f"{projname}-redis-timeresults.csv"), index=False)
        print(f"Saved time results to: {os.path.join(args.output_dir, f'{projname}-redis-timeresults.csv')}")
    else:
        print("Skipped saving time results (no time column or no time data)")


if __name__ == "__main__":
    main()