from typing import Iterable, Tuple, Set, Any, List, Dict, Union
import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def _parallel_best_match_worker(args):
    gen_smi, fp_gen, target_smiles, target_fp_items = args

    if fp_gen is None:
        return gen_smi, (None, 0.0, None)

    best_target_smi = None
    best_similarity = -1.0
    best_target_index = None

    for target_index, target_smi in enumerate(target_smiles):
        fp_target = target_fp_items.get(target_smi)
        if fp_target is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(fp_gen, fp_target)
        if similarity > best_similarity:
            best_similarity = similarity
            best_target_smi = target_smi
            best_target_index = target_index

    return gen_smi, (best_target_smi, best_similarity, best_target_index)

class RediscoveryEvaluation:
    def __init__(
        self,
        generated_smiles: Iterable[str],
        heldout_smiles: Iterable[str],
        heldout_year: Iterable[float | int] = None,
        name: str = None,
        parallel_calculation: bool = False,
        num_workers: int = None,
    ) -> None:
        self.name = name
        self.generated_smiles = generated_smiles
        self.heldout_smiles = heldout_smiles
        self.parallel_calculation = parallel_calculation
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() or 1)
        if isinstance(heldout_year, pd.Series):
            self.healdout_year = heldout_year.tolist()
        else:
            self.healdout_year = list(heldout_year)
        
        self.set_g = self.canonicalize(generated_smiles)
        self.set_h = self.canonicalize(heldout_smiles)
        
        self.fp_dict_g = self.build_fp_dict(generated_smiles)
        self.fp_dict_h = self.build_fp_dict(heldout_smiles)
        
        # Cache for similarity calculations to avoid redundant computations
        self._sim_cache_g_to_h = None  # Generated to Heldout similarities
        self._sim_cache_g_to_h_computed = False

        # Cache scaffold sets because multiple metrics reuse them
        self._scaffold_set_g = None
        self._scaffold_set_h = None
        
    def _compute_g_to_h_similarities(self):
        """
        Compute and cache similarity matrix between generated and heldout molecules.
        Returns a dictionary mapping gen_smi -> (best_held_smi, best_similarity, best_h_index).
        """
        if self._sim_cache_g_to_h_computed:
            return self._sim_cache_g_to_h

        heldout_smiles_list = list(self.heldout_smiles)
        best_matches = {}

        if self.parallel_calculation and len(self.generated_smiles) > 1 and len(heldout_smiles_list) > 0:
            worker_args = [
                (gen_smi, self.fp_dict_g.get(gen_smi), heldout_smiles_list, self.fp_dict_h)
                for gen_smi in self.generated_smiles
            ]
            with Pool(processes=self.num_workers) as pool:
                for gen_smi, best_match in pool.map(_parallel_best_match_worker, worker_args):
                    best_matches[gen_smi] = best_match
        else:
            for gen_smi in self.generated_smiles:
                if gen_smi not in self.fp_dict_g:
                    best_matches[gen_smi] = (None, 0.0, None)
                    continue

                fp_gen = self.fp_dict_g[gen_smi]
                best_held_smi = None
                best_similarity = -1.0
                best_h_index = None

                for h_index, held_smi in enumerate(heldout_smiles_list):
                    if held_smi not in self.fp_dict_h:
                        continue
                    fp_held = self.fp_dict_h[held_smi]
                    similarity = DataStructs.TanimotoSimilarity(fp_gen, fp_held)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_held_smi = held_smi
                        best_h_index = h_index

                best_matches[gen_smi] = (best_held_smi, best_similarity, best_h_index)

        self._sim_cache_g_to_h = best_matches
        self._sim_cache_g_to_h_computed = True
        return best_matches
        
    def exact_rediscovery(self):
        
        common_smiles = self.set_g & self.set_h
        n_common = len(common_smiles)

        precision = n_common / len(self.set_g) if self.set_g else float('nan')
        recall = n_common / len(self.set_h) if self.set_h else float('nan')

        return common_smiles, n_common, precision, recall
    
    def near_neighbor_rediscovery(self, threshold=.7):
        # Use cached similarities
        best_matches = self._compute_g_to_h_similarities()
        
        max_pairs = []
        for gen_smi in self.generated_smiles:
            best_held_smi, best_similarity, _ = best_matches.get(gen_smi, (None, 0.0, None))
            if best_held_smi is not None:
                max_pairs.append((gen_smi, best_similarity))
                
        if not max_pairs:
            mean_score = float('nan')
            proportion_above = float('nan')
            count_above = 0
        else:
            scores = np.array([s for _, s in max_pairs])
            mean_score = float(scores.mean())
            count_above = int((scores > threshold).sum())
            proportion_above = count_above / len(self.fp_dict_h)
        return max_pairs, mean_score, proportion_above, count_above
    
    def temporal_enrichment_analysis(self, train_smiles: Iterable[str], train_year: Iterable[Any], cutoff_year: float | int):
        if isinstance(train_smiles, pd.Series):
            train_smiles_list = train_smiles.tolist()
        else:
            train_smiles_list = list(train_smiles)
            
        if isinstance(train_year, pd.Series):
            train_year = train_year.tolist()
        else:
            train_year = list(train_year)
        
        if isinstance(self.heldout_smiles, pd.Series):
            hslist = self.canonicalize_to_list(train_smiles_list + self.heldout_smiles.tolist())
        else:
            hslist = self.canonicalize_to_list(train_smiles_list + list(self.heldout_smiles))
        hsyear = train_year + self.healdout_year
        fp_dict_hslist = self.build_fp_dict(hslist)
        
        best_yearg = []
        for smig, fpg_val in self.fp_dict_g.items():
            best_smih, best_sim_score = None, -1.0
            for smih, fph_val in fp_dict_hslist.items():
                sim_score = DataStructs.TanimotoSimilarity(fph_val, fpg_val)
                if sim_score > best_sim_score:
                    best_sim_score = sim_score
                    best_smih = smih
                    best_indexh = hslist.index(best_smih)
            best_yearg.append(hsyear[best_indexh])
        
        all_year = train_year + self.healdout_year
        all_year_above_cutoff = [y for y in all_year if y >= cutoff_year]
        p_future_data = len(all_year_above_cutoff)/len(all_year)
        
        best_yearg_above_cutoff = [y for y in best_yearg if y >= cutoff_year]
        p_future_gen = len(best_yearg_above_cutoff)/len(best_yearg)
        
        enrichment_factor_future = p_future_gen/p_future_data
        return enrichment_factor_future, p_future_gen, p_future_data
    
    def scaffold_rediscovery(self):
        scaff_h = self.get_heldout_scaffold_set()
        scaff_g = self.get_generated_scaffold_set()
        common_scaff = scaff_h & scaff_g

        common_over_heldout = len(common_scaff) / len(scaff_h) if scaff_h else float('nan')
        common_over_generated = len(common_scaff) / len(scaff_g) if scaff_g else float('nan')

        return (
            common_over_heldout,
            common_over_generated,
            len(common_scaff),
            len(scaff_g),
            len(scaff_h),
            common_scaff,
        )
    
    def scaffold_ratio_against_train(self, train_smiles: Iterable[str]):
        scaff_h = self.get_heldout_scaffold_set()
        scaff_g = self.get_generated_scaffold_set()
        scaff_t = self.scaffold_set(train_smiles)
        
        cs_ht = len(scaff_h & scaff_t) / len(scaff_t)
        cs_gt = len(scaff_g & scaff_t) / len(scaff_t)
        h2g = cs_gt/cs_ht
        
        return cs_ht, cs_gt, h2g, scaff_h & scaff_t, scaff_g & scaff_t

    def unique_cluster_counts(
        self,
        generated_cluster_index: Union[List[int], pd.Series],
        heldout_cluster_index: Union[List[int], pd.Series],
        train_cluster_index: Union[List[int], pd.Series],
    ):
        if isinstance(generated_cluster_index, pd.Series):
            generated_cluster_index = list(set(generated_cluster_index.tolist()))
        else:
            generated_cluster_index = list(set(list(generated_cluster_index)))

        if isinstance(heldout_cluster_index, pd.Series):
            heldout_cluster_index = list(set(heldout_cluster_index.tolist()))
        else:
            heldout_cluster_index = list(set(list(heldout_cluster_index)))

        if isinstance(train_cluster_index, pd.Series):
            train_cluster_index = list(set(train_cluster_index.tolist()))
        else:
            train_cluster_index = list(set(list(train_cluster_index)))

        unique_generated_clusters = len({c for c in generated_cluster_index if pd.notna(c)})
        unique_heldout_clusters = len({c for c in heldout_cluster_index if pd.notna(c)})
        unique_train_clusters = len({c for c in train_cluster_index if pd.notna(c)})

        return unique_train_clusters, unique_heldout_clusters, unique_generated_clusters
    
    def time_potency_rediscovery_matrix(self, start_years: Iterable[Any] = [2015, 2020], cutoff_sim:float = .8):
        # Use cached similarities
        best_matches = self._compute_g_to_h_similarities()
        
        best_yearg = []
        for gen_smi in self.generated_smiles:
            best_held_smi, best_similarity, best_h_index = best_matches.get(gen_smi, (None, 0.0, None))
            if best_h_index is not None:
                best_yearg.append((self.healdout_year[best_h_index], best_similarity))
        
        counts = []
        means  = []
        props = []
        
        starts = sorted(start_years)

        bounds = []
        bounds.append((float('-inf'), starts[0]))
        for i in range(len(starts)-1):
            bounds.append((starts[i], starts[i+1]))
        bounds.append((starts[-1], float('inf')))

        for low, high in bounds:
            subset = [score for year, score in best_yearg if low <= year < high]
            counts.append(len(subset))
            means.append(float(np.mean(subset)) if subset else float('nan'))
            props.append(len([s for s in subset if s >= cutoff_sim])/len(subset) if subset else float('nan'))
        
        return best_yearg, bounds, counts, means, props
    
    def build_fp_dict(self, smiles_set: Iterable[str], radius:int=2, nbits:int=2048) -> dict:
        d = {}
        for smi in smiles_set:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            morgan_gen = GetMorganGenerator(radius=radius, fpSize=nbits)
            fp = morgan_gen.GetFingerprint(mol)
            d[smi] = fp
        return d
    
    def canonicalize(self, coll: Iterable[str]) -> Set[str]:
        if isinstance(coll, pd.Series):
            coll = coll.to_numpy()
        canonical = set()
        for smi in coll:
            if not isinstance(smi, str):
                continue
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical.add(canon_smi)
            except Exception:
                continue
        return canonical
    
    def canonicalize_to_list(self, coll: Iterable[str]) -> Iterable[str]:
        if isinstance(coll, pd.Series):
            coll = coll.to_numpy()
        canonical = []
        for smi in coll:
            if not isinstance(smi, str):
                canonical.append(smi)
                continue
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    canonical.append(smi)
                    continue
                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical.append(canon_smi)
            except Exception:
                canonical.append(smi)
                continue
        return canonical
    
    def scaffold_set(self, smiles_iter: Iterable[str]) -> Set[str]:
        scaffolds = set()
        for smi in smiles_iter:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            smi_scaffold = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
            scaffolds.add(smi_scaffold)
        return scaffolds

    def get_generated_scaffold_set(self) -> Set[str]:
        if self._scaffold_set_g is None:
            self._scaffold_set_g = self.scaffold_set(self.generated_smiles)
        return self._scaffold_set_g

    def get_heldout_scaffold_set(self) -> Set[str]:
        if self._scaffold_set_h is None:
            self._scaffold_set_h = self.scaffold_set(self.heldout_smiles)
        return self._scaffold_set_h
        
    def diversity(self):
        diversity_gen = self.fingerprint_diversity(self.fp_dict_g)
        diversity_h = self.fingerprint_diversity(self.fp_dict_h)
        diversity_ratio = diversity_gen / diversity_h
        return diversity_ratio, diversity_gen, diversity_h
    
    def fingerprint_diversity(self, fp_dict):
        total_distance = 0.0
        num_pairs = 0
        fp = list(fp_dict.values())
        for i in range(len(fp)):
            for j in range(i+1, len(fp)):
                dist = 1 - DataStructs.TanimotoSimilarity(fp[i], fp[j])
                total_distance += dist
                num_pairs += 1
        diversity = total_distance / num_pairs
        return diversity
    
    def uniqueness(self):
        uniq_gen = self.smiles_uniqueness_canonical(self.generated_smiles)
        uniq_heldout = self.smiles_uniqueness_canonical(self.heldout_smiles)
        uniq_ratio = uniq_gen / uniq_heldout
        return uniq_ratio, uniq_gen, uniq_heldout
    
    def smiles_uniqueness_canonical(self, smiles_list, isomeric=False):
        if isinstance(smiles_list, pd.Series):
            if smiles_list.empty:
                return 0.0
            smiles_list = smiles_list.tolist()
        else:
            if not smiles_list:
                return 0.0
            smiles_list = smiles_list

        canonical_smiles = set()
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol,isomericSmiles=isomeric,canonical=True)
            canonical_smiles.add(canonical)

        unique = len(canonical_smiles)
        total   = len(smiles_list)
        return (unique / total) * 100.0
    
    def find_max_similar_heldout_clinical_phase(
        self,
        heldout_molecule_max_phase: Union[List[Union[int, None]], pd.Series]
    ) -> List[Dict[str, Union[str, float]]]:
        
        if isinstance(heldout_molecule_max_phase, pd.Series):
            heldout_molecule_max_phase = heldout_molecule_max_phase.tolist()
        else:
            heldout_molecule_max_phase = list(heldout_molecule_max_phase)
        
        # Use cached similarities
        best_matches = self._compute_g_to_h_similarities()
        
        results = []
        for gen_smi in self.generated_smiles:
            best_held_smi, best_similarity, best_h_index = best_matches.get(gen_smi, (None, 0.0, None))
            
            if best_h_index is not None:
                best_clinical_phase: Any = heldout_molecule_max_phase[best_h_index]
            else:
                best_clinical_phase = None
            
            results.append({
                'max_similar_heldout_smiles': best_held_smi,
                'max_similarity_score': float(best_similarity) if best_held_smi is not None else 0.0,
                'max_clinical_phase': best_clinical_phase,
            })
        
        return results
    
    def find_max_similar_heldout_all(
        self,
        heldout_dataframe: pd.DataFrame
    ) -> List[Dict[str, Union[str, float, Any]]]:
        
        if len(heldout_dataframe) != len(self.heldout_smiles):
            raise ValueError(
                f"Length mismatch: heldout_dataframe has {len(heldout_dataframe)} rows "
                f"but self.heldout_smiles has {len(self.heldout_smiles)} entries"
            )
        
        # Use cached similarities
        best_matches = self._compute_g_to_h_similarities()
        
        results = []
        for gen_smi in self.generated_smiles:
            best_held_smi, best_similarity, best_h_index = best_matches.get(gen_smi, (None, 0.0, None))
            
            # Build result dictionary
            result_dict = {
                'name': self.name,
                'generated_smiles': gen_smi,
                'max_similar_heldout_smiles': best_held_smi,
                'max_similarity_score': float(best_similarity) if best_held_smi is not None else 0.0,
            }
            
            # Add all column values from the dataframe
            if best_h_index is not None:
                for col in heldout_dataframe.columns:
                    result_dict[col] = heldout_dataframe.iloc[best_h_index][col]
            else:
                for col in heldout_dataframe.columns:
                    result_dict[col] = None
            
            results.append(result_dict)
        
        return results
    
    def _compute_best_matches_against_smiles(
        self,
        target_smiles: List[str],
        target_fp_dict: dict,
    ):
        if self.parallel_calculation and len(self.generated_smiles) > 1 and len(target_smiles) > 0:
            worker_args = [
                (gen_smi, self.fp_dict_g.get(gen_smi), target_smiles, target_fp_dict)
                for gen_smi in self.generated_smiles
            ]
            with Pool(processes=self.num_workers) as pool:
                return dict(pool.map(_parallel_best_match_worker, worker_args))

        best_matches = {}
        for gen_smi in self.generated_smiles:
            if gen_smi not in self.fp_dict_g:
                best_matches[gen_smi] = (None, 0.0, None)
                continue

            fp_gen = self.fp_dict_g[gen_smi]
            best_target_smi = None
            best_similarity = -1.0
            best_target_index = None

            for target_index, target_smi in enumerate(target_smiles):
                if target_smi not in target_fp_dict:
                    continue
                fp_target = target_fp_dict[target_smi]
                similarity = DataStructs.TanimotoSimilarity(fp_gen, fp_target)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target_smi = target_smi
                    best_target_index = target_index

            best_matches[gen_smi] = (best_target_smi, best_similarity, best_target_index)

        return best_matches

    def generate_clustering_metrics(
        self,
        train_smiles: Iterable[str],
        train_cluster_index: Union[List[int], pd.Series],
        train_centroid: Union[List[int], pd.Series],
        heldout_cluster_index: Union[List[int], pd.Series],
        heldout_centroid: Union[List[int], pd.Series],
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Union[str, int, float, bool]]]:
        """
        Calculate clustering metrics for generated molecules.
        
        For each generated molecule, identify the best cluster assignment based on
        similarity scores with train + heldout molecules.
        
        Args:
            train_smiles: Training set SMILES
            train_cluster_index: Cluster indices for training molecules
            train_centroid: Centroid flags (0 or 1) for training molecules
            heldout_cluster_index: Cluster indices for heldout molecules
            heldout_centroid: Centroid flags (0 or 1) for heldout molecules
            similarity_threshold: Minimum similarity threshold for cluster assignment
            
        Returns:
            List of dictionaries containing clustering metrics for each generated molecule
        """
        # Convert inputs to lists
        if isinstance(train_smiles, pd.Series):
            train_smiles_list = train_smiles.tolist()
        else:
            train_smiles_list = list(train_smiles)
            
        if isinstance(train_cluster_index, pd.Series):
            train_cluster_index = train_cluster_index.tolist()
        else:
            train_cluster_index = list(train_cluster_index)
            
        if isinstance(train_centroid, pd.Series):
            train_centroid = train_centroid.tolist()
        else:
            train_centroid = list(train_centroid)
            
        if isinstance(heldout_cluster_index, pd.Series):
            heldout_cluster_index = heldout_cluster_index.tolist()
        else:
            heldout_cluster_index = list(heldout_cluster_index)
            
        if isinstance(heldout_centroid, pd.Series):
            heldout_centroid = heldout_centroid.tolist()
        else:
            heldout_centroid = list(heldout_centroid)
        
        # Combine train and heldout data
        if isinstance(self.heldout_smiles, pd.Series):
            combined_smiles = train_smiles_list + self.heldout_smiles.tolist()
        else:
            combined_smiles = train_smiles_list + list(self.heldout_smiles)
        
        combined_cluster_index = train_cluster_index + heldout_cluster_index
        combined_centroid = train_centroid + heldout_centroid
        
        # Create origin labels (train vs heldout)
        combined_origin = ['train'] * len(train_smiles_list) + ['heldout'] * len(self.heldout_smiles)
        
        # Build fingerprint dictionary for combined set
        fp_dict_combined = self.build_fp_dict(combined_smiles)
        
        results = []
        best_matches = self._compute_best_matches_against_smiles(combined_smiles, fp_dict_combined)

        for gen_idx, gen_smi in enumerate(self.generated_smiles):
            best_similar_smi, best_similarity, best_idx = best_matches.get(gen_smi, (None, 0.0, None))

            if gen_smi not in self.fp_dict_g:
                results.append({
                    'generated_molecule_id': gen_idx,
                    'generated_smiles': gen_smi,
                    'best_cluster_id': None,
                    'best_similarity_score': 0.0,
                    'cluster_origin': None,
                    'is_centroid': None,
                    'best_similar_smiles': None,
                    'above_threshold': False
                })
                continue

            if best_idx is not None:
                best_cluster_id = combined_cluster_index[best_idx]
                cluster_origin = combined_origin[best_idx]
                is_centroid = bool(combined_centroid[best_idx])
                above_threshold = best_similarity >= similarity_threshold
            else:
                best_cluster_id = None
                cluster_origin = None
                is_centroid = None
                above_threshold = False

            results.append({
                'generated_molecule_id': gen_idx,
                'generated_smiles': gen_smi,
                'best_cluster_id': best_cluster_id,
                'best_similarity_score': float(best_similarity) if best_similar_smi is not None else 0.0,
                'cluster_origin': cluster_origin,
                'is_centroid': is_centroid,
                'best_similar_smiles': best_similar_smi,
                'above_threshold': above_threshold
            })

        return results
    
    def generate_clustering_metrics_for_centroid(
        self,
        train_smiles: Iterable[str],
        train_cluster_index: Union[List[int], pd.Series],
        train_centroid: Union[List[int], pd.Series],
        heldout_cluster_index: Union[List[int], pd.Series],
        heldout_centroid: Union[List[int], pd.Series],
        similarity_threshold: float = 0.4
    ) -> List[Dict[str, Union[str, int, float, bool]]]:
        """
        Calculate clustering metrics for generated molecules using only centroid molecules.
        
        For each generated molecule, identify the best cluster assignment based on
        similarity scores with centroid molecules only (where centroid >= 0) in train + heldout sets.
        
        Args:
            train_smiles: Training set SMILES
            train_cluster_index: Cluster indices for training molecules
            train_centroid: Centroid values for training molecules (-1 for non-centroid, >=0 for centroid cluster_index)
            heldout_cluster_index: Cluster indices for heldout molecules
            heldout_centroid: Centroid values for heldout molecules (-1 for non-centroid, >=0 for centroid cluster_index)
            similarity_threshold: Minimum similarity threshold for cluster assignment
            
        Returns:
            List of dictionaries containing clustering metrics for each generated molecule
        """
        # Convert inputs to lists
        if isinstance(train_smiles, pd.Series):
            train_smiles_list = train_smiles.tolist()
        else:
            train_smiles_list = list(train_smiles)
            
        if isinstance(train_cluster_index, pd.Series):
            train_cluster_index = train_cluster_index.tolist()
        else:
            train_cluster_index = list(train_cluster_index)
            
        if isinstance(train_centroid, pd.Series):
            train_centroid = train_centroid.tolist()
        else:
            train_centroid = list(train_centroid)
            
        if isinstance(heldout_cluster_index, pd.Series):
            heldout_cluster_index = heldout_cluster_index.tolist()
        else:
            heldout_cluster_index = list(heldout_cluster_index)
            
        if isinstance(heldout_centroid, pd.Series):
            heldout_centroid = heldout_centroid.tolist()
        else:
            heldout_centroid = list(heldout_centroid)
        
        # Filter to keep only centroid molecules (centroid >= 0)
        train_centroid_smiles = []
        train_centroid_cluster_index = []
        train_centroid_values = []
        
        for i, (smi, cluster_idx, centroid_val) in enumerate(zip(train_smiles_list, train_cluster_index, train_centroid)):
            if centroid_val >= 0:  # Is a centroid
                train_centroid_smiles.append(smi)
                train_centroid_cluster_index.append(cluster_idx)
                train_centroid_values.append(centroid_val)
        
        if isinstance(self.heldout_smiles, pd.Series):
            heldout_smiles_list = self.heldout_smiles.tolist()
        else:
            heldout_smiles_list = list(self.heldout_smiles)
        
        heldout_centroid_smiles = []
        heldout_centroid_cluster_index = []
        heldout_centroid_values = []
        
        for i, (smi, cluster_idx, centroid_val) in enumerate(zip(heldout_smiles_list, heldout_cluster_index, heldout_centroid)):
            if centroid_val >= 0:  # Is a centroid
                heldout_centroid_smiles.append(smi)
                heldout_centroid_cluster_index.append(cluster_idx)
                heldout_centroid_values.append(centroid_val)
        
        # Combine train and heldout centroid data
        combined_smiles = train_centroid_smiles + heldout_centroid_smiles
        combined_cluster_index = train_centroid_cluster_index + heldout_centroid_cluster_index
        combined_centroid = train_centroid_values + heldout_centroid_values
        
        # Create origin labels (train vs heldout)
        combined_origin = ['train'] * len(train_centroid_smiles) + ['heldout'] * len(heldout_centroid_smiles)
        
        # Build fingerprint dictionary for combined centroid set
        fp_dict_combined = self.build_fp_dict(combined_smiles)
        
        results = []
        best_matches = self._compute_best_matches_against_smiles(combined_smiles, fp_dict_combined)

        for gen_idx, gen_smi in enumerate(self.generated_smiles):
            best_similar_smi, best_similarity, best_idx = best_matches.get(gen_smi, (None, 0.0, None))

            if gen_smi not in self.fp_dict_g:
                results.append({
                    'generated_molecule_id': gen_idx,
                    'generated_smiles': gen_smi,
                    'best_cluster_id': None,
                    'best_similarity_score': 0.0,
                    'cluster_origin': None,
                    'is_centroid': None,
                    'best_similar_smiles': None,
                    'above_threshold': False
                })
                continue

            if best_idx is not None:
                best_cluster_id = combined_cluster_index[best_idx]
                cluster_origin = combined_origin[best_idx]
                is_centroid = True
                above_threshold = best_similarity >= similarity_threshold
            else:
                best_cluster_id = None
                cluster_origin = None
                is_centroid = None
                above_threshold = False

            results.append({
                'generated_molecule_id': gen_idx,
                'generated_smiles': gen_smi,
                'best_cluster_id': best_cluster_id,
                'best_similarity_score': float(best_similarity) if best_similar_smi is not None else 0.0,
                'cluster_origin': cluster_origin,
                'is_centroid': is_centroid,
                'best_similar_smiles': best_similar_smi,
                'above_threshold': above_threshold
            })

        return results