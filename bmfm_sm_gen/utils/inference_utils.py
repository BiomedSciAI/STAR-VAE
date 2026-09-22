import pandas as pd
import numpy as np
from typing import List, Dict, Optional

def sample_distribution_ranges(
    training_set: pd.DataFrame,
    label_column: str,
    num_samples: int,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    n_bins: int = 10,
    num_top_bins: int = None
) -> List[Dict[str, object]]:

    if label_column not in training_set.columns:
        raise ValueError(f"'{label_column}' not found in the DataFrame.")
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    col = training_set[label_column]
    if min_val is None:
        min_val = col.min()
    if max_val is None:
        max_val = col.max()
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val.")

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bins = pd.cut(col, bins=bin_edges, include_lowest=True, right=False)
    bin_counts = bins.value_counts().sort_index()

    if num_top_bins is not None:
        # filter by counts greater than 0, choose top k
        bin_counts = bin_counts[bin_counts>0].tail(num_top_bins)
        n_bins = bin_counts.size
    
    total_available = bin_counts.sum()

    if total_available == 0:
        proportions = np.full(n_bins, 1.0 / n_bins)
    else:
        proportions = bin_counts.values / total_available

    raw = proportions * num_samples
    floor_counts = np.floor(raw).astype(int)
    remaining = num_samples - floor_counts.sum()

    if remaining > 0:
        frac = raw - floor_counts
        idx = np.argsort(-frac)
        for i in idx[:remaining]:
            floor_counts[i] += 1

    result: List[Dict[str, object]] = []

    for i, interval in enumerate(bin_counts.index):

        start_rounded = round(float(interval.left), 1)
        end_rounded = round(float(interval.right), 1)

        result.append({
            "range": [start_rounded, end_rounded],
            "num": int(floor_counts[i]),
        })

    return result
