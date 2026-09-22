#!/usr/bin/env python3
"""
Revision audit C2: uncertainty for the ablation study.

Motivation
----------
The reviewer noted that ablation conclusions rest on point estimates and that
small differences (rho = 0.120 / 0.149 / 0.153) are interpreted mechanistically.
No independent *training* seeds exist for these runs, so training variance cannot
be estimated from existing outputs. What does exist is three independent
*generation* seeds (42, 0, 101) per configuration.

This script therefore reports two distinct quantities and is explicit that they
are not interchangeable:

  1. Between-generation-seed spread. Spearman rho recomputed within each seed,
     giving a seed-level mean and range. This captures sampling variability at
     fixed model weights.

  2. Bootstrap CI over conditioning levels. Resamples the requested-label groups
     (high/mid/low x molecules) rather than individual molecules, so the interval
     respects the grouped structure of the sweep.

Neither quantity captures variance from model fitting, data splitting, or
checkpoint selection. Any statement of the form "configuration A conditions
better than configuration B" that is smaller than the reported spread should be
treated as unresolved.

Usage
-----
    python bmfm_sm_gen/scripts/revision_c2_ablation_uncertainty.py \
        --root paper_results/exp19_ablation_studies \
        --out  paper_results/revision_audits
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = ["42", "0", "101"]
BUCKETS = ["high", "mid", "low"]
# Requested labels per bucket, from exp19_generate_ablation.py:52-56
BUCKET_TARGET = {"high": 0.165, "mid": 0.500, "low": 0.835}
N_BOOT = 2000
RNG_SEED = 12345

# SA score min-max bounds used for the ablation task
SA_MIN, SA_MAX = 1.437, 7.513


def normalize(raw: float) -> float:
    return min(1.0, max(0.0, (raw - SA_MIN) / (SA_MAX - SA_MIN)))


def load_run(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """
    Return {seed: [(requested_label, realized_normalized_score), ...]}.
    Missing seeds/buckets are skipped and reported by the caller.
    """
    per_seed: dict[str, list[tuple[float, float]]] = {}
    for seed in SEEDS:
        pairs: list[tuple[float, float]] = []
        for bucket in BUCKETS:
            f = run_dir / "generated" / f"seed_{seed}" / bucket / "generated.csv"
            if not f.exists():
                continue
            with f.open() as fh:
                for r in csv.DictReader(fh):
                    raw = r.get("sa_score")
                    if raw in (None, "", "nan"):
                        continue
                    try:
                        y_hat = normalize(float(raw))
                    except ValueError:
                        continue
                    pairs.append((BUCKET_TARGET[bucket], y_hat))
        if pairs:
            per_seed[seed] = pairs
    return per_seed


def spearman(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_by_level(
    pairs: list[tuple[float, float]], n_boot: int = N_BOOT, rng_seed: int = RNG_SEED
) -> tuple[float, float]:
    """
    Percentile CI for Spearman rho, resampling WITHIN each conditioning level.

    Resampling molecules within levels (rather than across the pooled set)
    preserves the fixed number of requested levels, which is the design of the
    sweep. Resampling the pooled set would additionally randomize how many
    molecules land in each level and understate the grouped structure.
    """
    if len(pairs) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    by_level: dict[float, list[float]] = {}
    for x, y in pairs:
        by_level.setdefault(x, []).append(y)
    levels = sorted(by_level)
    arrays = {k: np.asarray(v) for k, v in by_level.items()}

    stats_out = []
    for _ in range(n_boot):
        xs, ys = [], []
        for lv in levels:
            a = arrays[lv]
            idx = rng.integers(0, a.size, a.size)
            ys.append(a[idx])
            xs.append(np.full(a.size, lv))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        if np.unique(y).size < 2:
            continue
        stats_out.append(stats.spearmanr(x, y).statistic)
    if not stats_out:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats_out, [2.5, 97.5])
    return (float(lo), float(hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        d for d in args.root.iterdir() if d.is_dir() and (d / "generated").exists()
    )

    results, notes = [], []
    notes.append(
        "Between-seed spread reflects GENERATION seeds (42, 0, 101) at fixed model "
        "weights. No independent training seeds exist for these runs, so training "
        "variance is NOT estimated here."
    )
    notes.append(
        f"Bootstrap: {N_BOOT} resamples drawn within each conditioning level "
        f"(rng seed {RNG_SEED}); percentile 95% CI."
    )

    for d in run_dirs:
        per_seed = load_run(d)
        if not per_seed:
            notes.append(f"{d.name}: no per-seed generated molecules found; skipped")
            continue

        seed_rhos = {s: spearman(p) for s, p in per_seed.items()}
        finite = [v for v in seed_rhos.values() if not math.isnan(v)]
        pooled = [pair for p in per_seed.values() for pair in p]
        rho_pooled = spearman(pooled)
        lo, hi = bootstrap_by_level(pooled)

        results.append(
            {
                "run_id": d.name,
                "n_seeds": len(per_seed),
                "n_molecules_pooled": len(pooled),
                "rho_pooled": round(rho_pooled, 4) if not math.isnan(rho_pooled) else "",
                "rho_seed_mean": round(float(np.mean(finite)), 4) if finite else "",
                "rho_seed_min": round(min(finite), 4) if finite else "",
                "rho_seed_max": round(max(finite), 4) if finite else "",
                "rho_seed_range": round(max(finite) - min(finite), 4) if finite else "",
                "boot_ci_lo": round(lo, 4) if not math.isnan(lo) else "",
                "boot_ci_hi": round(hi, 4) if not math.isnan(hi) else "",
                **{
                    f"rho_seed_{s}": (round(v, 4) if not math.isnan(v) else "")
                    for s, v in seed_rhos.items()
                },
            }
        )

    cols = [
        "run_id", "n_seeds", "n_molecules_pooled", "rho_pooled",
        "rho_seed_mean", "rho_seed_min", "rho_seed_max", "rho_seed_range",
        "boot_ci_lo", "boot_ci_hi",
    ] + [f"rho_seed_{s}" for s in SEEDS]

    csv_path = args.out / "c2_ablation_uncertainty.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    (args.out / "c2_ablation_uncertainty.json").write_text(
        json.dumps({"rows": results, "notes": notes}, indent=2)
    )

    print("=" * 108)
    print("C2 ablation uncertainty (generation-seed spread and level-wise bootstrap)")
    print("=" * 108)
    for n in notes:
        print(f"  note: {n}")
    print()
    hdr = (
        f"{'run_id':<34}{'n_mol':>7}{'rho':>8}{'seedmean':>9}"
        f"{'seedmin':>8}{'seedmax':>8}{'range':>7}{'boot 95% CI':>20}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ci = (
            f"[{r['boot_ci_lo']:.3f}, {r['boot_ci_hi']:.3f}]"
            if r["boot_ci_lo"] != ""
            else "n/a"
        )
        print(
            f"{r['run_id']:<34}{r['n_molecules_pooled']:>7}"
            f"{r['rho_pooled']:>8}{r['rho_seed_mean']:>9}"
            f"{r['rho_seed_min']:>8}{r['rho_seed_max']:>8}"
            f"{r['rho_seed_range']:>7}{ci:>20}"
        )
    print()
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
