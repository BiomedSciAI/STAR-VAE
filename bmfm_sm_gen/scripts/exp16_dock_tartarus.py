#!/usr/bin/env python
"""
exp16_dock_tartarus.py
======================
STATUS : NEW, Experiment 16 (Tartarus Docking Controllability with QVina Oracle)
PURPOSE: Score generated molecules with the actual QVina oracle from the Tartarus
         benchmark, the SAME scoring function used to build the original Tartarus
         training set.  This avoids any double-dipping concern because:
           - The CVAE was trained on LABELS from QVina runs on the Tartarus dataset
           - We now score DIFFERENT (newly generated) molecules with QVina
           - No learned predictor model is involved; QVina is a physics-based
             software tool, not a fitted statistical model

IMPORTANT: This script must be run with the Tartarus conda environment:
    conda run -p <see .env> --no-capture-output \\
        python bmfm_sm_gen/scripts/exp16_dock_tartarus.py ...

  AND it must be run with cwd = <see .env>
  (because docking.py uses relative paths for qvina binary and receptor structures).

Usage:
    python exp16_dock_tartarus.py \\
        --target 6y2f --seed 0 --bucket high \\
        --input  /path/to/generated/6y2f/seed_0/high.csv \\
        --output /path/to/docked/6y2f/seed_0/high_docked.csv

    # Run all buckets for one target/seed:
    python exp16_dock_tartarus.py --target 6y2f --seed 0 --bucket all

    # Run everything:
    python exp16_dock_tartarus.py --target all --seed all --bucket all

Outputs:
    paper_results/exp16_tartarus_oracle/docked/{target}/seed_{seed}/{bucket}_docked.csv
    Columns: smiles, conditioning_y_star, bucket, target, seed, docking_score, valid_dock
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# docking.py requires cwd = Tartarus repo root
TARTARUS_ROOT = Path(os.environ.get("TARTARUS_ROOT", "data/tartarus"))

# Output root
REPO_ROOT = Path(os.environ.get("STAR_VAE_ROOT", str(Path(__file__).parents[2])))
GEN_ROOT     = REPO_ROOT / "paper_results/exp16_tartarus_oracle/generated"
DOCK_ROOT    = REPO_ROOT / "paper_results/exp16_tartarus_oracle/docked"

TARGETS  = ["1syh", "4lde", "6y2f"]
SEEDS    = [0, 42, 101]
BUCKETS  = ["high", "mid", "low"]

# Each worker runs docking in a separate process and writes to a separate
# temporary directory. Throughput and stable worker count depend on the local
# filesystem and external Tartarus/QVina installation; validate a small input
# before scaling up.
N_WORKERS = 8

FAILED_SCORE = 10_000.0   # sentinel returned by Tartarus when docking fails

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Docking worker (runs in subprocess pool, each worker imports tartarus)
# ---------------------------------------------------------------------------

# Path to tartarus conda env. obabel/obenergy live here.
# docking.py calls them as bare names, so this dir must be on PATH.
TARTARUS_ENV = os.environ.get("TARTARUS_ENV")


def _dock_one(args_tuple):
    """
    Top-level function (picklable) called by ProcessPoolExecutor.
    Returns (smiles, score).
    """
    smi, target = args_tuple
    try:
        # Guard: skip null / empty / non-string SMILES immediately,         # passing float('nan') or '' to obabel causes it to either hang
        # or emit a 0-byte pdbqt that can stall obenergy.
        if not isinstance(smi, str) or not smi.strip():
            return (smi, FAILED_SCORE)
        # Strip accidental surrounding single-quotes that confuse obabel.
        smi = smi.strip()
        if smi.startswith("'") and smi.endswith("'"):
            smi = smi[1:-1].strip()
        if not smi:
            return (smi, FAILED_SCORE)

        # Must set cwd to Tartarus root so relative paths in docking.py resolve
        os.chdir(str(TARTARUS_ROOT))
        sys.path.insert(0, str(TARTARUS_ROOT))
        # Ensure obabel/obenergy are on PATH for subprocess calls inside docking.py.
        # Workers inherit os.environ from the parent at fork, but if the parent
        # was launched without the tartarus env active the PATH will be missing.
        if TARTARUS_ENV:
            tartarus_bin = str(Path(TARTARUS_ENV) / "bin")
            path_dirs = os.environ.get("PATH", "").split(os.pathsep)
            if tartarus_bin not in path_dirs:
                os.environ["PATH"] = tartarus_bin + os.pathsep + os.environ.get("PATH", "")
        from tartarus.docking import perform_calc_single
        score = perform_calc_single(smi, target, docking_program="qvina")
    except Exception as e:
        score = FAILED_SCORE
        # suppress. The caller logs it
    return (smi, float(score))


# ---------------------------------------------------------------------------
# Per-file docking
# ---------------------------------------------------------------------------

def dock_csv(in_csv: Path, target: str, out_csv: Path, n_workers: int = N_WORKERS):
    """Dock all molecules in in_csv and write results to out_csv."""
    df = pd.read_csv(in_csv)
    smiles_list = df["smiles"].tolist()
    n = len(smiles_list)
    log.info(f"  Docking {n} molecules for target={target} ...")

    results = {}   # smi → score
    args_list = [(smi, target) for smi in smiles_list]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_dock_one, a): a[0] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            smi, score = fut.result()
            results[smi] = score
            done += 1
            if done % 50 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(f"    {done}/{n} docked  ({rate:.1f} mol/s)  "
                         f"ETA {(n - done)/rate/60:.1f} min" if rate > 0 else
                         f"    {done}/{n} docked")

    scores = [results.get(smi, FAILED_SCORE) for smi in smiles_list]
    df["docking_score"] = scores
    df["valid_dock"]    = [s < 9000 for s in scores]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_valid = sum(1 for s in scores if s < 9000)
    n_fail  = n - n_valid
    log.info(f"  Done: {n_valid} valid, {n_fail} failed → {out_csv}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True,
                   choices=TARGETS + ["all"])
    p.add_argument("--seed",   required=True,
                   help="Seed (0, 42, 101) or 'all'.")
    p.add_argument("--bucket", required=True,
                   choices=BUCKETS + ["all"])
    p.add_argument("--n-workers", type=int, default=N_WORKERS,
                   help="Parallel docking workers.")
    p.add_argument("--tartarus-root", default=str(TARTARUS_ROOT),
                   help="Tartarus checkout root (default: $TARTARUS_ROOT or data/tartarus).")
    p.add_argument("--tartarus-env", default=TARTARUS_ENV,
                   help="Optional Tartarus environment prefix containing bin/ (default: $TARTARUS_ENV).")
    # Optional direct path overrides (for single-file mode)
    p.add_argument("--input",  default=None)
    p.add_argument("--output", default=None)
    # Optional root directory overrides (for stratified/docked separation)
    p.add_argument("--input-root",  default=None,
                   help="Override generated/ root directory.")
    p.add_argument("--output-root", default=None,
                   help="Override docked/ root directory.")
    return p.parse_args()


def main():
    global TARTARUS_ROOT, TARTARUS_ENV
    args = parse_args()
    TARTARUS_ROOT = Path(args.tartarus_root).expanduser().resolve()
    TARTARUS_ENV = args.tartarus_env
    if not TARTARUS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Tartarus checkout not found: {TARTARUS_ROOT}. Set --tartarus-root "
            "or TARTARUS_ROOT to a checkout containing tartarus/docking.py."
        )

    targets = TARGETS if args.target == "all" else [args.target]
    seeds   = SEEDS   if args.seed   == "all" else [int(args.seed)]
    buckets = BUCKETS if args.bucket == "all" else [args.bucket]

    # Resolve root directories (allow override for stratified runs)
    gen_root  = Path(args.input_root)  if args.input_root  else GEN_ROOT
    dock_root = Path(args.output_root) if args.output_root else DOCK_ROOT

    # Single-file override
    if args.input and args.output:
        in_csv  = Path(args.input)
        out_csv = Path(args.output)
        dock_csv(in_csv, args.target, out_csv, args.n_workers)
        return

    for target in targets:
        for seed in seeds:
            for bucket in buckets:
                in_csv  = gen_root  / target / f"seed_{seed}" / f"{bucket}.csv"
                out_csv = dock_root / target / f"seed_{seed}" / f"{bucket}_docked.csv"

                if out_csv.exists():
                    log.info(f"SKIP (already exists): {out_csv}")
                    continue

                if not in_csv.exists():
                    log.warning(f"MISSING input: {in_csv}. Run generation first.")
                    continue

                log.info(f"\n=== {target} / seed={seed} / {bucket} ===")
                dock_csv(in_csv, target, out_csv, args.n_workers)

    log.info("\n=== exp16_dock_tartarus.py complete ===")


if __name__ == "__main__":
    main()
