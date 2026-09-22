#!/usr/bin/env python
"""
exp02_run_moses_full_metrics.py
================================
Run the full MOSES benchmark (get_all_metrics) on STAR-VAE unconditional
generation.  Computes: Validity, Unique@1K, Unique@10K, Novelty, Filters,
IntDiv1, IntDiv2, Scaffold, FCD/TestSF, SNN/TestSF, and the **/Test** variants.

Usage (standalone):
    python bmfm_sm_gen/scripts/exp02_run_moses_full_metrics.py \\
        [--ckpt PATH] [--vocab PATH] [--input-csv PATH] \\
        [--n-gen 30000] [--batch-size 256] [--temperature 1.1] [--top-k 20] \\
        [--output-dir paper_results/exp02_moses_benchmark] \\
        [--n-jobs 8] [--device cuda]

If --input-csv is given, generation is skipped and the CSV (column 'smiles')
is used directly.  Otherwise molecules are generated from the pretrained VAE.

The optional 'molsets' (MOSES) package must be installed before running this
script; see the error from ensure_moses() for compatible installation options.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check the optional molsets (MOSES) dependency
# ---------------------------------------------------------------------------

def ensure_moses() -> None:
    try:
        import moses  # noqa: F401
        log.info("moses already installed.")
        return
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MOSES is required but is not installed. Install it in the project "
            "environment before running, for example with `uv pip install "
            "--no-deps git+https://github.com/molecularsets/moses.git`, then "
            "verify its RDKit, FCD, and PyTorch dependencies."
        ) from exc


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_smiles(ckpt_path: str, vocab_path: str,
                    n_gen: int, batch_size: int,
                    temperature: float, top_k: int,
                    device: str) -> list[str]:
    """Generate n_gen molecules unconditionally from the VAE prior."""
    log.info(f"Loading model from {ckpt_path} …")

    # Inline import so the script can be run without the full package installed
    # as long as the working directory is the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from bmfm_sm_gen.models.vae import TransformerVAE
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.launch.run_generate_molecules import (
        generate_unconstrained_molecules,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {dev}")

    lm = VAELightningModule.load_from_checkpoint(
        ckpt_path,
        vocab_file=vocab_path,
        map_location=dev,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
    )
    model = TransformerVAE(**lm.model_params)
    import math  # noqa, already imported above in run_generate_molecules
    checkpoint = torch.load(ckpt_path, map_location=dev)
    weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(weights)
    model.to(dev)
    model.eval()

    log.info(f"Generating {n_gen} molecules (batch={batch_size}, T={temperature}, topk={top_k}) …")
    smiles = generate_unconstrained_molecules(
        lm, model,
        num_samples=n_gen,
        temperature=temperature,
        top_k=top_k,
        batch_size=batch_size,
    )
    log.info(f"Generated {len(smiles)} valid SMILES before MOSES filtering.")
    return smiles


def generate_conditioned_smiles(ckpt_path: str, vocab_path: str,
                                 n_gen: int, batch_size: int,
                                 temperature: float, top_k: int,
                                 device: str, moses_csv_path: str, seed: int) -> list[str]:
    """Generate n_gen molecules conditioned on random train seeds using the VAE encoder-decoder."""
    log.info(f"Loading model from {ckpt_path} for seed-conditioned generation …")

    # Inline import
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bmfm_sm_gen.models.vae import TransformerVAE
    from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
    from bmfm_sm_gen.launch.run_generate_molecules import (
        generate_from_multiple_targets_batch,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {dev}")

    lm = VAELightningModule.load_from_checkpoint(
        ckpt_path,
        vocab_file=vocab_path,
        map_location=dev,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
    )
    model = TransformerVAE(**lm.model_params)
    checkpoint = torch.load(ckpt_path, map_location=dev)
    weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(weights)
    model.to(dev)
    model.eval()

    # Load random training seeds
    log.info(f"Loading {n_gen} random training seeds from {moses_csv_path} …")
    df_moses = pd.read_csv(moses_csv_path)
    df_train = df_moses[df_moses['SPLIT'] == 'train'].copy()
    seeds = df_train['SMILES'].dropna().sample(n=n_gen, random_state=seed or 42).tolist()

    # Generate in batches
    log.info(f"Generating molecules conditioned on {len(seeds)} seeds …")
    generated_smiles = []
    sub_batch_size = 512
    for i in range(0, len(seeds), sub_batch_size):
        sub_seeds = seeds[i:i+sub_batch_size]
        results = generate_from_multiple_targets_batch(
            model=model,
            lm=lm,
            target_smiles_list=sub_seeds,
            num_samples_per_target=1,
            std_dev=1.0,
            sampling_scheme="around_mu",
            temperature=temperature,
            top_k=top_k,
        )
        for target, gen, sim in results:
            if gen:
                generated_smiles.append(gen)

    log.info(f"Generated {len(generated_smiles)} valid SMILES from seeds before MOSES filtering.")
    return generated_smiles


# ---------------------------------------------------------------------------
# MOSES evaluation
# ---------------------------------------------------------------------------

def run_moses(gen_smiles: list[str], n_jobs: int, device: str,
              output_dir: Path, run_tag: str, moses_csv_path: str,
              manifest: dict) -> dict:
    """Run MOSES get_all_metrics and save results."""
    # Patch Pandas 2.0+ DataFrame.append removal for compatibility with older moses/molsets
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        log.info("Patching pandas.DataFrame.append for moses compatibility")
        pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)

    # Patch SciPy correlation/cosine distance int64 overflow by forcing float64 casting
    import moses.metrics.metrics as moses_metrics
    import numpy as np
    def patched_cos_similarity(ref_counts, gen_counts):
        if len(ref_counts) == 0 or len(gen_counts) == 0:
            return np.nan
        keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
        ref_vec = np.array([ref_counts.get(k, 0) for k in keys], dtype=np.float64)
        gen_vec = np.array([gen_counts.get(k, 0) for k in keys], dtype=np.float64)
        return 1 - moses_metrics.cos_distance(ref_vec, gen_vec)
    moses_metrics.cos_similarity = patched_cos_similarity
    log.info("Patched moses.metrics.metrics.cos_similarity for float64 precision")

    import moses
    from moses.metrics import get_all_metrics
    from moses.metrics.metrics import get_statistics

    log.info(f"Loading local MOSES reference splits from {moses_csv_path} …")
    df_moses = pd.read_csv(moses_csv_path)
    train_smiles = df_moses[df_moses['SPLIT'] == 'train']['SMILES'].dropna().tolist()
    test_smiles = df_moses[df_moses['SPLIT'] == 'test']['SMILES'].dropna().tolist()
    testSF_smiles = df_moses[df_moses['SPLIT'] == 'test_scaffolds']['SMILES'].dropna().tolist()
    log.info(f"Loaded {len(train_smiles)} train, {len(test_smiles)} test, {len(testSF_smiles)} test_scaffolds molecules.")

    log.info("Loading bundled precalculated MOSES statistics …")
    ptest = get_statistics('test')
    ptest_scaffolds = get_statistics('test_scaffolds')

    log.info(f"Running get_all_metrics on {len(gen_smiles)} generated SMILES …")
    log.info(f"  n_jobs={n_jobs}, device={device}")

    metrics = get_all_metrics(
        gen=gen_smiles,
        k=[1000, 10000],
        n_jobs=n_jobs,
        device=device,
        test=test_smiles,
        test_scaffolds=testSF_smiles,
        ptest=ptest,
        ptest_scaffolds=ptest_scaffolds,
        train=train_smiles,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["metrics"] = {
        "moses_valid": float(metrics["valid"]),
        "moses_valid_population": "submitted_population",
        "moses_submitted_count": len(gen_smiles),
    }
    manifest_path = output_dir / f"run_manifest_{run_tag}.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info(f"Saved run manifest → {manifest_path}")

    log.info("MOSES metrics computed:")
    for k, v in sorted(metrics.items()):
        log.info(f"  {k:30s}: {v}")

    # ---- Save outputs -------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON (machine-readable, all decimals preserved)
    json_path = output_dir / f"moses_metrics_{run_tag}.json"
    with open(json_path, "w") as fh:
        json.dump({k: float(v) if not isinstance(v, str) else v
                   for k, v in metrics.items()}, fh, indent=2)
    log.info(f"Saved JSON  → {json_path}")

    # 2. Human-readable markdown table
    md_path = output_dir / f"moses_metrics_{run_tag}.md"
    lines = [
        f"# STAR-VAE MOSES Full Metrics, {run_tag}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**N generated:** {len(gen_smiles)}  ",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in sorted(metrics.items()):
        val_str = f"{float(v):.6f}" if not isinstance(v, str) else v
        lines.append(f"| {k} | {val_str} |")
    md_path.write_text("\n".join(lines) + "\n")
    log.info(f"Saved Markdown → {md_path}")

    # 3. Generated molecules CSV
    csv_path = output_dir / f"generated_smiles_{run_tag}.csv"
    pd.DataFrame({"smiles": gen_smiles}).to_csv(csv_path, index=False)
    log.info(f"Saved generated molecules CSV → {csv_path}")

    # 4. Summary CSV for easy import into paper table
    summary_keys = [
        "valid", "unique@1000", "unique@10000", "novelty",
        "Filters", "IntDiv", "IntDiv2", "Scaffold",
        "FCD/Test", "FCD/TestSF", "SNN/Test", "SNN/TestSF",
    ]
    summary = {k: metrics.get(k, "N/A") for k in summary_keys}
    summary_path = output_dir / f"moses_summary_{run_tag}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    log.info(f"Saved summary CSV → {summary_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full MOSES benchmark on STAR-VAE unconditional generation."
    )
    p.add_argument(
        "--ckpt",
        default=str(
            Path(os.environ.get("OUTPUT_DIR", "outputs"))
            / "expt_80m_400k_8gpus_1node_dm512_df2048_ld256_d1_kld_original"
            / "checkpoints/vae-epochepoch=07-step=303912-val_loss=0.29.ckpt"
        ),
        help="Path to VAE checkpoint.",
    )
    p.add_argument(
        "--vocab",
        default=os.environ.get("VOCAB_FILE", "checkpoints/vae_pubchem80m/vocab.json"),
        help="Path to vocabulary JSON file.",
    )
    p.add_argument(
        "--input-csv",
        default=None,
        help="Pre-generated SMILES CSV (column 'smiles'). Skip generation if given.",
    )
    p.add_argument(
        "--requested-attempts",
        type=int,
        default=None,
        help="Original decode-attempt count for --input-csv provenance; unknown if omitted.",
    )
    p.add_argument("--n-gen", type=int, default=30000,
                   help="Number of molecules to generate (if --input-csv not given).")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Decoding batch size.")
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--output-dir",
        default="paper_results/exp02_moses_benchmark",
        help="Directory to save metric JSON/CSV/MD outputs.",
    )
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Parallel workers for MOSES metrics.")
    p.add_argument("--device", default="cuda",
                   help="Device for FCD model ('cuda' or 'cpu').")
    p.add_argument("--run-tag", default=None,
                   help="Tag appended to output filenames. Defaults to datetime.")
    p.add_argument(
        "--moses-csv",
        default=os.environ.get("MOSES_RAW_CSV", "data/moses_raw/dataset_v1.csv"),
        help="Path to local MOSES dataset CSV containing splits.",
    )
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--conditioned", action="store_true",
                   help="Enable seed-conditioned generation around training seeds.")
    return p.parse_args()


def main() -> None:
    # Patch Pandas 2.0+ DataFrame.append removal for compatibility with older moses/molsets
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)

    args = parse_args()

    if args.requested_attempts is not None and not args.input_csv:
        raise ValueError("--requested-attempts is only valid with --input-csv")
    if args.requested_attempts is not None and args.requested_attempts < 1:
        raise ValueError("--requested-attempts must be positive")

    if args.seed is not None:
        log.info(f"Setting random seed to {args.seed} …")
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Always install moses first
    ensure_moses()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Obtain generated SMILES ----------------------------------------
    protocol = "input_csv" if args.input_csv else "seed_conditioned" if args.conditioned else "random_prior"
    requested_attempts = args.requested_attempts if args.input_csv else args.n_gen
    if args.input_csv:
        log.info(f"Loading generated SMILES from {args.input_csv} …")
        df = pd.read_csv(args.input_csv)
        # accept column named 'smiles' or 'generated_smiles'
        if "smiles" in df.columns:
            gen_smiles = df["smiles"].dropna().tolist()
        elif "generated_smiles" in df.columns:
            gen_smiles = df["generated_smiles"].dropna().tolist()
        else:
            raise ValueError(
                f"CSV must have a 'smiles' or 'generated_smiles' column. "
                f"Found: {list(df.columns)}"
            )
        log.info(f"Loaded {len(gen_smiles)} SMILES from CSV.")
    elif args.conditioned:
        gen_smiles = generate_conditioned_smiles(
            ckpt_path=args.ckpt,
            vocab_path=args.vocab,
            n_gen=args.n_gen,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            moses_csv_path=args.moses_csv,
            seed=args.seed,
        )
    else:
        gen_smiles = generate_smiles(
            ckpt_path=args.ckpt,
            vocab_path=args.vocab,
            n_gen=args.n_gen,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )

    # ---- Run MOSES evaluation -------------------------------------------
    from rdkit import Chem
    decoded_count = sum(isinstance(s, str) and bool(s.strip()) for s in gen_smiles)
    valid_count = sum(
        isinstance(s, str) and bool(s.strip()) and Chem.MolFromSmiles(s) is not None
        for s in gen_smiles
    )
    manifest = {
        "run_tag": run_tag,
        "protocol": protocol,
        "requested_attempts": requested_attempts,
        "decoded_count": decoded_count,
        "valid_count_before_moses": valid_count,
        "submitted_count": len(gen_smiles),
        "input_csv": args.input_csv,
        "checkpoint": None if args.input_csv else args.ckpt,
        "vocab": None if args.input_csv else args.vocab,
        "moses_csv": args.moses_csv,
        "seed": args.seed,
        "parameters": {
            "n_gen": args.n_gen,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "n_jobs": args.n_jobs,
            "device_requested": args.device,
            "device_used_for_moses": args.device if torch.cuda.is_available() else "cpu",
        },
        "count_notes": {
            "requested_attempts": (
                "User-supplied original attempt count for --input-csv."
                if args.input_csv and args.requested_attempts is not None
                else "Unknown because --input-csv was used without --requested-attempts."
                if args.input_csv
                else "Requested generation count from --n-gen."
            ),
            "decoded_count": "Non-empty decoded strings retained by the selected generation/input path.",
            "valid_count_before_moses": "Submitted strings accepted by RDKit before MOSES evaluation.",
            "submitted_count": "Population passed unchanged to MOSES get_all_metrics.",
        },
    }
    metrics = run_moses(
        gen_smiles=gen_smiles,
        n_jobs=args.n_jobs,
        device=args.device if torch.cuda.is_available() else "cpu",
        output_dir=output_dir,
        run_tag=run_tag,
        moses_csv_path=args.moses_csv,
        manifest=manifest,
    )

    log.info("=" * 60)
    log.info("DONE. Key metrics:")
    for k in ["valid", "unique@10000", "novelty", "Filters",
              "IntDiv", "FCD/Test", "SNN/Test"]:
        log.info(f"  {k:20s}: {metrics.get(k, 'N/A')}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
