#!/usr/bin/env python3
"""
Recompute the docking rows of the conditioning-summary table under one rule.

Standardized population:

* Generation validity = RDKit-parseable generated rows / 3003 attempts.
* Docking completion = valid_dock=True and score below the sentinel.
* Rho, beta, valid-only SR, and IntDiv are computed over that completed-docking
  population, including real non-negative QVina scores.
* Attempt-level SR uses all 3003 attempts in the denominator.

This matches the operational failure definition in the attempt-level Pareto
audit: the docking sentinel, not score sign, denotes a failed docking job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from scipy.stats import linregress, spearmanr

RDLogger.DisableLog("rdApp.*")

ATTEMPTS = 3003
SENTINEL = 10000.0
ALPHAS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
TARGETS = {
    "6y2f": {"s_min": -9.4, "prop_range": 9.4},
    "1syh": {"s_min": -10.0, "prop_range": 9.9},
    "4lde": {"s_min": -10.0, "prop_range": 9.9},
}


def internal_diversity(smiles: list[str], batch_size: int = 2000) -> float:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            arr = np.zeros(1024, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    if len(fps) < 2:
        return float("nan")
    x = torch.tensor(np.stack(fps), dtype=torch.float32)
    diag = (x * x).sum(dim=1)
    total = 0.0
    count = 0
    for i in range(0, len(x), batch_size):
        a = x[i : i + batch_size]
        da = diag[i : i + batch_size]
        for j in range(i, len(x), batch_size):
            b = x[j : j + batch_size]
            db = diag[j : j + batch_size]
            dot = a @ b.T
            tani = dot / (da[:, None] + db[None, :] - dot).clamp_min(1e-8)
            if i == j:
                idx = torch.triu_indices(len(a), len(b), offset=1)
                vals = tani[idx[0], idx[1]]
            else:
                vals = tani.reshape(-1)
            total += float(vals.sum())
            count += vals.numel()
    return 1.0 - total / count


def summarize(path: Path, target: str, alpha: float) -> dict:
    cfg = TARGETS[target]
    all_generated = pd.read_csv(path)
    docked = all_generated[
        (all_generated["valid_dock"] == True) & (all_generated["docking_score"] < SENTINEL)
    ].copy()
    if docked.empty:
        raise ValueError(f"No docked rows: {path}")

    y_star = docked["conditioning_y_star"].to_numpy(float)
    y_model = 1.0 - y_star
    scores = docked["docking_score"].to_numpy(float)
    rho = float(spearmanr(y_model, scores).statistic)
    beta = float(linregress(y_model, scores).slope)

    target_score = cfg["s_min"] * (1.0 - y_model)
    abs_err = np.abs(scores - target_score)
    hits10 = abs_err < 0.1 * cfg["prop_range"]
    hits20 = abs_err < 0.2 * cfg["prop_range"]

    return {
        "target": target,
        "alpha": alpha,
        "n_attempted": ATTEMPTS,
        "n_generated": len(all_generated),
        "n_docked": len(docked),
        "generation_validity": len(all_generated) / ATTEMPTS,
        "docking_completion": len(docked) / ATTEMPTS,
        "rho": rho,
        "beta": beta,
        "sr10_docked": float(hits10.mean()),
        "sr20_docked": float(hits20.mean()),
        "sr10_attempt": float(hits10.sum() / ATTEMPTS),
        "sr20_attempt": float(hits20.sum() / ATTEMPTS),
        "intdiv_all": internal_diversity(docked["smiles"].dropna().tolist()),
        "intdiv_sr20": internal_diversity(docked.loc[hits20, "smiles"].dropna().tolist()),
        "n_positive_scores": int((scores >= 0).sum()),
        "rho_negative_only": float(spearmanr(-docked.loc[docked["docking_score"] < 0, "conditioning_y_star"], docked.loc[docked["docking_score"] < 0, "docking_score"]).statistic),
        "n_negative_scores": int((scores < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for target in TARGETS:
        for alpha in ALPHAS:
            path = args.root / "docked" / target / f"alpha_{alpha}" / "sweep_docked.csv"
            rows.append(summarize(path, target, alpha))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "docking_summary_sentinel_only.csv", index=False)
    (args.out_dir / "docking_summary_sentinel_only.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    # Build the complete conditioning-summary table body. SA rows retain the
    # existing scored-population analysis; docking rows use the sentinel-only
    # population above. Keeping both Gen.Valid and Dock columns prevents the
    # prior ambiguity where a negative-score fraction was labeled "Validity".
    sa_table = pd.read_csv(args.root / "stats" / "controllability_metrics_table.csv")
    sa_table = sa_table[sa_table["target"] == "sa_score"].copy()

    paper_alpha = {"sa_score": 12.0, "6y2f": 2.0, "1syh": 12.0, "4lde": 12.0}
    target_label = {"sa_score": "SA score", "6y2f": "6Y2F", "1syh": "1SYH", "4lde": "4LDE"}
    tex_lines = []
    for target in ["sa_score", "6y2f", "1syh", "4lde"]:
        source = sa_table if target == "sa_score" else df[df.target == target]
        for row_i, (_, row) in enumerate(source.sort_values("alpha").iterrows()):
            best = abs(float(row.alpha) - paper_alpha[target]) < 1e-12
            def show(value, decimals=3):
                text = f"{float(value):.{decimals}f}"
                return rf"\textbf{{{text}}}" if best else text
            def pct(value):
                text = f"{100 * float(value):.1f}\\%"
                return rf"\textbf{{{text}}}" if best else text

            name = target_label[target] if row_i == 3 else ""
            if target == "sa_score":
                vals = [
                    show(row.rho), show(row.beta), pct(row.validity), "--",
                    show(row.sr_01), show(row.sr_02), show(row.div_all), show(row.div_success),
                ]
            else:
                vals = [
                    show(row.rho), show(row.beta), pct(row.generation_validity),
                    pct(row.docking_completion), show(row.sr10_docked),
                    show(row.sr20_docked), show(row.intdiv_all), show(row.intdiv_sr20),
                ]
            tex_lines.append(f"{name} & {show(row.alpha, 1)} & " + " & ".join(vals) + r" \\")
        tex_lines.append(r"\midrule")
    tex_lines.pop()
    (args.out_dir / "conditioning_summary_body.tex").write_text("\n".join(tex_lines) + "\n")

    print(df.to_string(index=False))
    print(f"Wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
