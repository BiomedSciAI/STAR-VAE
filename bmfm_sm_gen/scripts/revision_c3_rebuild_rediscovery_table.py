#!/usr/bin/env python3
"""
Rebuild the rediscovery table from the canonical combined CSV.

The hand-transcribed table previously shifted four columns:

* ``CrossSim Ratio`` contained ``Diversity | D(Gen) / D(Heldout)``.
* ``InterSim`` inconsistently used CrossSim(Generated, Train).
* ``Div`` and ``Uniq`` both came from the uniqueness ratio under different
  rounding conventions.

This script writes a flat corrected CSV and a LaTeX body using explicit column
names. CrossSim is reported as the implemented all-pairs mean Tanimoto between
generated and held-out scaffold sets; it is not normalized by held-out
InterSim. Values are therefore in [0, 1].

Usage:
    python bmfm_sm_gen/scripts/revision_c3_rebuild_rediscovery_table.py \
      --input paper_results/figures_and_tables/fig_redis_combined_all_targets_all_sizes.csv \
      --out-dir paper_results/revision_audits/rediscovery_table
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


COL = {
    "exact_p": "Exact Discovery | Precision",
    "exact_r": "Exact Discovery | Recall",
    "nn": "Nearest Neighbor Discovery | Mean Sim Score",
    "clust": "Clustering | Heldout Proportion (sim >= 0.6)",
    "scaff": "Scaffold Rediscovery | Scaff(Generated) ∩ Scaff(Heldout) / Scaff(Heldout)",
    "cross": "Scaffold Similarity | CrossSim(Scaff(Generated) & Scaff(Heldout))",
    "inter": "Scaffold Similarity | InterSim(Scaff(Generated))",
    "div": "Diversity | D(Gen) / D(Heldout)",
    "uniq": "Uniqueness | U(Gen) / U(Heldout)",
}

TARGET_ORDER = ["ABL1", "BRD4", "GRL", "hiv1protease"]
MODEL_ORDER = ["CVAE", "PUBCHEM", "VAE"]


def fmt(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def format_exact(value: float) -> str:
    """Main table reports exact precision/recall in units of 1e-4."""
    scaled = float(value) * 1e4
    if abs(scaled) < 0.0005:
        return "0"
    return f"{scaled:.2f}".rstrip("0").rstrip(".")


def bold(values: list[float], prefer_low: bool = False) -> list[bool]:
    best = min(values) if prefer_low else max(values)
    return [abs(v - best) < 5e-13 for v in values]


def tex_model(model: str) -> str:
    return "PUBCHEM" if model == "PUBCHEM" else model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [name for name in [*COL.values(), "dataset", "model", "num_samples"] if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["dataset"].isin(TARGET_ORDER) & df["model"].isin(MODEL_ORDER)].copy()
    # Keep strings in output artifacts; temporary rank columns provide stable
    # manuscript ordering without pandas' categorical serialization warnings.
    df["_dataset_rank"] = df["dataset"].map({v: i for i, v in enumerate(TARGET_ORDER)})
    df["_model_rank"] = df["model"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    df = df.sort_values(["_dataset_rank", "num_samples", "_model_rank"])

    if len(df) != 60:
        raise ValueError(f"Expected 60 rows, found {len(df)}")

    out_rows: list[dict] = []
    tex_lines: list[str] = []
    for target in TARGET_ORDER:
        target_df = df[df["dataset"] == target]
        for sample_size in sorted(target_df["num_samples"].unique()):
            group = target_df[target_df["num_samples"] == sample_size].sort_values("_model_rank")
            if list(group["model"]) != MODEL_ORDER:
                raise ValueError(f"Unexpected model order for {target}/{sample_size}")

            best_flags: dict[str, list[bool]] = {}
            for key in ["exact_p", "exact_r", "nn", "clust", "scaff", "cross", "div", "uniq"]:
                best_flags[key] = bold(group[COL[key]].astype(float).tolist())
            best_flags["inter"] = bold(group[COL["inter"]].astype(float).tolist(), prefer_low=True)

            for row_index, (_, row) in enumerate(group.iterrows()):
                target_label = "HIV-1 Protease" if target == "hiv1protease" else target
                raw = {
                    "dataset": target,
                    "model": str(row["model"]),
                    "num_samples": int(row["num_samples"]),
                    **{key: float(row[col]) for key, col in COL.items()},
                }
                out_rows.append(raw)

                displayed = {
                    "exact_p": format_exact(raw["exact_p"]),
                    "exact_r": format_exact(raw["exact_r"]),
                    **{key: fmt(raw[key]) for key in ["nn", "clust", "scaff", "cross", "inter", "div", "uniq"]},
                }
                for key, is_best in ((k, best_flags[k][row_index]) for k in best_flags):
                    if is_best:
                        displayed[key] = rf"\textbf{{{displayed[key]}}}"

                tex_lines.append(
                    f"{target_label} & {tex_model(raw['model'])} & {raw['num_samples']} & "
                    + " & ".join(displayed[k] for k in ["exact_p", "exact_r", "nn", "clust", "scaff", "cross", "inter", "div", "uniq"])
                    + r" \\"
                )
            tex_lines.append(r"\grouprule")
        tex_lines[-1] = r"\midrule" if target != TARGET_ORDER[-1] else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out_dir / "rediscovery_table_corrected.csv", index=False)
    (args.out_dir / "rediscovery_table_body.tex").write_text("\n".join(line for line in tex_lines if line) + "\n")
    (args.out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "column_mapping": COL,
                "crosssim_definition": "mean all-pairs Tanimoto between deduplicated generated and held-out scaffold sets",
                "row_count": len(out_rows),
            },
            indent=2,
        )
        + "\n"
    )

    # Regression guard for the row that exposed the original shift.
    row = pd.DataFrame(out_rows).query("dataset == 'ABL1' and model == 'CVAE' and num_samples == 10000").iloc[0]
    assert abs(row["cross"] - 0.1095314860533048) < 1e-15
    assert abs(row["inter"] - 0.1039266547093906) < 1e-15
    assert abs(row["div"] - 1.096329198024618) < 1e-15
    assert abs(row["uniq"] - 1.013927576601671) < 1e-15

    print(f"Wrote {len(out_rows)} corrected rows to {args.out_dir}")


if __name__ == "__main__":
    main()
