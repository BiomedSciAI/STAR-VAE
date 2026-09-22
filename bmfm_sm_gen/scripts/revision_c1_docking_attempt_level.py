#!/usr/bin/env python3
"""
Revision audit C1: attempt-level accounting and operating-point (Pareto) table
for the docking-score conditional generation experiments.

Motivation
----------
The reviewer noted that the docking results report valid-only statistics and
designate a single "recommended" guidance scale without a stated objective. This
script reconstructs, for every protein target and every guidance scale alpha:

    n_attempted      1,001 y* levels x 3 samples = 3,003
    n_generated      rows written by the generation script (RDKit-parseable)
    n_docked         rows with a usable docking score (sentinel 10000 excluded)
    n_scored_neg     docked rows with a negative (physically meaningful) score
    validity         n_generated / n_attempted
    dock_success     n_docked / n_attempted
    rho, beta        association between requested label and docking score
    SR10/SR20        valid-only and attempt-level
    strong_frac      fraction of ATTEMPTS yielding a score <= a strong-binding
                     cutoff, i.e. enrichment at the strong end
    pareto           whether the alpha is Pareto-optimal in
                     (rho, validity, SR20_attempt, strong_frac)

The strong-binding cutoff is taken as the 10th percentile of the target's
training-set docking distribution when available, else the 10th percentile of
the alpha=1 generated distribution; the choice is recorded in the output.

Nothing here re-runs docking. It only re-aggregates existing sweep_docked.csv
files.

Usage
-----
    python bmfm_sm_gen/scripts/revision_c1_docking_attempt_level.py \
        --root paper_results/exp17_improving_conditioning/C_film_only_finetune \
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

ATTEMPTS = 3003
ALPHAS = ["1.0", "1.5", "2.0", "3.0", "5.0", "8.0", "12.0"]
TARGETS = ["1syh", "4lde", "6y2f"]
FAILED_DOCK_SENTINEL = 10000.0
TARGET_S_MIN = {"1syh": -10.0, "4lde": -10.0, "6y2f": -9.4}
TARGET_PROP_RANGE = {"1syh": 9.9, "4lde": 9.9, "6y2f": 9.4}


def load_docked(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def summarize_target_alpha(
    rows: list[dict], strong_cutoff: float | None, target_s_min: float,
    target_prop_range: float,
) -> dict:
    """Aggregate one (target, alpha) condition at attempt level."""
    n_generated = len(rows)

    scores, y_stars = [], []
    for r in rows:
        try:
            s = float(r["docking_score"])
        except (TypeError, ValueError):
            continue
        valid_flag = str(r.get("valid_dock", "True")).strip().lower() in ("true", "1")
        if not valid_flag or s >= FAILED_DOCK_SENTINEL:
            continue
        scores.append(s)
        y_stars.append(float(r["conditioning_y_star"]))

    scores = np.asarray(scores, dtype=float)
    y_stars = np.asarray(y_stars, dtype=float)
    n_docked = int(scores.size)
    n_neg = int((scores < 0).sum()) if n_docked else 0

    out = {
        "n_attempted": ATTEMPTS,
        "n_generated": n_generated,
        "n_docked": n_docked,
        "n_scored_negative": n_neg,
        "validity_generated": n_generated / ATTEMPTS,
        "dock_success_attempt": n_docked / ATTEMPTS,
        "rho": float("nan"),
        "beta": float("nan"),
        "mean_score": float("nan"),
        "best_score": float("nan"),
        "SR10_valid_only": float("nan"),
        "SR20_valid_only": float("nan"),
        "SR10_attempt_level": float("nan"),
        "SR20_attempt_level": float("nan"),
        "strong_frac_of_docked": float("nan"),
        "strong_frac_of_attempts": float("nan"),
        "strong_cutoff_used": strong_cutoff,
    }
    if n_docked < 3:
        return out

    # The generation files save sweep coordinate y* and model-side coordinate
    # y_model = 1-y*. Report the model-side coordinate throughout, so rho,
    # beta, and the target score all share one orientation.
    y_model = 1.0 - y_stars
    out["rho"] = float(stats.spearmanr(y_model, scores).statistic)
    slope, intercept = np.polyfit(y_model, scores, 1)
    out["beta"] = float(slope)
    out["mean_score"] = float(scores.mean())
    out["best_score"] = float(scores.min())

    # Target score implied by y_model, on the fixed training-normalization
    # scale used by Table 4. Do not derive this from each generated sample's
    # observed minimum: doing so changes the SR target across alpha values.
    target_score = target_s_min * (1.0 - y_model)
    prop_range = target_prop_range
    abs_err = np.abs(scores - target_score)
    for eps, key in ((0.10, "SR10"), (0.20, "SR20")):
        hits = int((abs_err < eps * prop_range).sum())
        out[f"{key}_valid_only"] = hits / n_docked
        out[f"{key}_attempt_level"] = hits / ATTEMPTS

    if strong_cutoff is not None:
        n_strong = int((scores <= strong_cutoff).sum())
        out["strong_frac_of_docked"] = n_strong / n_docked
        out["strong_frac_of_attempts"] = n_strong / ATTEMPTS

    return out


def pareto_front(records: list[dict], objectives: list[tuple[str, int]]) -> list[bool]:
    """
    Mark Pareto-optimal records. objectives is a list of (key, sign) where
    sign=+1 means larger is better.
    """
    n = len(records)
    flags = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal, strictly_better = True, False
            for key, sign in objectives:
                vi, vj = records[i][key], records[j][key]
                if math.isnan(vi) or math.isnan(vj):
                    better_or_equal = False
                    break
                if sign * vj < sign * vi:
                    better_or_equal = False
                    break
                if sign * vj > sign * vi:
                    strictly_better = True
            if better_or_equal and strictly_better:
                flags[i] = False
                break
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    notes: list[str] = []

    for target in TARGETS:
        # Strong-binding cutoff: 10th percentile of the alpha=1 docked scores.
        base = args.root / "docked" / target / "alpha_1.0" / "sweep_docked.csv"
        strong_cutoff = None
        if base.exists():
            rows = load_docked(base)
            sc = []
            for r in rows:
                try:
                    s = float(r["docking_score"])
                except (TypeError, ValueError):
                    continue
                if str(r.get("valid_dock", "True")).strip().lower() in ("true", "1") and s < FAILED_DOCK_SENTINEL:
                    sc.append(s)
            if sc:
                strong_cutoff = float(np.percentile(sc, 10))
                notes.append(
                    f"{target}: strong-binding cutoff = 10th percentile of alpha=1 docked "
                    f"scores = {strong_cutoff:.3f} kcal/mol (n={len(sc)})"
                )
        else:
            notes.append(f"{target}: alpha=1.0 file NOT FOUND at {base}; no cutoff derived")

        per_target: list[dict] = []
        for a in ALPHAS:
            p = args.root / "docked" / target / f"alpha_{a}" / "sweep_docked.csv"
            if not p.exists():
                notes.append(f"{target} alpha={a}: NOT FOUND {p}")
                continue
            rec = summarize_target_alpha(
                load_docked(p), strong_cutoff, TARGET_S_MIN[target],
                TARGET_PROP_RANGE[target],
            )
            rec["target"] = target
            rec["alpha"] = float(a)
            per_target.append(rec)

        if per_target:
            flags = pareto_front(
                per_target,
                [
                    ("rho", +1),
                    ("validity_generated", +1),
                    ("SR20_attempt_level", +1),
                    ("strong_frac_of_attempts", +1),
                ],
            )
            for rec, f in zip(per_target, flags):
                rec["pareto_optimal"] = bool(f)
            all_rows.extend(per_target)

    cols = [
        "target", "alpha", "n_attempted", "n_generated", "n_docked",
        "n_scored_negative", "validity_generated", "dock_success_attempt",
        "rho", "beta", "mean_score", "best_score",
        "SR10_valid_only", "SR10_attempt_level",
        "SR20_valid_only", "SR20_attempt_level",
        "strong_cutoff_used", "strong_frac_of_docked", "strong_frac_of_attempts",
        "pareto_optimal",
    ]
    csv_path = args.out / "c1_docking_attempt_level.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(
                {
                    k: (
                        ""
                        if isinstance(r.get(k), float) and math.isnan(r.get(k))
                        else (round(r[k], 6) if isinstance(r.get(k), float) else r.get(k))
                    )
                    for k in cols
                }
            )
    (args.out / "c1_docking_attempt_level.json").write_text(
        json.dumps({"rows": all_rows, "notes": notes}, indent=2)
    )

    # Generated manuscript table body. This keeps Table 5 synchronized with
    # the latest attempt-level CSV rather than relying on hand transcription.
    labels = {"1syh": "1SYH", "4lde": "4LDE", "6y2f": "6Y2F"}
    tex_lines = []
    previous = None
    for r in all_rows:
        if previous is not None and r["target"] != previous:
            tex_lines.append(r"\midrule")
        label = labels[r["target"]] if r["target"] != previous else ""
        check = r"$\checkmark$" if r["pareto_optimal"] else ""
        tex_lines.append(
            f"{label} & {r['alpha']:.1f} & {r['n_generated']} & {r['n_docked']} & "
            f"{r['validity_generated']:.3f} & {r['dock_success_attempt']:.3f} & "
            f"{r['rho']:.3f} & {r['SR20_attempt_level']:.3f} & "
            f"{r['strong_frac_of_attempts']:.4f} & {check} " + r"\\"
        )
        previous = r["target"]
    (args.out / "docking_pareto_body.tex").write_text("\n".join(tex_lines) + "\n")

    print("=" * 118)
    print("C1 docking attempt-level accounting and Pareto analysis")
    print("=" * 118)
    for n in notes:
        print(f"  note: {n}")
    print()
    hdr = (
        f"{'target':<7}{'alpha':>6}{'n_att':>7}{'n_gen':>7}{'n_dock':>7}{'valid':>8}"
        f"{'dockOK':>8}{'rho':>8}{'beta':>8}{'SR20vo':>8}{'SR20att':>9}"
        f"{'strong%att':>11}{'pareto':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(
            f"{r['target']:<7}{r['alpha']:>6.1f}{r['n_attempted']:>7}{r['n_generated']:>7}"
            f"{r['n_docked']:>7}{r['validity_generated']:>8.3f}{r['dock_success_attempt']:>8.3f}"
            f"{r['rho']:>8.3f}{r['beta']:>8.2f}{r['SR20_valid_only']:>8.3f}"
            f"{r['SR20_attempt_level']:>9.3f}{r['strong_frac_of_attempts']:>11.4f}"
            f"{'YES' if r['pareto_optimal'] else '':>8}"
        )
    print()
    print(f"wrote {csv_path}")
    print(f"wrote {args.out / 'docking_pareto_body.tex'}")


if __name__ == "__main__":
    main()
