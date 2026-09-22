#!/usr/bin/env python3
"""
Revision audit B2: reconstruct attempt-level denominators for the SA-score
conditional-generation comparison (main table + supplementary Table S2).

Motivation
----------
The reviewer observed that per-bucket "attempts" in supplementary Table S2 sum
to the aggregate *valid* count rather than to the number of generation attempts,
producing an apparent 100% per-bucket validity for STAR-VAE and REINVENT 4.

This script recomputes, per model and per bucket:
    n_attempted, n_valid, validity, valid-only SR10/SR20, attempt-level SR10/SR20

Denominator conventions
-----------------------
* Attempts are defined by the *requested* label, before any validity filtering.
  Bucket membership is therefore a deterministic function of y* and is known for
  every attempt even when the attempt produced no parseable molecule.
* STAR-VAE / MolGPT sweep 1,001 y* levels x 3 samples = 3,003 attempts. Bucket
  bounds follow the generation code (exp17c_sa_score_generate.py:66-76):
      high: [0.000, 0.333)   mid: [0.333, 0.667)   low: [0.667, 1.001)
  NOTE: those code-level names are inverted relative to the manuscript's
  presentation (manuscript "low y*" == easy synthesis == y* near 0). This
  script reports buckets by y* range and does not rely on the code's labels.
* REINVENT 4 ran three separate campaigns of num_smiles=1001 each
  (configs/sampling_sa_{low,mid,high}.toml), with unique_molecules=true. The
  saved CSVs contain only unique valid survivors, so attempts are recovered as
  1,001 per campaign and cross-checked against the "Removed N invalid SMILES"
  lines in logs/sampling_sa_*.log.
  IMPORTANT: that log message is emitted by reinvent's filter_valid(), which
  keeps only SmilesState.VALID and therefore removes DUPLICATE rows as well as
  INVALID ones. For the low-y* campaign all 640 written SMILES are RDKit
  parseable and mutually distinct, and the log contains zero sanitization
  failures, so essentially all 361 removals are duplicates rather than invalid
  chemistry. The n_val/n_att ratio for REINVENT is therefore a unique-valid
  *retention* rate, not an RDKit validity rate; it is reported as such.
* DrugDiff emits 1,001 molecules per bucket and is valid by construction.

Attempt-level SR treats an invalid/unscored attempt as a failure:
    SR_attempt = (# valid molecules meeting |y_hat - y*| <= eps) / n_attempted

Usage
-----
    python bmfm_sm_gen/scripts/revision_b2_sa_denominators.py \
        --root paper_results/exp18_conditional_generation_sascore \
        --out  paper_results/revision_audits
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Bucket bounds on the requested label y*, taken from
# bmfm_sm_gen/scripts/exp17c_sa_score_generate.py:66-69
BUCKET_BOUNDS = [
    ("y*<0.333", 0.000, 0.333),
    ("0.333<=y*<0.667", 0.333, 0.667),
    ("y*>=0.667", 0.667, 1.001),
]

SWEEP_LEVELS = 1001
SAMPLES_PER_LEVEL = 3
EPS = {"SR10": 0.10, "SR20": 0.20}


def assign_bucket(y: float) -> str:
    for name, lo, hi in BUCKET_BOUNDS:
        if lo <= y < hi:
            return name
    return BUCKET_BOUNDS[-1][0]


def expected_sweep_attempts() -> dict[str, int]:
    """Attempts per bucket for a 1,001-level x 3-sample uniform y* sweep."""
    counts: dict[str, int] = defaultdict(int)
    for i in range(SWEEP_LEVELS):
        y = round(i / (SWEEP_LEVELS - 1), 6)
        counts[assign_bucket(y)] += SAMPLES_PER_LEVEL
    return dict(counts)


def read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def summarize(
    rows: list[dict],
    attempts: dict[str, int],
    y_field: str,
    yhat_field: str,
    scored_field: str | None,
) -> dict[str, dict]:
    """Aggregate valid/scored molecules into per-bucket statistics."""
    per: dict[str, dict] = {
        b: {"n_valid": 0, "n_scored": 0, "hit10": 0, "hit20": 0, "yhat_sum": 0.0}
        for b, _, _ in BUCKET_BOUNDS
    }

    for r in rows:
        y_star = float(r[y_field])
        bucket = assign_bucket(y_star)
        per[bucket]["n_valid"] += 1

        if scored_field is not None and r.get(scored_field, "True") not in ("True", "true", "1"):
            continue
        raw = r.get(yhat_field, "")
        if raw in ("", None):
            continue
        y_hat = float(raw)
        per[bucket]["n_scored"] += 1
        per[bucket]["yhat_sum"] += y_hat
        if abs(y_hat - y_star) <= EPS["SR10"]:
            per[bucket]["hit10"] += 1
        if abs(y_hat - y_star) <= EPS["SR20"]:
            per[bucket]["hit20"] += 1

    out: dict[str, dict] = {}
    for bucket, d in per.items():
        n_att = attempts.get(bucket, 0)
        n_val, n_scored = d["n_valid"], d["n_scored"]
        out[bucket] = {
            "n_attempted": n_att,
            "n_valid": n_val,
            "n_scored": n_scored,
            "validity": (n_val / n_att) if n_att else float("nan"),
            "mean_yhat": (d["yhat_sum"] / n_scored) if n_scored else float("nan"),
            "SR10_valid_only": (d["hit10"] / n_scored) if n_scored else float("nan"),
            "SR20_valid_only": (d["hit20"] / n_scored) if n_scored else float("nan"),
            "SR10_attempt_level": (d["hit10"] / n_att) if n_att else float("nan"),
            "SR20_attempt_level": (d["hit20"] / n_att) if n_att else float("nan"),
        }
    return out


def reinvent_attempts(root: Path) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """
    Recover REINVENT attempts per campaign.

    Returns (attempts_by_bucket, removed_by_campaign, notes).
    Each campaign requests num_smiles molecules; the written CSV holds only
    unique valid survivors. The log records how many invalid SMILES were removed.
    """
    notes: list[str] = []
    attempts: dict[str, int] = {}
    removed: dict[str, int] = {}

    # y* actually used per campaign. NOTE: the merged sweep file's "bucket"
    # column uses the generation code's inverted naming (code "high" == easy
    # synthesis == y* near 0), while the campaign filenames use the chemical
    # convention (low SA == easy). We therefore key on the campaign's SMILES
    # membership rather than on either label, to avoid crossing the two.
    sweep = read_rows(root / "reinvent4" / "generated" / "sweep.csv")
    smiles_to_y = {r["smiles"]: float(r["conditioning_y_star"]) for r in sweep}
    campaign_y: dict[str, float] = {}
    for campaign in ("low", "mid", "high"):
        path = root / "reinvent4" / "generated" / f"sampling_sa_{campaign}.csv"
        if not path.exists():
            continue
        ys = {
            smiles_to_y[r["SMILES"]]
            for r in read_rows(path)
            if r.get("SMILES") in smiles_to_y
        }
        if len(ys) == 1:
            campaign_y[campaign] = ys.pop()
        elif ys:
            notes.append(
                f"REINVENT {campaign}: AMBIGUOUS y* across campaign members: {sorted(ys)}"
            )

    for campaign in ("low", "mid", "high"):
        cfg = root / "reinvent4" / "configs" / f"sampling_sa_{campaign}.toml"
        log = root / "reinvent4" / "logs" / f"sampling_sa_{campaign}.log"

        num_smiles = None
        if cfg.exists():
            m = re.search(r"^num_smiles\s*=\s*(\d+)", cfg.read_text(), re.M)
            if m:
                num_smiles = int(m.group(1))
        n_removed = None
        if log.exists():
            m = re.search(r"Removed (\d+) invalid SMILES", log.read_text())
            if m:
                n_removed = int(m.group(1))

        if num_smiles is None:
            notes.append(f"REINVENT {campaign}: num_smiles NOT FOUND in config")
            continue

        csv_rows = len(read_rows(root / "reinvent4" / "generated" / f"sampling_sa_{campaign}.csv"))
        removed[campaign] = n_removed if n_removed is not None else -1

        # The requested y* determines which y*-bucket these attempts belong to.
        y_star = campaign_y.get(campaign)
        bucket = assign_bucket(y_star) if y_star is not None else campaign
        attempts[bucket] = attempts.get(bucket, 0) + num_smiles

        if n_removed is not None:
            implied = num_smiles - n_removed
            flag = "OK" if implied == csv_rows else "MISMATCH"
            notes.append(
                f"REINVENT {campaign}: requested={num_smiles} removed_invalid={n_removed} "
                f"implied_valid={implied} csv_rows={csv_rows} [{flag}] "
                f"y*={y_star} -> bucket {bucket}"
            )
        else:
            notes.append(
                f"REINVENT {campaign}: requested={num_smiles} csv_rows={csv_rows} "
                f"(no 'Removed' line found) y*={y_star} -> bucket {bucket}"
            )
    return attempts, removed, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", default="12.0", help="STAR-VAE guidance scale to audit")
    args = ap.parse_args()

    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=True)

    sweep_attempts = expected_sweep_attempts()
    results: dict[str, dict] = {}
    notes: list[str] = [
        f"Uniform sweep attempts per bucket ({SWEEP_LEVELS} levels x "
        f"{SAMPLES_PER_LEVEL} samples = {SWEEP_LEVELS * SAMPLES_PER_LEVEL}): {sweep_attempts}"
    ]

    # ---- STAR-VAE -------------------------------------------------------
    sv = root / "starvae" / "scored" / "sa_score" / f"alpha_{args.alpha}" / "sweep_scored.csv"
    if sv.exists():
        rows = read_rows(sv)
        results[f"STAR-VAE (alpha={args.alpha})"] = summarize(
            rows, sweep_attempts, "conditioning_y_star", "sa_score_norm", "scored"
        )
        notes.append(f"STAR-VAE: {len(rows)} scored rows from {sv}")
    else:
        notes.append(f"STAR-VAE: NOT FOUND {sv}")

    # ---- MolGPT ---------------------------------------------------------
    mg = root / "molgpt" / "scored" / "sweep_scored.csv"
    if mg.exists():
        rows = read_rows(mg)
        results["MolGPT"] = summarize(
            rows, sweep_attempts, "conditioning_y_star", "sa_score_normalized", "scored"
        )
        notes.append(f"MolGPT: {len(rows)} scored rows from {mg}")
    else:
        notes.append(f"MolGPT: NOT FOUND {mg}")

    # ---- REINVENT 4 -----------------------------------------------------
    rv = root / "reinvent4" / "scored" / "sweep_scored.csv"
    if rv.exists():
        rv_attempts, _removed, rv_notes = reinvent_attempts(root)
        notes.extend(rv_notes)
        rows = read_rows(rv)
        results["REINVENT 4"] = summarize(
            rows, rv_attempts, "conditioning_y_star", "sa_score_normalized", "scored"
        )
        notes.append(f"REINVENT 4: {len(rows)} scored rows from {rv}")
    else:
        notes.append(f"REINVENT 4: NOT FOUND {rv}")

    # ---- DrugDiff -------------------------------------------------------
    dd = root / "drugdiff" / "scored" / "sweep_scored.csv"
    if dd.exists():
        rows = read_rows(dd)
        dd_attempts: dict[str, int] = defaultdict(int)
        for r in rows:
            dd_attempts[assign_bucket(float(r["conditioning_y_star"]))] += 1
        results["DrugDiff"] = summarize(
            rows, dict(dd_attempts), "conditioning_y_star", "sa_score_normalized", "scored"
        )
        notes.append(
            f"DrugDiff: {len(rows)} scored rows from {dd}; valid by construction, so "
            f"attempts are taken as emitted molecules per bucket {dict(dd_attempts)}"
        )
    else:
        notes.append(f"DrugDiff: NOT FOUND {dd}")

    # ---- Write outputs --------------------------------------------------
    csv_path = out / "b2_sa_denominators.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model", "bucket", "n_attempted", "n_valid", "n_scored", "validity",
                "mean_yhat", "SR10_valid_only", "SR20_valid_only",
                "SR10_attempt_level", "SR20_attempt_level",
            ]
        )
        for model, buckets in results.items():
            for bname, _, _ in BUCKET_BOUNDS:
                d = buckets.get(bname)
                if not d:
                    continue

                def fmt(x, nd=4):
                    return "" if (isinstance(x, float) and math.isnan(x)) else (
                        f"{x:.{nd}f}" if isinstance(x, float) else x
                    )

                w.writerow(
                    [
                        model, bname, d["n_attempted"], d["n_valid"], d["n_scored"],
                        fmt(d["validity"]), fmt(d["mean_yhat"]),
                        fmt(d["SR10_valid_only"]), fmt(d["SR20_valid_only"]),
                        fmt(d["SR10_attempt_level"]), fmt(d["SR20_attempt_level"]),
                    ]
                )

    (out / "b2_sa_denominators.json").write_text(
        json.dumps({"results": results, "notes": notes}, indent=2)
    )

    # ---- Console report -------------------------------------------------
    print("=" * 100)
    print("B2 SA-score denominator audit")
    print("=" * 100)
    for n in notes:
        print(f"  note: {n}")
    print()
    hdr = (
        f"{'model':<26} {'bucket':<17} {'n_att':>6} {'n_val':>6} {'valid':>7} "
        f"{'SR10_vo':>8} {'SR10_att':>9} {'SR20_vo':>8} {'SR20_att':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for model, buckets in results.items():
        tot_att = tot_val = 0
        for bname, _, _ in BUCKET_BOUNDS:
            d = buckets.get(bname)
            if not d:
                continue
            tot_att += d["n_attempted"]
            tot_val += d["n_valid"]
            print(
                f"{model:<26} {bname:<17} {d['n_attempted']:>6} {d['n_valid']:>6} "
                f"{d['validity']:>7.3f} {d['SR10_valid_only']:>8.3f} "
                f"{d['SR10_attempt_level']:>9.3f} {d['SR20_valid_only']:>8.3f} "
                f"{d['SR20_attempt_level']:>9.3f}"
            )
        print(
            f"{'  -> TOTAL':<26} {'':<17} {tot_att:>6} {tot_val:>6} "
            f"{(tot_val / tot_att if tot_att else float('nan')):>7.3f}"
        )
        print()
    print(f"wrote {csv_path}")
    print(f"wrote {out / 'b2_sa_denominators.json'}")


if __name__ == "__main__":
    main()
