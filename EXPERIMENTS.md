# STAR-VAE Experiment Guide

This public guide covers supported entry points and the inputs a user must
provide. It intentionally omits private filesystem layouts, scheduler wrappers,
and manuscript-revision-only reanalyses. Paths below are user-controlled and
relative to the repository root unless shown as environment variables.

This is a source-checkout guide. The wheel intentionally excludes
`bmfm_sm_gen/scripts/`, `configs/`, and `examples/`. A wheel installation exposes
only the library and the three `star-vae-*` console commands. Paper experiment
commands below require the repository checkout plus separately acquired data
and checkpoints.

## Setup

```bash
uv sync --frozen --extra dev --extra analysis
source .venv/bin/activate
cp .env.example .env
source .env
```

The project does not distribute datasets. Obtain each dataset from its provider,
review its license and access terms, and transform it into the columns required
by the selected config. Checkpoints are separate. See [README.md](README.md#checkpoints).

Open <https://ibm.ent.box.com/v/star-vae-checkpoints> in a browser, download
`star_vae_checkpoints_v1.0.0.zip`, and place it in the repository root. Then run
`bash scripts/install_checkpoints.sh` to verify and extract the local archive.
The script does not download it. The installer creates the complete
`checkpoints/` hierarchy and places every model and vocabulary at the canonical
paths used below. Do not manually flatten or rename archive members. See
[`checkpoint_bundle.json`](checkpoint_bundle.json) for the machine-readable
checkpoint-to-vocabulary mapping.

PyTorch `.ckpt` loading can execute code. Use only trusted checkpoints.

CI checks all installed entry points but only the source-script help commands
listed in `.github/workflows/ci.yml`. It does not certify every historical or
specialized script. See [SUPPORTED_SCRIPTS.md](SUPPORTED_SCRIPTS.md) for the
support boundary.

## Protocol Definitions

| Protocol | Implementation |
|---|---|
| Random-prior VAE | Draw `z ~ N(0,I)` and decode without a molecular seed or property label. |
| Seed-based VAE | Encode each supplied molecule and sample around its posterior latent mean. This is not unconditional generation. |
| Label-conditioned CVAE | Decode with a target label, sampling `p(z|y)` when the checkpoint enables a conditional prior, or `N(0,I)` when explicitly selected or unavailable. |

Generic conditional generation does not implement classifier-free guidance.
Specialized `exp17c_*_generate.py` scripts implement paper-specific CFG logits,
use fixed experiment paths, and are not general replacements for the generic
generator.

## Environment Variables

| Variable | Default after `source .env` |
|---|---|
| `STAR_VAE_ROOT` | Directory containing `.env` |
| `CHECKPOINT_DIR` | `$STAR_VAE_ROOT/checkpoints` |
| `DATA_DIR` | `$STAR_VAE_ROOT/data` |
| `OUTPUT_DIR` | `$STAR_VAE_ROOT/outputs` |
| `VOCAB_FILE` | `$CHECKPOINT_DIR/vae_pubchem80m/vocab.json` |
| `PRETRAINED_VAE_CKPT` | `$CHECKPOINT_DIR/vae_pubchem80m/last.ckpt` |
| `PRETRAINED_VAE_MOSES_CKPT` | `$CHECKPOINT_DIR/vae_moses/last.ckpt` |
| `SMILES_VOCAB_MOSES` | Historical 28-token SMILES vocabulary asset. The bundled MOSES SELFIES VAE does not use it |
| `CKPT_SA_SCORE` | `$CHECKPOINT_DIR/cvae_sa_score/last.ckpt` |
| `CKPT_1SYH`, `CKPT_4LDE`, `CKPT_6Y2F` | Original Tartarus CVAEs |
| `CKPT_1SYH_STRAT`, `CKPT_4LDE_STRAT`, `CKPT_6Y2F_STRAT` | Stratified Tartarus CVAEs |
| `CKPT_B3DB` | B3DB CVAE |

The full dataset-path template is in [.env.example](.env.example).
Checkpoint-backed compatibility tests use the same variables and can be run
with `pytest -m requires_ckpt` after archive installation.

## Generation

### Random-prior VAE

```bash
python examples/generate_unconditional.py \
  --checkpoint "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --num-samples 100 \
  --seed 42 \
  --output "$OUTPUT_DIR/random.csv"
```

### Seed-based VAE

```bash
python examples/generate_unconditional.py \
  --checkpoint "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --input-smiles "CC1=CC=CC=C1" \
  --noise-scale 0.5 \
  --num-samples 100 \
  --output "$OUTPUT_DIR/seeded.csv"
```

Bulk seed-based generation requires a CSV and a named SMILES column. With
`--cluster-sample`, the cluster column is also required.

```bash
star-vae-generate-bulk \
  --ckpt-path "$PRETRAINED_VAE_CKPT" \
  --vocab-file "$VOCAB_FILE" \
  --input-csv /path/to/seeds.csv \
  --smiles-col target_smiles \
  --output-csv "$OUTPUT_DIR/seeded_bulk.csv" \
  --num-samples 1 \
  --sampling-scheme around_mu
```

### Label-conditioned CVAE

The generic module uses hyphenated aliases below (underscore forms are also
accepted). `--prior-mode auto` uses a learned conditional prior when available.

```bash
python -m bmfm_sm_gen.launch.run_generate_random_molecules_conditional \
  --checkpoint "$CKPT_SA_SCORE" \
  --vocab-file "$VOCAB_FILE" \
  --target-label 0.25 \
  --prior-mode auto \
  --num-samples 100 \
  --seed 101 \
  --output-file "$OUTPUT_DIR/sa_score_025.csv"
```

This generic module supports target-label generation or label sampling based on
`--training-set`, `--label-column`, and `--num-top-bins`. It has no CFG option.

## MOSES

Obtain `dataset_v1.csv` from the
[MOSES project](https://github.com/molecularsets/moses). The preprocessing
script expects columns `SMILES` and `SPLIT` and uses MOSES train as training and
MOSES test as validation.

Despite the historical filename `smiles_vocab_moses.json`, the released
`vae_moses` checkpoint is a `SELFIESTokenizer` model and uses the same bundled
328-token PubChem SELFIES vocabulary as the PubChem VAE (`$VOCAB_FILE`). The
28-token MOSES SMILES vocabulary is retained only as a historical reproduction
asset for SMILES-tokenization work and must not be supplied to `vae_moses`.

```bash
export MOSES_RAW_CSV=/path/to/moses/dataset_v1.csv
export MOSES_PROCESSED_DIR="$DATA_DIR/moses_processed"
python bmfm_sm_gen/data_prep/prep_moses_selfies.py
python -m bmfm_sm_gen.launch.run_train fit \
  --config configs/model_vae_train_moses.yaml \
  --data.init_args.train_csv "$MOSES_PROCESSED_DIR/train_selfies.csv" \
  --data.init_args.val_csv "$MOSES_PROCESSED_DIR/val_selfies.csv"
```

The full metric script additionally requires an independently installed and
validated upstream MOSES environment. The published `molsets` release does not
build reliably on Python 3.11, so it is not included in the STAR-VAE lock. Its
random-prior mode is the default. Add `--conditioned` for the distinct
seed-based benchmark. Verify MOSES's FCD runtime requirements on the target
platform.

```bash
# Activate a separately validated MOSES benchmark environment first.
python bmfm_sm_gen/scripts/exp02_run_moses_full_metrics.py \
  --ckpt "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --moses-csv "$MOSES_RAW_CSV" \
  --n-gen 30000 \
  --batch-size 256 \
  --temperature 1.1 \
  --top-k 20 \
  --seed 42 \
  --output-dir "$OUTPUT_DIR/exp02_moses" \
  --n-jobs 8 \
  --device cuda
```

Use `--device cpu` if CUDA is unavailable. FCD and generation will be slower.
`--n-gen` is the requested generation count. Random-prior generation retains one
entry per decode attempt, including failed decodes. Seed-conditioned generation
retains only successful decodes. The run manifest therefore records requested,
decoded, RDKit-valid, and submitted counts separately. MOSES then computes some
metrics on its own filtered/scored subsets. When evaluating a pre-generated CSV,
pass `--requested-attempts N` if the original attempt count is known. Otherwise
the manifest records it as unknown. MOSES `valid` is explicitly labeled as
submitted-population validity, because the submitted list may already have been
filtered during decoding.

## Custom Conditional Fine-tuning

The generic config expects separate CSVs with a `selfies` column and a
`y_normalized_minmax` label by default. Change `label_columns` if your schema is
different. A separate validation file is required for held-out validation.

```bash
python -m bmfm_sm_gen.launch.run_train fit \
  --config configs/model_vae_zvela_label_conditioning_lora_prior_template.yaml \
  --data.init_args.train_csv /path/to/train.csv \
  --data.init_args.val_csv /path/to/validation.csv \
  --trainer.default_root_dir "$OUTPUT_DIR/cvae_custom"
```

Training resumes automatically if `last.ckpt` exists under the selected
`trainer.default_root_dir/checkpoints`. Use a new output directory when a fresh
run is intended.

## SA-score Fine-tuning

The preparation script expects a user-provided CSV containing `sascore` and a
model-ready molecular representation such as `selfies`. It writes the full
normalized CSV and deterministic row-level train and validation CSVs. The
defaults are a 20% validation fraction and seed 42, matching the filenames used
by the training config. It does not canonicalize molecules, group related
structures, or audit overlap, so this split must not be described as a scaffold
split or leakage audit.

```bash
export PUBCHEM_10K_SA=/path/to/pubchem_sa_sample.csv
export SA_SCORE_TRAIN_CSV="$DATA_DIR/pubchem/sa_score_minmax_train.csv"
export SA_SCORE_VAL_CSV="$DATA_DIR/pubchem/sa_score_minmax_val.csv"

python bmfm_sm_gen/data_prep/prep_sa_score_minmax.py \
  --input "$PUBCHEM_10K_SA" \
  --output-dir "$DATA_DIR/pubchem" \
  --val-fraction 0.2 \
  --seed 42

python -m bmfm_sm_gen.launch.run_train fit \
  --config configs/model_vae_sa_score_minmax_lora_prior.yaml
```

Record the row-level split method, random seed, and row counts in any derived
result. Perform and report a separate canonical or scaffold overlap check if a
claim requires one. The preparation script does not provide that evidence.

## Tartarus Stratified Fine-tuning

Obtain Tartarus data from the
[Tartarus project](https://github.com/aspuru-guzik-group/tartarus). The
preparation script expects an input directory containing
`tartarus_<target>_minmax.csv`. Each file must include `smiles`, `selfies`, and
the target score column. It canonicalizes and deduplicates before a seeded
bucket-balanced split, restricts oversampling with score jitter to training,
and asserts zero canonical-SMILES overlap.

```bash
python bmfm_sm_gen/scripts/exp16_prep_stratified_tartarus.py \
  --target 1syh \
  --input-dir /path/to/prepared/tartarus \
  --output-dir "$DATA_DIR/tartarus/stratified" \
  --val-fraction 0.2 \
  --seed 42

python -m bmfm_sm_gen.launch.run_train fit \
  --config configs/model_vae_1syh_stratified.yaml
```

Equivalent configs exist for 4LDE and 6Y2F. They initialize from `CKPT_4LDE`
and `CKPT_6Y2F`, respectively.

The non-CFG stratified generation script has an experiment-specific interface:

```bash
python bmfm_sm_gen/scripts/exp16_generate_tartarus.py \
  --target 1syh \
  --seed 42 \
  --checkpoint-override "$CKPT_1SYH_STRAT" \
  --label-space percentile \
  --num-samples 100 \
  --output-root "$OUTPUT_DIR/tartarus_generated"
```

This script samples standard-normal latent vectors and supplies percentile
labels to the decoder. It is not the generic conditional-prior path and does
not apply CFG. The percentile convention follows ascending raw docking score:
`0` is the strongest end (most negative score) and `1` is the weakest end (less
negative, approaching zero).

## Tartarus Docking

Docking is not installed by STAR-VAE. It requires a compatible Tartarus
checkout, receptor assets, QVina, Open Babel (`obabel` and `obenergy`), and an
environment satisfying Tartarus' dependencies. QVina is external recomputation
using the same procedure that produced training labels, not an independent
oracle. Scores can be stochastic.

Single-file invocation:

```bash
python bmfm_sm_gen/scripts/exp16_dock_tartarus.py \
  --target 1syh \
  --seed 42 \
  --bucket high \
  --input "$OUTPUT_DIR/tartarus_generated/1syh/seed_42/high.csv" \
  --output "$OUTPUT_DIR/tartarus_docked/1syh/seed_42/high_docked.csv" \
  --tartarus-root /path/to/tartarus-checkout \
  --tartarus-env /path/to/tartarus-environment \
  --n-workers 8
```

Docking thousands of molecules is a long CPU workload and may consume
substantial temporary storage. Validate external executables and receptor files
on a small input before scaling up.

## Specialized CFG Scripts

The paper's CFG implementation is embedded in specialized scripts:

```bash
python bmfm_sm_gen/scripts/exp17c_sa_score_generate.py --alpha 1.0 --seed 42 --smoke-test
python bmfm_sm_gen/scripts/exp17c_generate.py --target 1syh --alpha 1.0 --seed 42 --smoke-test
```

For portable runs, both scripts accept `--checkpoint`, `--base-checkpoint`,
`--vocab`, and `--output-dir`. `--smoke-test` is hyphenated. With
`exp17c_generate.py --target all`, checkpoint overrides must contain a literal
`{target}` placeholder. Omitting the overrides preserves the historical
`paper_results/` and `OUTPUT_DIR` defaults. Full sweeps omit `--smoke-test` and
request 3,003 attempts per target/seed. Their `sweep.csv` files contain valid
decoded molecules only. The scripts log attempted and valid counts separately.

## Ablation Utilities

Ablation parser spelling is mixed and must be used exactly: `--smoke_test`,
`--run_id`, `--n_samples`, `--generated_dir`, `--train_csv`, and `--label_col`
use underscores, while training uses the hyphenated `--output-suffix`. The
sanitized public subset under `configs/ablations/` isolates the conditional
prior while preserving FiLM conditioning. These are new executable reproduction
protocols, not claims about unavailable exact historical run configurations. A
reference configuration can be trained, generated, and evaluated as follows:

```bash
python bmfm_sm_gen/scripts/exp19_train_ablation.py \
  --config configs/ablations/sa_reference_public.yaml \
  --seed 42 \
  --output-suffix _ablation

python bmfm_sm_gen/scripts/exp19_generate_ablation.py \
  --run_id ref_selfies_pubchem \
  --checkpoint "$OUTPUT_DIR/ablations/sa_reference_public_ablation/checkpoints/last.ckpt" \
  --config configs/ablations/sa_reference_public.yaml \
  --outdir "$OUTPUT_DIR/exp19/ref_selfies_pubchem" \
  --n_samples 1000 \
  --seeds 42 0 101

python bmfm_sm_gen/scripts/exp19_evaluate_ablation.py \
  --run_id ref_selfies_pubchem \
  --generated_dir "$OUTPUT_DIR/exp19/ref_selfies_pubchem" \
  --train_csv "$SA_SCORE_TRAIN_CSV" \
  --label_col y_normalized_minmax \
  --outdir "$OUTPUT_DIR/exp19/analysis" \
  --checkpoint "$OUTPUT_DIR/ablations/sa_reference_public_ablation/checkpoints/last.ckpt"
```

These are expensive GPU/analysis workflows. `--smoke_test` on the training
script runs only two steps, but still requires the config's data, vocabulary,
and pretrained checkpoint. Generation schedules exactly `--n_samples` attempts
for each target and seed. Each `generated.csv` preserves one row per attempt
with `attempt_id`, `smiles`, and `valid`. A failed batch or missing decode is
recorded as an invalid row rather than retried away. Attempt-level validity is
therefore `sum(valid) / len(rows)`. SA-score analyses must exclude missing or
invalid SMILES from valid-only property statistics while retaining all rows for
attempt-level denominators. The current evaluator predates this output correction:
its property metrics drop rows without an SA score, while its RDKit validity
path reparses the `smiles` column and may require blank cells to be normalized
before use. Treat the CSV's explicit `valid` column as the attempt record and
inspect evaluator outputs rather than assuming all metrics use the same
denominator. The evaluator also reads its fixed three default seeds and target
buckets, so do not pass custom `--seeds` or `--targets` to generation unless the
evaluation code is adapted separately.

## Optional Benchmarks and Baselines

- GuacaMol requires `uv sync --frozen --extra guacamol` and user-acquired reference
  data. Inspect the goal-directed interface with
  `python -m bmfm_sm_gen.launch.run_guacomol_goal_directed_benchmark --help`.
- B3DB data must be obtained from the
  [B3DB project](https://github.com/theochem/B3DB).
- ChEMBL-derived silver labels and target checkpoints are not distributed.
- REINVENT4, MolGPT, and DrugDiff comparisons require separate upstream
  repositories, dependencies, weights, and license review.

Commands for those systems are not presented as primary commands because a
fresh STAR-VAE checkout does not contain their required assets.

## Resource and Reproducibility Notes

- Full PubChem pretraining is multi-GPU scale and unsuitable as an installation
  test. Runtime depends heavily on accelerator count and data throughput.
- CPU generation is supported but slow. Docking is CPU-intensive and external.
- Seeds do not guarantee bitwise-identical CUDA execution.
- Historical result directories may predate current attempt-preserving scripts,
  may contain valid-only CSVs, and may lack exact configs, training seeds, or run
  manifests. Do not claim exact reproduction from a current command unless its
  checkpoint, config, data version, seed, and denominator are all recorded.
- Report attempted, valid decoded, externally scored, and successful docking
  counts separately. Never silently replace an attempt-level denominator with a
  valid-only denominator.
- Record dataset versions, preprocessing, split seeds, checkpoint identity,
  package versions, hardware, and every CLI override for a reproducible run.
