# STAR-VAE: A Scalable Latent-Variable Transformer for Controllable Molecular Generation

[![CI](https://github.com/BiomedSciAI/STAR-VAE/actions/workflows/ci.yml/badge.svg)](https://github.com/BiomedSciAI/STAR-VAE/actions/workflows/ci.yml)
[![License: OpenMDW-1.0](https://img.shields.io/badge/License-OpenMDW--1.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02769-b31b1b.svg)](https://arxiv.org/abs/2511.02769)

STAR-VAE is a SELFIES Transformer variational autoencoder for molecular
generation. The main model uses a 12-layer bidirectional encoder, a 12-layer
autoregressive decoder, and a 256-dimensional latent space. Conditional models
add FiLM, LoRA fine-tuning, and optionally a learned conditional prior.

The software is research code. Generated molecules are not guaranteed to be
valid, synthesizable, safe, active, or suitable for clinical use. SELFIES makes
the token representation robust, but decoding and chemistry parsing can still
fail. Reported validity must therefore state its denominator.

## Generation Modes

| Mode | Latent source | Conditioning |
|---|---|---|
| Random-prior VAE | `z ~ N(0, I)` | None |
| Seed-based VAE | Encode a supplied molecule and perturb its latent mean | Structural seed only |
| Label-conditioned CVAE | `p(z|y)` when the checkpoint has a conditional prior, otherwise `N(0, I)` | Target label is also supplied to the decoder |

`examples/generate_conditional.py --prior-mode auto` selects `p(z|y)` when the
loaded checkpoint enables it. `--prior-mode standard` forces `N(0, I)`, and
`--prior-mode conditional` fails if the checkpoint has no conditional prior.
Input-seeded CVAE generation is a separate mode: it perturbs the encoded seed
and conditions the decoder on the requested label.

The generic examples and `run_generate_random_molecules_conditional` do **not**
implement classifier-free guidance (CFG). CFG sweeps used for paper analyses
live in specialized experiment scripts such as `exp17c_*_generate.py`. Their
arguments, label scales, checkpoint assumptions, and output layouts are
experiment-specific.

## Installation

### Supported environment

- CPython 3.11 only (`pyproject.toml` requires `>=3.11,<3.12`).
- PyTorch 2.1.0 and torchvision 0.16.0 are pinned.
- Linux is the supported platform. CI runs the locked CPython 3.11 environment
  on GitHub-hosted `ubuntu-latest` runners and checks package compilation, every
  installed console entry point's help path, a selected set of documented source
  scripts, and bounded CPU tests. It does not execute or validate every script in
  `bmfm_sm_gen/scripts/`. macOS, Windows, GPUs, CUDA, and multi-node training are
  not covered by CI.
- CUDA is strongly recommended for generation and required for practical
  training. CPU generation can be very slow.
- The Python environment requires several GB. The compressed checkpoint archive
  is about 6.1 GB and requires additional temporary space during extraction.

Install the project environment:

```bash
git clone https://github.com/BiomedSciAI/STAR-VAE.git
cd STAR-VAE
uv sync --frozen
source .venv/bin/activate
export CLEARML_ENABLE=False
```

Development tools are optional:

```bash
uv sync --frozen --extra dev
```

Optional features are not installed by default:

```bash
uv sync --frozen --extra analysis  # YAML/Altair analysis utilities
uv sync --frozen --extra guacamol  # GuacaMol and CMA-ES benchmark support
uv sync --frozen --extra web       # Streamlit web application
uv sync --frozen --extra clearml   # optional experiment tracking
```

The former `safe` extra was removed because `safe-mol` is not available from the
project's supported package index. `SAFETokenizer` remains an optional integration
point for environments that install a compatible `safe-mol` distribution by
their own validated method. The locked install and CI do not cover it.

Extras can be combined, for example
`uv sync --frozen --extra dev --extra analysis`.

For a platform-specific CUDA build, follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/)
and keep the versions compatible with the project pins.

## Environment

From the repository root:

```bash
cp .env.example .env
source .env
```

`.env` initializes `STAR_VAE_ROOT` from its own location, so it can be sourced
from another working directory. Python programs do not universally source
`.env`, so source it in your shell before invoking commands. The template disables
optional ClearML integration. ClearML is declared only in the `clearml` extra.
After installing that extra, set `CLEARML_ENABLE=true` and optionally
`CLEARML_PROJECT` and `CLEARML_TASK_NAME` to enable tracking.

The primary checkpoint variables are:

| Variable | Purpose |
|---|---|
| `PRETRAINED_VAE_CKPT` | PubChem-80M VAE |
| `CKPT_SA_SCORE` | SA-score CVAE |
| `CKPT_1SYH`, `CKPT_4LDE`, `CKPT_6Y2F` | Original Tartarus CVAEs |
| `CKPT_1SYH_STRAT`, `CKPT_4LDE_STRAT`, `CKPT_6Y2F_STRAT` | Stratified Tartarus CVAEs |
| `CKPT_B3DB` | B3DB CVAE |
| `VOCAB_FILE` | PubChem-80M SELFIES vocabulary |

## Checkpoints

Open <https://ibm.ent.box.com/v/star-vae-checkpoints> in a web browser and
download `star_vae_checkpoints_v1.0.0.zip`. Browser download is required because
the Box page does not provide a stable direct-download URL for command-line
tools. Place the ZIP in the repository root, then verify and extract it with:

```bash
bash scripts/install_checkpoints.sh
```

The script does not download the archive. It verifies the local ZIP's SHA-256,
validates its contents, and extracts it into `checkpoints/`.

Do not download separate copies of `checkpoint_bundle.json` or
`checkpoints.sha256`. They are small, version-controlled release
metadata included in this source repository. The installer reads these local
files automatically to verify the downloaded ZIP's identity, exact membership,
and extracted-file checksums before installation. Keeping the authoritative
metadata with the source keeps verification independent of the archive host.

For another archive or destination:

```bash
bash scripts/install_checkpoints.sh \
  --zip /path/to/star_vae_checkpoints_v1.0.0.zip \
  --dir /path/to/checkpoints
```

**Integrity status:** the release archive tested with this source has SHA-256
`d47ff7b460bc4eb3bae604407ea6b6067994c31fb7283ecd420e2a0fbdcf3e56`.
The installer verifies this digest by default, validates ZIP structure, rejects
traversal and symlink entries, stages extraction, checks expected files, and
refuses to overwrite a non-empty destination. Use
`--accept-unverified-integrity` only for a deliberately different trusted
archive. The archive is retained after installation.

The curated bundle contains the primary VAE/CVAE checkpoints and ten verified
ablation checkpoints. The unusable PubChem SMILES-tokenization ablation is not
included because its exact 231-token vocabulary is unavailable. Every included
model was load-tested with its declared vocabulary.
See [CHECKPOINTS.md](CHECKPOINTS.md),
[`checkpoint_bundle.json`](checkpoint_bundle.json), and
[`checkpoints.sha256`](checkpoints.sha256) for the model inventory, per-file
identities, scope, and the omitted unsupported vocabulary-dependent ablation.
The executed compatibility record is in
[CHECKPOINT_VERIFICATION.md](CHECKPOINT_VERIFICATION.md).
The curated checkpoints are inference-focused exports. The archive builder
removes local paths, callbacks, optimizer state, logger state, and
training-loop state before publication.
Per-model config provenance, dataset family, training-seed status, paper role,
and historical caveats are recorded in `checkpoint_bundle.json`. `UNKNOWN`
means the released artifacts do not establish the value. Do not infer it
from a generation seed or a current reproduction config.

### Verified checkpoint smoke tests

The release archive above was verified under Python 3.11 on CPU. The PubChem
VAE loaded and produced RDKit-valid molecules in both random-prior and
input-seeded modes. The SA-score CVAE loaded and produced a valid molecule from
its learned conditional prior. The original and stratified 1SYH/4LDE/6Y2F
CVAEs and the B3DB CVAE also passed bounded valid-generation tests. These are
software/checkpoint compatibility checks, not paper-scale quality evaluations.

To repeat the checkpoint-backed tests after sourcing `.env`:

```bash
pytest -m requires_ckpt
```

The complete canonical directory tree and maintainer rebuild command are in
[CHECKPOINTS.md](CHECKPOINTS.md). No manual checkpoint or vocabulary placement
is required.

**Trust warning:** PyTorch checkpoints can contain pickle data. Loading an
untrusted `.ckpt` may execute code. Only install and load checkpoints obtained
from a source you trust. See [SECURITY.md](SECURITY.md).

## Quick Start

The standalone examples are the clearest public interface. They create output
directories and expose an explicit device and random seed.

Random-prior VAE generation:

```bash
python examples/generate_unconditional.py \
  --checkpoint "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --num-samples 32 \
  --seed 42 \
  --output "$OUTPUT_DIR/random.csv"
```

Seed-based VAE generation:

```bash
python examples/generate_unconditional.py \
  --checkpoint "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --input-smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --noise-scale 0.5 \
  --num-samples 32 \
  --output "$OUTPUT_DIR/aspirin_analogues.csv"
```

Label-conditioned conditional-prior generation:

```bash
python examples/generate_conditional.py \
  --checkpoint "$CKPT_SA_SCORE" \
  --vocab "$VOCAB_FILE" \
  --target-label 0.25 \
  --prior-mode conditional \
  --num-samples 32 \
  --output "$OUTPUT_DIR/sa_score_025.csv"
```

The installed VAE command is a Click group, so its single-molecule interface
requires the `cli` subcommand:

```bash
star-vae-generate cli \
  --ckpt-path "$PRETRAINED_VAE_CKPT" \
  --vocab-file "$VOCAB_FILE" \
  --num-samples 32 \
  --unconstrained \
  --output-csv "$OUTPUT_DIR/random_cli.csv"
```

See [examples/README.md](examples/README.md) for all example modes and
[EXPERIMENTS.md](EXPERIMENTS.md) for dataset preparation, training, benchmarks,
and specialized scripts.

### Wheel and source-checkout boundary

The built wheel provides the `bmfm_sm_gen` library and the three installed
commands `star-vae-generate`, `star-vae-generate-bulk`, and `star-vae-train`.
The one-off programs under `bmfm_sm_gen/scripts/`, repository-root `configs/`,
standalone `examples/`, and experiment documentation are repository assets, not
installed wheel contents. Commands in `EXPERIMENTS.md` must therefore be run
from a repository checkout. Installing the wheel alone does not reproduce the
paper experiments or provide datasets, checkpoints, configs, or result files.

## Data and External Tools

Datasets are not bundled. Users are responsible for obtaining PubChem, MOSES,
GuacaMol, B3DB, Tartarus, and ChEMBL data from their respective providers,
accepting applicable licenses or access terms, and preparing the exact columns
required by a selected config. Do not assume a path under `data/` exists in a
fresh checkout.

Additional workflows require optional extras or external software:

- MOSES metrics require the upstream MOSES package and its FCD dependencies.
  The published `molsets` release does not build reliably on Python 3.11, so
  install MOSES in a separately validated benchmark environment rather than
  assuming it is part of the locked STAR-VAE environment.
- GuacaMol evaluation requires `uv sync --frozen --extra guacamol` and user-acquired
  reference data.
- Tartarus docking requires a separate Tartarus checkout, QVina, Open Babel,
  receptor files, and substantial CPU time and storage.
- Baseline comparisons require their upstream projects, model weights, and
  licenses. They are not installed by `uv sync`.

Large pretraining and fine-tuning runs are expensive. The published pretraining
was multi-GPU scale. Exact runtime depends on hardware, storage throughput,
dataset preprocessing, and distributed settings. Docking thousands of molecules
is also an external, long-running workload. Start with `--help` and small,
user-supplied inputs before scheduling full experiments.

## Training and Evaluation

MOSES preprocessing and the small MOSES training config:

```bash
export MOSES_RAW_CSV=/path/to/moses/dataset_v1.csv
export MOSES_PROCESSED_DIR="$DATA_DIR/moses_processed"
python bmfm_sm_gen/data_prep/prep_moses_selfies.py
python -m bmfm_sm_gen.launch.run_train fit \
  --config configs/model_vae_train_moses.yaml
```

MOSES random-prior benchmark (optional MOSES package required):

```bash
python bmfm_sm_gen/scripts/exp02_run_moses_full_metrics.py \
  --ckpt "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --moses-csv "$MOSES_RAW_CSV" \
  --n-gen 30000 \
  --output-dir "$OUTPUT_DIR/exp02_moses"
```

SA-score fine-tuning requires a user-provided CSV with `sascore` and a
model-ready molecular representation such as `selfies`. The preparation script
normalizes the labels and writes deterministic train and validation files using
the requested row-level random split. The default is 20% validation with seed
42. This is not a scaffold split or a canonical-SMILES overlap audit.

```bash
python bmfm_sm_gen/data_prep/prep_sa_score_minmax.py \
  --input "$PUBCHEM_10K_SA" \
  --output-dir "$DATA_DIR/pubchem" \
  --val-fraction 0.2 \
  --seed 42
python -m bmfm_sm_gen.launch.run_train fit \
  --config configs/model_vae_sa_score_minmax_lora_prior.yaml
```

## Reproducibility Limits

- Generation seeds control pseudorandom sampling but do not guarantee bitwise
  reproducibility across devices, CUDA versions, or kernels.
- Generic examples preserve one row per attempt (failed decodes have an empty
  SMILES cell), while several historical specialized scripts write valid rows
  only. Record requested/actual attempts, valid decoded
  rows, externally scored rows, docking attempts, and successful docking rows
  separately. Compute validity against attempts rather than against the
  valid-only CSV.
- The SA-score preparer creates a seeded row-level train/validation split but
  does not canonicalize molecules or audit structural overlap. The Tartarus
  preparer canonicalizes and deduplicates before splitting, restricts
  oversampling and score jitter to training, and asserts zero canonical-SMILES
  overlap between its generated splits.
- Tartarus percentile labels increase with the raw docking score: `0` denotes
  the strongest end (more negative score), while `1` denotes the weakest end
  (less negative, approaching zero). This convention applies to percentile
  labels produced by the current Tartarus ranking function. This convention is
  also stated by the conditional example's `--help` output.
- Dataset acquisition and compliance remain external to this repository. The
  released STAR-VAE checkpoint weights are provided as Model Materials under
  OpenMDW-1.0. Datasets are not included.
- Historical raw outputs, exact private-era configs, and revision audit bundles
  are not part of this source release. `checkpoint_bundle.json` labels unknown
  seeds and non-exact public configs explicitly. Current commands are executable
  protocols rather than proof of bitwise historical reproduction.

## Citation

```bibtex
@article{kwon2026starvae,
  title   = {{STAR-VAE}: A Scalable Latent-Variable Transformer for Controllable Molecular Generation},
  author  = {Bum Chul Kwon and Ben Shapira and Moshiko Raboh and Shreyans Sethi and Shruti Murarka and Joseph A. Morrone and Leili Zhang and Wendy Cornell and Pablo Meyer and Jianying Hu and Parthasarathy Suryanarayanan},
  journal = {arXiv preprint arXiv:2511.02769},
  year    = {2026},
  url     = {https://arxiv.org/abs/2511.02769}
}
```

See [CITATION.cff](CITATION.cff) for the same citation in machine-readable form.
No DOI is currently listed.

## Community and License

Use [GitHub Issues](https://github.com/BiomedSciAI/STAR-VAE/issues) according to
[SUPPORT.md](SUPPORT.md). Contributions are described in
[CONTRIBUTING.md](CONTRIBUTING.md), and private vulnerability reporting in
[SECURITY.md](SECURITY.md).

STAR-VAE is provided under the [OpenMDW License Agreement 1.0](LICENSE).
The STAR-VAE checkpoint weights are provided under the same agreement. Datasets
are not redistributed and remain under their providers' terms. External tools
and third-party baseline assets may also have separate terms.

The bundled RDKit SA-score fragment resource
`bmfm_sm_gen/utils/fpscores.pkl.gz` is distributed with the attribution in
`bmfm_sm_gen/utils/sascorer.py`. Its release SHA-256 is
`10dcef9340c873e7b987924461b0af5365eb8dd96be607203debe8ddf80c1e73`.
