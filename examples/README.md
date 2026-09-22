# Generation Examples

These standalone scripts load STAR-VAE Lightning checkpoints without training
configs. Run them from the repository root after `source .env`.

First open <https://ibm.ent.box.com/v/star-vae-checkpoints> in a browser and
download `star_vae_checkpoints_v1.0.0.zip` into the repository root. Then run
`bash scripts/install_checkpoints.sh` to verify and extract the local archive.
The script does not download it. The installer creates `checkpoints/` and all
model/vocabulary subdirectories. See [../CHECKPOINTS.md](../CHECKPOINTS.md).

PyTorch checkpoint loading is unsafe for untrusted files. Only use checkpoints
from a source you trust.

The examples are CI-checked public interfaces. CI also checks a selected list of
experiment-script `--help` paths rather than every script in the source tree. See
[../SUPPORTED_SCRIPTS.md](../SUPPORTED_SCRIPTS.md).

## VAE: Random Prior

Omitting `--input-smiles` samples `z ~ N(0, I)` and decodes it. This is
random-prior generation, not seed-based generation.

```bash
python examples/generate_unconditional.py \
  --checkpoint "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --num-samples 32 \
  --temperature 1.0 \
  --seed 42 \
  --output "$OUTPUT_DIR/random.csv"
```

## VAE: Seed Based

With `--input-smiles`, the script encodes the molecule to its latent mean and
adds isotropic Gaussian noise scaled by `--noise-scale`. Similarity is an
intended tendency, not a guarantee.

```bash
python examples/generate_unconditional.py \
  --checkpoint "$PRETRAINED_VAE_CKPT" \
  --vocab "$VOCAB_FILE" \
  --input-smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --noise-scale 0.5 \
  --num-samples 32 \
  --output "$OUTPUT_DIR/aspirin_analogues.csv"
```

VAE options are exactly those shown by:

```bash
python examples/generate_unconditional.py --help
```

The checkpoint default is `$PRETRAINED_VAE_CKPT`. The vocabulary default is
`$VOCAB_FILE`. Defaults for `--noise-scale`, `--num-samples`, `--temperature`,
and `--seed` are `0.3`, `32`, `1.0`, and `42`. `--top-k` is off by default and
`--device` auto-selects CUDA when available.

## CVAE: Label-Conditioned Prior

Without `--input-smiles`, the CVAE script uses the selected latent prior and
always conditions decoding on `--target-label`:

- `--prior-mode auto`: use `p(z|y)` if enabled by the checkpoint, else `N(0,I)`.
- `--prior-mode conditional`: require and use `p(z|y)`.
- `--prior-mode standard`: force `N(0,I)` while retaining decoder label conditioning.

```bash
python examples/generate_conditional.py \
  --checkpoint "$CKPT_SA_SCORE" \
  --vocab "$VOCAB_FILE" \
  --target-label 0.25 \
  --prior-mode conditional \
  --num-samples 32 \
  --output "$OUTPUT_DIR/sa_score_025.csv"
```

`generate_conditional.py` accepts labels only in `[0,1]`. The label's meaning
comes from the checkpoint's training data. It is not universally a raw property
value. For `cvae_sa_score`, larger normalized values represent larger SA score.
For stratified Tartarus checkpoints produced by the current ranking function,
labels are ascending percentile ranks of docking score: `0` is the strongest
end (most negative score) and `1` is the weakest end (less negative, approaching
zero). Do not transfer a label convention between checkpoints.

## CVAE: Seed and Label

This mode perturbs the encoded seed and conditions the decoder on a label. It
does not sample the learned conditional prior.

```bash
python examples/generate_conditional.py \
  --checkpoint "$CKPT_SA_SCORE" \
  --vocab "$VOCAB_FILE" \
  --input-smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --target-label 0.25 \
  --noise-scale 0.4 \
  --num-samples 32 \
  --output "$OUTPUT_DIR/aspirin_sa_analogues.csv"
```

CVAE options are exactly those shown by:

```bash
python examples/generate_conditional.py --help
```

The checkpoint default is `$CKPT_SA_SCORE`. `--target-label` is required.

## CFG Is Separate

Neither standalone example applies classifier-free guidance. Paper-specific CFG
implementations are specialized `bmfm_sm_gen/scripts/exp17c_*_generate.py`
programs. Do not treat them as interchangeable general-purpose CLIs.
Inspect each script's `--help`, checkpoint constants, label scale, and output
layout before running it.

The standalone example CSVs preserve one row per requested attempt. A failed
decode has an empty `smiles` cell. Downstream scoring or docking can reduce the
usable denominator, so report attempted, valid decoded, scored, and successfully
docked counts separately. Some historical experiment-specific scripts write
valid decoded molecules only. Consult their run manifests and logs rather than
inferring attempted counts from CSV row counts.
