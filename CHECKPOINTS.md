# Checkpoint Manifest

The curated checkpoint archive covered by `checkpoints.sha256` accompanies
STAR-VAE version 1.0.0 as Model Materials under the repository's OpenMDW License
Agreement 1.0. The archive contains model weights produced for the STAR-VAE
study. Datasets are not included and retain their original providers' terms.

The PubChem VAE supports random-prior and seed-based generation. The SA-score,
B3DB, and Tartarus checkpoints are label-conditioned CVAEs. Stratified Tartarus
labels are percentile ranks where 0 is the strongest/more-negative end and 1 is
the weakest/less-negative end. The `ablations/` files are experiment variants.

All included models load with the bundled 328-token PubChem SELFIES vocabulary,
including `vae_moses`. The historically named
`vae_moses/smiles_vocab_moses.json` is a separate 28-token SMILES vocabulary
reproduction asset and is not the vocabulary for the bundled MOSES checkpoint.
The PubChem SMILES-tokenization ablation is intentionally omitted because its
exact 231-token vocabulary is unavailable. The MOSES SMILES vocabulary is not
interchangeable with that missing PubChem vocabulary.

`checkpoint_bundle.json` is the machine-readable source of truth for archive
membership, checkpoint paths, vocabulary paths, tokenizers, and supported
generation protocols. It also records per-model provenance. A `null` config or
`UNKNOWN` training seed means that the exact historical artifact was not
established. We do not silently present a current related config as the exact
training config. `scripts/build_checkpoint_archive.py` rebuilds the ZIP from an
extracted source directory without manually copying files.

## Canonical layout

After installation, the repository contains:

```text
checkpoints/
  vae_pubchem80m/{last.ckpt,vocab.json}
  vae_moses/{last.ckpt,smiles_vocab_moses.json}
  cvae_sa_score/last.ckpt
  cvae_b3db/last.ckpt
  cvae_{1syh,4lde,6y2f}/last.ckpt
  cvae_{1syh,4lde,6y2f}_stratified/last.ckpt
  ablations/<verified-variant>/last.ckpt
```

The installer creates this layout. Users should not unpack individual files or
move vocabularies next to arbitrary checkpoints.

## Build and install

### Reviewer and user download

The only file users need to download from the external checkpoint host is:

```text
star_vae_checkpoints_v1.0.0.zip
```

Open <https://ibm.ent.box.com/v/star-vae-checkpoints> in a web browser and
download `star_vae_checkpoints_v1.0.0.zip`. Place it in the repository root.
The Box page requires browser download. The installation script does not attempt
to download it.

`checkpoint_bundle.json` and `checkpoints.sha256` are distributed with the
matching source release and should not be downloaded separately. The installer
uses the repository copies to authenticate the ZIP, require its exact expected
contents, verify every extracted file, and create the canonical directory
layout.

From the repository root, verify and extract the downloaded ZIP with:

```bash
bash scripts/install_checkpoints.sh
```

For a ZIP saved elsewhere, pass `--zip /path/to/archive.zip`. A source checkout
and checkpoint archive from different release versions may not verify together.
Use all artifacts from the same tagged release.

### Maintainer build

Maintainers can rebuild the curated archive from an extracted source bundle:

```bash
python scripts/build_checkpoint_archive.py \
  --source /path/to/extracted/checkpoints \
  --output star_vae_checkpoints_v1.0.0.zip \
  --checksums checkpoints.sha256
```

Maintainers can test the resulting archive from the repository root:

```bash
bash scripts/install_checkpoints.sh
cp .env.example .env
source .env
pytest -m requires_ckpt
```

`--dir` controls the extracted checkpoint root. The script retains the archive
at its original location.

The archive and extracted files are pickle-capable PyTorch data. Load only
files obtained from a trusted source and verify them with `checkpoints.sha256`.
The smoke tests establish software compatibility and production of at least one
RDKit-valid molecule. They do not reproduce paper-scale quality metrics.

The builder sanitizes each Lightning checkpoint to model tensors, portable
architecture hyperparameters, basic epoch/global-step provenance, and a
vocabulary SHA-256. It removes callbacks, optimizer/scheduler state, logger
state, training loops, and local filesystem paths before creating the public
archive.

Several weights are historical `last.ckpt` artifacts. Compatibility verification
does not establish their original training seed, exact data snapshot, best-model
selection, or bitwise reproducibility. Consult the provenance fields and caveats
in `checkpoint_bundle.json` before making an exact-result claim.
