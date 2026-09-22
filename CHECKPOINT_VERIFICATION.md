# Checkpoint Verification Record

Verification date: 2026-09-19

Archive: `star_vae_checkpoints_v1.0.0.zip`

- Size: 6,096,332,747 bytes
- SHA-256: `d47ff7b460bc4eb3bae604407ea6b6067994c31fb7283ecd420e2a0fbdcf3e56`
- Payload: 20 supported model checkpoints and two vocabulary files
- Environment: Linux x86-64, CPython 3.11.13, CPU-only execution

## Archive checks

- ZIP integrity passed with no compressed-data errors.
- Archive membership exactly matched `checkpoint_bundle.json`.
- The unsupported PubChem SMILES ablation was absent.
- Installer archive and extracted-file SHA-256 checks passed.
- Installation into an empty temporary directory produced the canonical
  `checkpoints/` hierarchy without modifying the repository.
- Two independent builder runs were byte-for-byte identical.
- Release checkpoints contain model tensors and sanitized portable
  hyperparameters. The archive builder removes optimizer, callback, logger,
  loop, and local-path metadata.
- A recursive scan of all non-tensor metadata found no personal/internal paths,
  usernames, credentials, or private hostnames in the curated checkpoints.

## Verification commands and outcomes

The following commands were run against the curated archive in an isolated
temporary directory. Exact paths below are illustrative. Substitute any empty
temporary destination.

```bash
sha256sum star_vae_checkpoints_v1.0.0.zip
unzip -tq star_vae_checkpoints_v1.0.0.zip
bash scripts/install_checkpoints.sh \
  --zip star_vae_checkpoints_v1.0.0.zip \
  --dir /tmp/star-vae-checkpoints

PRETRAINED_VAE_CKPT=/tmp/star-vae-checkpoints/vae_pubchem80m/last.ckpt \
VOCAB_FILE=/tmp/star-vae-checkpoints/vae_pubchem80m/vocab.json \
pytest tests/test_generation.py::test_distributed_checkpoint_bundle_loads_expected_models \
  -m requires_ckpt -v
```

Observed exit status was zero for archive integrity, installation, all 20 model
loads, and the bounded generation commands summarized below. The complete
source test suite separately reported 139 passed and 52 deselected during the
workspace sweep. The final clean-source snapshot reported 103 passed and 23
deselected. The full combined CPU checkpoint-generation command exceeded its
one-hour orchestration timeout after eight generation cases passed. B3DB then
passed separately, and the independent 20-model load test passed.

## Model checks

- All 20 included checkpoints loaded with exact model-state coverage.
- PubChem VAE random-prior generation produced an RDKit-valid molecule.
- PubChem VAE input-seeded generation produced an RDKit-valid molecule.
- SA-score CVAE conditional-prior generation produced an RDKit-valid molecule.
- Original 1SYH, 4LDE, and 6Y2F CVAEs produced valid molecules.
- Stratified 1SYH, 4LDE, and 6Y2F CVAEs produced valid molecules.
- B3DB CVAE produced a valid molecule.
- The installed `star-vae-generate cli` path produced a valid molecule.

These checks establish archive, loader, tokenizer, and bounded generation
compatibility. They do not establish paper-scale generation quality or reproduce
benchmark metrics. They also do not recover unknown historical training seeds,
prove that a current config exactly created a historical checkpoint, or establish
that a `last.ckpt` was the best validation checkpoint.

The MOSES checkpoint was loaded with the manifest-declared 328-token PubChem
SELFIES vocabulary. The separately archived 28-token
`vae_moses/smiles_vocab_moses.json` is a historical SMILES-tokenization asset,
not that checkpoint's runtime vocabulary.

## Commands

Only `star_vae_checkpoints_v1.0.0.zip` is downloaded from the external
checkpoint host. The matching `checkpoint_bundle.json` and
`checkpoints.sha256` are already included in the source release and are read
automatically by the installer.

```bash
# Either place the archive at this path in the repository root, or change --zip.
bash scripts/install_checkpoints.sh \
  --zip star_vae_checkpoints_v1.0.0.zip \
  --dir /tmp/star-vae-checkpoints

cp .env.example .env
source .env
pytest -m requires_ckpt
```

Maintainers can reproduce the archive using the command in `CHECKPOINTS.md`.
