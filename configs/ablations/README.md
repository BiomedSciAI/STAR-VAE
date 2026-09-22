# Public SA-score ablation subset

These configs define an executable, one-factor public reproduction subset. They
are not asserted to be the exact historical Exp19 run configs. The historical
YAML files are under `paper_results/` and contain local paths and, in several
cases, reuse one CSV for training and validation.

The subset compares the integrated reference model with a model that disables
only the learned conditional prior while retaining posterior/decoder FiLM
conditioning. Set `PRETRAINED_VAE_CKPT`, `VOCAB_FILE`, `SA_SCORE_TRAIN_CSV`, and
`SA_SCORE_VAL_CSV`. The train and validation CSVs must be distinct files with
the same `selfies` and `y_normalized_minmax` schema.

`prep_sa_score_minmax.py` writes these files using a deterministic row-level
split (20% validation and seed 42 by default). It does not canonicalize
molecules or perform a scaffold/structural overlap audit. Run and report such an
audit separately before making a leakage-control claim.

```bash
python bmfm_sm_gen/scripts/exp19_train_ablation.py \
  --config configs/ablations/sa_reference_public.yaml --seed 42

python bmfm_sm_gen/scripts/exp19_train_ablation.py \
  --config configs/ablations/sa_no_conditional_prior_public.yaml --seed 42
```

These are GPU training jobs, not smoke tests. Add `--smoke_test` only for an
interface check. It does not produce scientifically interpretable results.
