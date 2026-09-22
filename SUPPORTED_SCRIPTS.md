# Supported Command Boundary

The installed commands `star-vae-generate`, `star-vae-generate-bulk`, and
`star-vae-train` are the primary packaged interfaces. The standalone generation
examples in `examples/` are the primary source-checkout examples.

CI runs `--help` for every installed command and for a selected set of documented
data-preparation and experiment scripts listed explicitly in
`.github/workflows/ci.yml`. It also runs cheap parser-validation paths and bounded
CPU tests. This selection is not a claim that every file under
`bmfm_sm_gen/scripts/` is tested, portable, or supported.

Scripts named `exp*`, `revision_*`, or `run_*` may encode historical paths,
checkpoint assumptions, external environments, expensive workloads, or analysis
for a particular paper result. A command documented in [EXPERIMENTS.md](EXPERIMENTS.md)
is supported only to the extent stated there. Before a full run, inspect
`--help`, supply explicit paths, run a smoke test when available, and preserve
the config, checkpoint digest, dataset version, seed, logs, and attempt counts.

Historical result directories can predate current script corrections. Current
commands are executable reproduction protocols. They are not evidence that an
old result used the same source revision unless provenance explicitly links the
result to that revision.

The public repository includes the supported experiment scripts listed below.
Other local scripts are historical or internal working material and are excluded
by `.gitignore`.

The supported source-script set is:

- `exp02_run_moses_full_metrics.py`
- `exp16_prep_stratified_tartarus.py`
- `exp16_generate_tartarus.py`
- `exp16_dock_tartarus.py`
- `exp17c_sa_score_generate.py`
- `exp17c_generate.py`
- `exp18_drugdiff_download_checkpoints.py`
- `exp19_train_ablation.py`
- `exp19_generate_ablation.py`
- `exp19_evaluate_ablation.py`
- `revision_b2_sa_denominators.py`
- `revision_c1_docking_attempt_level.py`
- `revision_c2_ablation_uncertainty.py`
- `revision_c3_rebuild_rediscovery_table.py`
- `revision_c4_recompute_docking_summary.py`

The `revision_*` entries above are portable, bounded CPU reducers used to make
metric denominators and corrected tables auditable. Internal scheduler wrappers
and large overlap/training jobs remain excluded.
