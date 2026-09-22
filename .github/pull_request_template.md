## Summary

Describe the problem and the smallest change that resolves it.

## Verification

- [ ] `uv run python -m compileall -q bmfm_sm_gen examples tests`
- [ ] `uv run pytest -m "not slow and not requires_ckpt" -q`
- [ ] New or changed public commands match their current parser help

## Scientific and Release Checklist

- [ ] No credentials, private paths, restricted data, checkpoints, or generated artifacts were added
- [ ] Data provenance, splits, seeds, metrics, and denominators are documented where relevant
- [ ] Random-prior, seed-based, label-conditioned, and CFG modes are named accurately
- [ ] External dependencies and runtime or hardware requirements are disclosed
