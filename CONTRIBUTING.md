# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Before Opening a Change

- Search existing issues and open an issue for substantial behavioral or
  scientific changes.
- Do not submit private datasets, model checkpoints, credentials, internal
  paths, patient information, or material you cannot redistribute.
- Keep generated outputs and large artifacts out of pull requests.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Development Setup

```bash
uv sync --extra dev
source .venv/bin/activate
```

Run bounded checks before submitting:

```bash
uv run python -m compileall -q bmfm_sm_gen examples tests
uv run pytest -q
```

These bounded checks do not imply that every program under
`bmfm_sm_gen/scripts/` is supported. If adding a documented source command,
classify it in [SUPPORTED_SCRIPTS.md](SUPPORTED_SCRIPTS.md) and add a cheap CI
help or parser check when feasible.

Tests requiring datasets, checkpoints, GPUs, docking, or external baseline
repositories must be clearly marked and must not download resources implicitly.

## Pull Requests

- Describe the problem, implementation, and verification performed.
- Add or update tests for behavioral changes.
- Keep CLI documentation synchronized with parser option names.
- For scientific changes, state data provenance, split construction, random
  seeds, metric definitions, denominators, hardware, and limitations.
- Distinguish random-prior, seed-based, and label-conditioned generation.
- Do not call same-procedure rescoring an independent oracle.
- For checkpoints or other publisher-supplied artifacts, document config, data
  family, training seed (or `UNKNOWN`), paper role, and historical limitations.
  Do not shift publisher rights-clearance obligations to downstream users.

Contributions are accepted under the repository's
[OpenMDW License Agreement 1.0](LICENSE).
