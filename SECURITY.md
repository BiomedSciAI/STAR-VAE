# Security Policy

## Reporting a Vulnerability

Do not disclose a suspected vulnerability in a public issue. Use GitHub's
private vulnerability reporting for this repository. If that feature is not
available, contact the maintainers listed in `pyproject.toml` and include a
minimal reproduction, affected version, and impact. Do not include sensitive
data in the initial report.

No guaranteed response or remediation timeline is currently published.

## Checkpoint and Data Trust

PyTorch checkpoints may contain pickle data, and loading an untrusted `.ckpt`
can execute code. Only load checkpoints from trusted sources. The public
checkpoint archive tested for this release has its SHA-256 recorded in the
installer and README. A matching digest checks integrity against that tested
file, but users must still trust the publication channel and maintainers. The
archive's stated OpenMDW license and provenance metadata do not make pickle
deserialization safe or replace publisher-side institutional rights confirmation.

The primary generation interfaces also print this trust warning immediately
before loading a checkpoint.

Treat third-party datasets, molecule files, archives, and external tools as
untrusted input. Review their provenance and terms, scan archives before use,
and run expensive or external tooling with least privilege. Never place secrets
in `.env`, configs, issue reports, logs, or generated artifacts intended for
publication.

STAR-VAE is research software and is not a security, safety, clinical, or
toxicity assessment system.
