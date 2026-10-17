# Repository guidance

- Make small, incremental changes and keep the CPU-only foundation usable on Windows and Linux.
- Never fabricate financial figures, evidence, citations, or evaluation results.
- Preserve provenance: document identity, page, reporting period, currency, and units must remain
  attached to evidence and derived values.
- Tests must not download external resources, execute models, probe credentials, or require a GPU.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest` for relevant
  changes. Verify both `ledgerlens` and `python -m ledgerlens` when changing the CLI.
- Submit heavy cluster work through Slurm; never run it on a login node.
- Do not commit secrets, environment files, financial reports, datasets, model weights,
  checkpoints, experiment logs, or generated artifacts.
