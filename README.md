# LedgerLens

LedgerLens is an evidence-based AI assistant for answering questions about company annual
reports. Its intended first version will work with one company and two consecutive reports,
retrieve page-level evidence, perform deterministic calculations, and produce answers whose
claims can be traced back to source documents.

> [!IMPORTANT]
> This repository implements a CPU-only baseline for local PDF text ingestion and inspection. It
> is not yet a working financial-analysis assistant: retrieval, financial interpretation,
> calculations, answer generation, and evaluation remain planned. No reports, datasets, or models
> are included.

## Repository layout

```text
src/ledgerlens/       Installable Python package and pipeline component namespaces
tests/                CLI smoke and behavior tests
configs/              Future version-controlled experiment and pipeline configuration
scripts/slurm/         Guidance for future NUS SoC Slurm jobs
data/raw/             Local source reports (ignored by Git)
data/processed/       Local derived document data (ignored by Git)
artifacts/             Local outputs, checkpoints, and experiment results (ignored by Git)
docs/                  Architecture and evaluation plans
.github/workflows/     Continuous integration
```

See the [ingestion guide](docs/ingestion.md), [architecture](docs/architecture.md), and
[evaluation](docs/evaluation.md) documents.

## Setup

Python 3.11 is the initial tested baseline. Dependencies are managed and locked with
[uv](https://docs.astral.sh/uv/). PDF text extraction uses `pdfplumber`; it does not perform OCR
or structured table extraction.

### Windows PowerShell

```powershell
uv python install 3.11
uv sync --locked --group dev --python 3.11
uv run ledgerlens --help
uv run ledgerlens doctor
uv run ledgerlens ingest "data/raw/report.pdf" --company "Example Company" --report-year 2024
```

### Linux

```bash
uv python install 3.11
uv sync --locked --group dev --python 3.11
uv run ledgerlens --help
uv run ledgerlens doctor
uv run ledgerlens ingest data/raw/report.pdf --company "Example Company" --report-year 2024
```

Run the module entry point equivalently with:

```bash
uv run python -m ledgerlens --help
uv run python -m ledgerlens --version
```

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ruff format .
uv run pytest
```

The first two commands check lint and formatting, the third applies formatting, and the fourth
runs the test suite.

## Implemented scope

The `ingest` command reads a user-supplied local PDF, retains every physical page, and writes
page-linked text plus metadata and an inspection report beneath
`data/processed/<document-id>/`. The stable document ID is derived from the PDF content's SHA-256
hash. Existing output is protected unless `--overwrite` is supplied. An optional `--source-url`
records provenance metadata but is never fetched.

Review [docs/ingestion.md](docs/ingestion.md) before treating extraction output as usable. Text
extraction does not validate financial figures, infer currencies or units, recover scanned text,
or preserve table structure reliably.

## Local development and cluster execution

Local Windows or Linux development is sufficient for the current PDF ingestion and future
retrieval, calculation, and application code. Everything currently implemented runs on CPU and
without network access. Future model inference, evaluation, and possible fine-tuning may run
as scheduled Slurm jobs on NUS SoC GPU infrastructure. Heavy work must not run on cluster login
nodes. Cluster parameters remain intentionally unspecified until they are confirmed; see the
[Slurm guidance](scripts/slurm/README.md).
