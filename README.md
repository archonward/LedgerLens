# LedgerLens

LedgerLens is an evidence-based AI assistant for answering questions about company annual
reports. Its intended first version will work with one company and two consecutive reports,
retrieve page-level evidence, perform deterministic calculations, and produce answers whose
claims can be traced back to source documents.

> [!IMPORTANT]
> This repository implements CPU-only local PDF ingestion, reviewed evidence selection, and
> page-level BM25, semantic, chunk-aware semantic, and hybrid RRF retrieval, a validated financial
> fact representation, deterministic calculations, a narrow evidence-grounded answer-generation
> API, plus offline retrieval evaluation. It is not yet a complete financial-analysis assistant:
> automatic financial extraction, input selection, general question answering, and later evaluation
> stages remain planned. No reports, datasets, or models are included.

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

See the [ingestion guide](docs/ingestion.md), [reviewed evidence guide](docs/evidence.md),
[BM25 retrieval guide](docs/retrieval.md), [semantic retrieval guide](docs/semantic-retrieval.md),
[chunk-aware semantic retrieval guide](docs/chunk-semantic-retrieval.md),
[hybrid retrieval guide](docs/hybrid-retrieval.md),
[financial fact representation](docs/financial-facts.md),
[deterministic calculations](docs/calculations.md),
[NVIDIA FY2026 reviewed evidence demonstration](docs/real-financial-evidence.md),
[evidence-grounded answer generation](docs/answer-generation.md),
[architecture](docs/architecture.md), and [evaluation](docs/evaluation.md) documents.

## Setup

Python 3.11 is the initial tested baseline. Dependencies are managed and locked with
[uv](https://docs.astral.sh/uv/). Canonical PDF text extraction uses `pdfplumber`. Targeted
single-page OCR is optional; structured table extraction is not implemented.

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

Ingestion flags blank text, unresolved `(cid:<digits>)` tokens, Unicode replacement characters,
and unexpected Unicode controls (ordinary tabs and line endings are allowed). Pages with detected
issues require review and are marked unsuitable for automatic indexing. These limited checks do
not establish completeness, correct reading order, financial accuracy, or verification.

An optional pypdf-based command can extract a single page as a separate recovery candidate:

```bash
uv sync --locked --group dev --extra recovery
uv run ledgerlens recover-page "data/raw/report.pdf" --page 128
```

Recovery candidates never replace canonical ingestion text automatically and always require
visual review against the source PDF.

Targeted OCR uses `pypdfium2` to render only the requested physical page and invokes a separately
installed Tesseract executable. It preserves the image, exact OCR text, word-level TSV, settings,
versions, and quality findings in an engine-specific directory:

```bash
uv sync --locked --group dev --extra ocr
uv run ledgerlens recover-page "data/raw/report.pdf" --page 128 --engine tesseract
```

Tesseract is system software and is not installed or configured by LedgerLens. Use
`--tesseract-path` when it is not on `PATH`. OCR confidence is not proof of correctness.

An explicit, hash-bound review manifest can select a corrected recovery without modifying the
canonical extraction or original OCR artifacts:

```bash
uv run ledgerlens prepare-evidence "data/processed/sha256-<hash>" \
  --review-manifest "artifacts/reviewed-recovery/sha256-<hash>/review-manifest.json"
```

Selected evidence is intended only for a page-level retrieval prototype. Review permission does
not validate structured financial calculations or make automated screening equivalent to
verification. See [docs/evidence.md](docs/evidence.md) for the artifact distinctions and manifest
format.

The `retrieve` command validates and loads that selected evidence, builds an in-memory BM25 index,
and returns positive-score physical pages with exact source excerpts, hashes, provenance, warnings,
and limitations:

```bash
uv run ledgerlens retrieve "artifacts/evidence/sha256-<hash>" \
  --query "What drove Data Center revenue growth?" --top-k 5
```

Add `--json` for structured output. Retrieval never reads the PDF or canonical ingestion directly,
and it does not discover recovery candidates. BM25 scores rank lexical matches; they are not
confidence, correctness, or financial-accuracy measures. See [docs/retrieval.md](docs/retrieval.md).

Verified and provisional question sets can be evaluated separately from retrieval outputs:

```bash
uv run ledgerlens evaluate-retrieval "artifacts/evidence/sha256-<hash>" \
  "configs/evaluation/questions.json" --split development --k 1 3 5 10 \
  --output "artifacts/evaluation/development-bm25.json"
```

Only manually verified questions contribute to Hit@K, Recall@K, and MRR. See
[docs/evaluation.md](docs/evaluation.md) for the schema, split discipline, and metric definitions.

Semantic retrieval is an optional CPU baseline using a pinned, quantized ONNX embedding model.
Model download and embedding generation are explicit and write only beneath ignored `artifacts/`
directories:

```bash
uv sync --locked --group dev --extra semantic
uv run ledgerlens embed-evidence "artifacts/evidence/sha256-<hash>"
uv run ledgerlens semantic-retrieve "artifacts/evidence/sha256-<hash>" \
  "artifacts/embeddings/sha256-<hash>/baai-bge-small-en-v1-5-c32e6154d1bb" \
  --query "What caused revenue to increase?" --top-k 5
```

The embedding cache is hash-bound to selected evidence. Semantic scores are ranking signals, not
confidence or correctness measures. Long pages are truncated at the model input limit during
embedding. See [docs/semantic-retrieval.md](docs/semantic-retrieval.md).

Chunk-aware semantic retrieval avoids that page-level truncation by making overlapping, exact-text
slices within each physical page and aggregating chunk similarities back to page rankings:

```bash
uv run ledgerlens embed-evidence-chunks "artifacts/evidence/sha256-<hash>" --offline
uv run ledgerlens chunk-semantic-retrieve "artifacts/evidence/sha256-<hash>" \
  "artifacts/chunk-embeddings/sha256-<hash>/<model-and-chunk-key>" \
  --query "What caused revenue to increase?" --top-k 5
```

Chunks retain exact character and token offsets, but do not restore table structure or validate
financial meaning. See [docs/chunk-semantic-retrieval.md](docs/chunk-semantic-retrieval.md).

Hybrid retrieval reuses an existing page-embedding cache and combines unchanged BM25 and
page-semantic ranks with reciprocal rank fusion:

```bash
uv run ledgerlens hybrid-retrieve "artifacts/evidence/sha256-<hash>" \
  "artifacts/embeddings/sha256-<hash>/<model-key>" \
  --query "What caused revenue to increase?" --top-k 5
```

The fused result exposes both component ranks and contributions. It does not mix raw BM25 and
cosine scores or establish evidence correctness. See [docs/hybrid-retrieval.md](docs/hybrid-retrieval.md).

The Python calculation API accepts only verified financial facts paired with the evidence corpus
that validates them. It currently supports year-over-year growth, profit margins, and compatible
two-input ratios using a fixed Decimal context. Results retain the formula, complete input fact
snapshots, units, periods, and source provenance. Calculation input selection and CLI integration
are not implemented. See [docs/calculations.md](docs/calculations.md).

An offline NVIDIA FY2026 demonstration validates four locally reviewed report facts against the
unchanged selected-evidence export, then reproduces revenue growth and net profit margins. The
report, fact records, review record, and results remain Git-ignored. The review was performed by
Codex and is explicitly not described as independent human verification. See
[docs/real-financial-evidence.md](docs/real-financial-evidence.md).

The Python answer-generation API accepts one explicitly selected revenue-growth or net-profit-margin
calculation and the exact verified facts behind it. It revalidates the evidence, reproduces the
calculation, renders every number and physical-page citation deterministically, and limits a
configurable local open-model backend to a checked qualitative sentence. Offline tests use a mock
backend; LedgerLens does not download or require a model or GPU. See
[docs/answer-generation.md](docs/answer-generation.md).

An optional standard-library Ollama adapter supports only an already-installed local
`qwen3:4b-instruct` tag and never downloads or substitutes a model. Its real-model NVIDIA smoke
test is disabled unless `LEDGERLENS_RUN_OLLAMA_SMOKE=1` is explicitly set. Model identity and
digest, inference settings, response latency, and strict grounding outcome are retained in answer
provenance.

Review [docs/ingestion.md](docs/ingestion.md) before treating extraction output as usable. Text
extraction does not validate financial figures, infer currencies or units, recover scanned text,
or preserve table structure reliably.

## Local development and cluster execution

Local Windows or Linux development is sufficient for current ingestion and retrieval, and future
calculation and application code. Everything currently implemented runs on CPU and without
network access. Future model inference, evaluation, and possible fine-tuning may run
as scheduled Slurm jobs on NUS SoC GPU infrastructure. Heavy work must not run on cluster login
nodes. Cluster parameters remain intentionally unspecified until they are confirmed; see the
[Slurm guidance](scripts/slurm/README.md).
