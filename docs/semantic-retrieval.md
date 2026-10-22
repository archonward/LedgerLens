# Page-level semantic retrieval

LedgerLens implements an optional CPU semantic-retrieval baseline over the same selected physical
pages used by BM25. It never reads the source PDF, canonical extraction, or unapproved recovery
candidates. Results therefore retain document identity, physical page, selected-text and source
hashes, extraction method, review provenance, warnings, and limitations.

## Model choice

The baseline uses `BAAI/bge-small-en-v1.5` through FastEmbed's quantized ONNX representation. It was
selected because it is English retrieval-oriented, produces 384-dimensional vectors, is small
enough for local CPU use, and avoids a PyTorch dependency. FastEmbed documents its CPU-oriented
ONNX implementation and separate `passage_embed` and `query_embed` APIs in its
[official documentation](https://qdrant.github.io/fastembed/Getting%20Started/).

For reproducibility, LedgerLens pins the underlying repository
`Qdrant/bge-small-en-v1.5-onnx-Q` at revision
`c32e6154d1bb7a0e47c5e745fd895e7700f44385`. The model configuration has 512 positions, so text
beyond the model input limit is truncated. The model repository, revision, FastEmbed version,
dimension, encoding methods, normalization, similarity, batch size, and evidence hashes are stored
in the index manifest.

## Setup and cache creation

Install the optional dependency:

```powershell
uv sync --locked --group dev --extra semantic --python 3.11
```

Build the page embeddings on Windows PowerShell:

```powershell
uv run ledgerlens embed-evidence `
  "artifacts/evidence/sha256-<hash>" `
  --output-root artifacts/embeddings `
  --model-cache artifacts/models
```

Linux uses the same arguments:

```bash
uv run ledgerlens embed-evidence \
  "artifacts/evidence/sha256-<hash>" \
  --output-root artifacts/embeddings \
  --model-cache artifacts/models
```

This command may download the pinned model. It does not probe credentials: the public download is
explicitly unauthenticated. Use `--offline` to require an existing local model snapshot. Model files
and generated embeddings remain under Git-ignored `artifacts/` paths. An existing index is protected
unless `--overwrite` is supplied.

The resulting directory contains:

- `manifest.json`: evidence hashes, physical page order, model identity and revision, configuration,
  cache hash, and limitations;
- `vectors.jsonl`: one normalized vector per selected physical page, with its page number and exact
  selected-text hash.

Loading rejects changed evidence, changed page order, stale text hashes, tampered vector data,
dimension mismatches, non-finite vectors, or a model configuration that differs from the cache.
Hash agreement establishes input identity, not extraction or financial correctness.

## Querying

```powershell
uv run ledgerlens semantic-retrieve `
  "artifacts/evidence/sha256-<hash>" `
  "artifacts/embeddings/sha256-<hash>/<model-key>" `
  --model-cache artifacts/models `
  --query "What caused revenue to increase?" `
  --top-k 5
```

The query command is offline and fails clearly if the pinned model is absent. Add `--json` for
structured output. Query vectors and cached page vectors are L2-normalized; ranking uses cosine
similarity with physical page number as the deterministic tie-breaker. The excerpt is copied from
the selected evidence text and is never generated or paraphrased.

## Evaluation

Use the existing judgments unchanged:

```powershell
uv run ledgerlens evaluate-retrieval `
  "artifacts/evidence/sha256-<hash>" `
  "artifacts/evaluation/questions.json" `
  --retriever semantic `
  --embedding-index "artifacts/embeddings/sha256-<hash>/<model-key>" `
  --model-cache artifacts/models `
  --split development --k 1 3 5 10 `
  --output "artifacts/evaluation/development-semantic.json"
```

Select models and configuration using development questions only. Freeze the configuration before
running held-out evaluation. Reports use the same Hit@K, Recall@K, and MRR definitions as BM25 and
record the model and cache hashes needed for comparison.

## Limitations

- One embedding represents one physical page, but only the first 512 model tokens are available to
  this baseline. Relevant text late on a long page may be invisible to the embedding.
- No passage splitting, table reconstruction, hybrid ranking, reranking, query rewriting, synonym
  list, financial interpretation, or numerical calculation is performed.
- Dense similarity may improve paraphrase matching while confusing pages with broadly similar
  accounting language.
- A cosine score is only a ranking signal. It is not confidence, evidence verification, extraction
  accuracy, or support for a financial conclusion.
- Reviewed OCR text and page limitations remain visible, but semantic retrieval does not remove or
  resolve those limitations.

This page-level baseline remains unchanged for reproducible comparison. The separately implemented
[chunk-aware variant](chunk-semantic-retrieval.md) embeds bounded slices and aggregates them back to
physical pages. The [hybrid retriever](hybrid-retrieval.md) reuses this page-level cache and fuses
its ranking with BM25.
