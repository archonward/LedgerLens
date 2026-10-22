# Chunk-aware semantic retrieval

The chunk-aware retriever uses the same selected evidence and pinned
`BAAI/bge-small-en-v1.5` model as the page-level semantic baseline. It addresses one specific
problem: a full annual-report page can exceed the model's 512-position input limit, hiding later
content from a single page embedding. The existing page-level index and retriever remain unchanged.

## Chunking configuration

The default configuration is:

- at most 384 model tokens per chunk, leaving room for the model's special tokens;
- 64 tokens of overlap between consecutive chunks;
- a search over the final 64 tokens for a paragraph break, then a line break, before falling back
  to the token boundary; and
- no chunk crossing a physical-page boundary.

LedgerLens loads the tokenizer from the exact pinned model snapshot and disables special tokens
while calculating boundaries. Tokenizer offsets are character spans into the original string, as
documented by the official [Encoding API](https://huggingface.co/docs/tokenizers/en/api/encoding).
Every stored chunk is an exact Python slice of selected page text and records its physical page,
chunk index, half-open character offsets, half-open token offsets, token count, and SHA-256 hash.
Paragraph and line preferences are best-effort: long table rows or paragraphs can still be split.

## Build the cache

Install the semantic extra and create an index on Windows PowerShell:

```powershell
uv sync --locked --group dev --extra semantic --python 3.11
uv run ledgerlens embed-evidence-chunks `
  "artifacts/evidence/sha256-<hash>" `
  --output-root artifacts/chunk-embeddings `
  --model-cache artifacts/models `
  --max-tokens 384 --overlap-tokens 64 --boundary-window-tokens 64
```

Linux uses the same settings:

```bash
uv run ledgerlens embed-evidence-chunks \
  "artifacts/evidence/sha256-<hash>" \
  --output-root artifacts/chunk-embeddings \
  --model-cache artifacts/models \
  --max-tokens 384 --overlap-tokens 64 --boundary-window-tokens 64
```

Use `--offline` to prohibit a model download and require the pinned snapshot in `artifacts/models`.
The generated, Git-ignored directory contains `chunks.jsonl` and `manifest.json`. The manifest binds
the cache to the document/source identity, selected-evidence JSONL and manifest hashes, physical
page order, pinned model/revision, tokenizer-driven chunk configuration, and chunks-file hash.
Loading rejects stale evidence, changed source slices, malformed offsets, altered vectors, model
mismatches, and missing pages. Existing output is protected unless `--overwrite` is explicit.

## Retrieve physical pages

```powershell
uv run ledgerlens chunk-semantic-retrieve `
  "artifacts/evidence/sha256-<hash>" `
  "artifacts/chunk-embeddings/sha256-<hash>/<model-and-chunk-key>" `
  --model-cache artifacts/models `
  --query "What drove Data Center revenue growth?" --top-k 5
```

Chunks are ranked by cosine similarity. `--aggregation max` (the default) assigns each page its
best chunk score. `--aggregation mean_top_n --aggregation-top-n 2` instead averages its two best
available chunks. Page score ties use physical page order; chunk ties use chunk order. Each result
shows the best exact source excerpt, page and chunk scores, character/token offsets, evidence and
chunk hashes, extraction method, review provenance, warnings, and limitations. `--json` emits the
same information in a structured form.

The page score is a ranking signal, not a confidence value, extraction check, or financial-accuracy
measure. Maximum aggregation can over-reward an isolated broadly similar chunk; top-N averaging can
penalize pages that contain one concise relevant passage.

## Evaluation discipline

Use `--retriever chunk-semantic` with `evaluate-retrieval`. Compare candidate chunk sizes, overlaps,
and aggregation rules only on the development split. Record that comparison, freeze one
configuration, and then run the held-out split once. Use the same evidence export, question set,
relevance judgments, and cutoffs as BM25 and page-level semantic retrieval. A higher score on one
small benchmark does not establish general retrieval quality.

## Limitations

- Chunking preserves exact text, not headings-to-table-row relationships or structured columns.
- Overlap duplicates content and increases storage and CPU inference work.
- Best-effort boundaries cannot reliably identify financial-statement sections or keep every row,
  unit, sign, year heading, and footnote together.
- Page-level aggregation can hide useful secondary chunks and cannot satisfy multi-page questions
  by itself.
- Reviewed OCR provenance and page-specific limitations remain attached to the page; chunking does
  not resolve them.
- No hybrid search, reranking, query rewriting, financial calculation, answer generation, or GPU
  execution is included.
