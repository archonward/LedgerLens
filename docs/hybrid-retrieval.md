# Hybrid evidence retrieval

LedgerLens implements an explainable hybrid baseline that combines the existing page-level BM25
and page-level semantic rankings. It reads the same validated selected-evidence corpus and reuses an
existing hash-validated embedding cache. It does not regenerate page embeddings or use chunk-aware
retrieval.

## Reciprocal rank fusion

For a physical page `d`, weighted reciprocal rank fusion computes:

```text
score(d) = bm25_weight / (rrf_k + bm25_rank(d))
         + semantic_weight / (rrf_k + semantic_rank(d))
```

The defaults are `rrf_k = 60` and equal positive weights of `1`. BM25 contributes only when a page
has a positive lexical match; an absent BM25 rank contributes zero. Page semantic ranks every
selected page. The component retrievers run to the full selected-corpus depth before fusion, so the
requested display cutoff does not change fused ranks. Ties are resolved by ascending physical PDF
page number.

RRF uses ranks because BM25 and cosine similarities have different score scales. Raw component
scores are retained for inspection but are never added, normalized, or treated as probabilities.
The fused score is not confidence, evidence verification, or financial accuracy.

## Querying

Windows PowerShell:

```powershell
uv run ledgerlens hybrid-retrieve `
  "artifacts/evidence/sha256-<hash>" `
  "artifacts/embeddings/sha256-<hash>/<model-key>" `
  --model-cache artifacts/models `
  --rrf-k 60 --bm25-weight 1 --semantic-weight 1 `
  --query "What drove Data Center revenue growth?" --top-k 5
```

Linux:

```bash
uv run ledgerlens hybrid-retrieve \
  "artifacts/evidence/sha256-<hash>" \
  "artifacts/embeddings/sha256-<hash>/<model-key>" \
  --model-cache artifacts/models \
  --rrf-k 60 --bm25-weight 1 --semantic-weight 1 \
  --query "What drove Data Center revenue growth?" --top-k 5
```

The command is offline: the selected evidence, page-embedding index, and pinned model snapshot must
already exist. It validates the embedding cache against the evidence before searching. `--json`
includes the fusion configuration, embedding-cache hashes, component rank/score/contribution,
excerpt source, selected-text and source hashes, extraction method, review provenance, warnings,
and limitations.

When BM25 ranks a page, its matched source passage is used as the excerpt. Otherwise, the semantic
retriever's exact page-start excerpt is used. Neither is generated or paraphrased.

The separate [retrieval-to-reviewed-fact association layer](retrieval-fact-association.md) can use
these provenance-bound page results to constrain the financial-answer orchestrator. It matches
complete evidence identities rather than treating rank, page number, or excerpt similarity as fact
verification.

## Evaluation

```powershell
uv run ledgerlens evaluate-retrieval `
  "artifacts/evidence/sha256-<hash>" `
  "artifacts/evaluation/questions.json" `
  --retriever hybrid `
  --embedding-index "artifacts/embeddings/sha256-<hash>/<model-key>" `
  --rrf-k 60 --bm25-weight 1 --semantic-weight 1 `
  --model-cache artifacts/models `
  --split development --k 1 3 5 10 `
  --output "artifacts/evaluation/development-hybrid.json"
```

Use development questions to select the fixed constant and weights. LedgerLens records all fusion,
component, model, cache, and evidence identities in the report. Retrieval results remain separate
from independently reviewed relevance judgments. An already examined held-out set cannot be
described as a fresh independent evaluation.

## Limitations

- RRF discards score magnitudes and depends only on rank positions and fixed weights.
- A large common rank constant compresses differences near the top; weights can cause one component
  to dominate, especially on a small corpus.
- Semantic truncation and BM25 vocabulary mismatch remain unchanged.
- Fusion cannot repair extraction errors, flattened tables, ambiguous columns, or incomplete
  relevance judgments.
- Page-level ranking does not prove that the displayed excerpt alone supports the question.
- This retriever itself performs no chunk fusion, learned weighting, query rewriting, reranking,
  calculation, answer generation, or GPU execution.
