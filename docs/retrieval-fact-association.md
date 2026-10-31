# Retrieval-to-reviewed-financial-fact association

LedgerLens can conservatively constrain the existing financial-answer orchestrator to reviewed
facts whose selected evidence was returned by the existing page-level hybrid retriever. This is an
optional boundary: it does not extract facts from retrieved text, interpret tables, or change the
existing explicit-facts answer path.

## Exact association identity

A reviewed fact is eligible only when it still validates against the supplied evidence corpus and
its evidence identity exactly matches a retrieved selected page. The identity includes:

- document ID and source-document SHA-256;
- evidence JSONL and evidence-manifest SHA-256 hashes;
- physical PDF page number;
- selected page-text SHA-256; and
- selected evidence method.

The page number is therefore necessary but never sufficient. Fact validation additionally checks
company, report year, the exact supporting-text substring and its hash, warnings, limitations, and
review provenance. Retrieval excerpts and keyword similarity are not used to manufacture or
validate a fact.

All matching facts remain separate records. For example, revenue and net income from one statement
page are both associated by their fact IDs; the existing orchestrator still selects exact company,
metric, and reporting-period roles. Retrieval relevance never becomes verification.

## Structured workflow

`RetrievalFinancialAnswerOrchestrator` performs four bounded steps:

1. run the supplied `HybridRetriever` with the financial question and explicit cutoff;
2. validate every returned page and retrieval hash against the evidence corpus;
3. associate only exact-identity reviewed facts and record a decision for every supplied fact; and
4. pass only associated facts to `FinancialAnswerOrchestrator`.

There is no fallback to the full fact collection. If a prior-period revenue fact falls outside the
retrieval cutoff, year-over-year growth abstains even if that fact was present in the input file.

The structured result records retrieved rank, fused score, fusion-component details, evidence
identity, associated fact IDs, per-fact verification status and outcome, downstream selection
audit, calculation, answer, and citations. Retrieval-specific abstentions include:

- `retrieval_failed`;
- `no_retrieved_evidence`;
- `invalid_or_stale_retrieval`;
- `no_associated_reviewed_facts`;
- `invalid_or_stale_association`;
- `insufficient_associated_facts`;
- `ambiguous_associated_facts`; and
- `unverified_associated_fact`.

Unsupported questions, calculation incompatibility, model failures, and grounding rejections
remain downstream abstentions under the existing policies.

## CLI

Supply an existing hash-validated page embedding index to enable the retrieval-constrained path:

```powershell
uv run ledgerlens answer "How much did NVIDIA revenue grow in FY2026?" `
  --company "NVIDIA" --target-period-end 2026-01-25 `
  --evidence-directory "artifacts/evidence/sha256-<document-hash>" `
  --financial-facts "artifacts/financial-facts/sha256-<document-hash>/facts.jsonl" `
  --hybrid-embedding-index `
    "artifacts/embeddings/sha256-<document-hash>/<model-key>" `
  --retrieval-model-cache artifacts/models --retrieval-top-k 5 `
  --ollama --json
```

The semantic model and index must already exist locally. The command uses offline model loading and
does not download or substitute a model. `--ollama` remains mandatory and still enables only the
existing loopback `qwen3:4b-instruct` generation backend with its fixed inference configuration and
grounding policy 1.1.

Omit `--hybrid-embedding-index` to retain the prior explicit-facts CLI behavior unchanged.

## Tests and manual real-artifact verification

The normal suite uses controlled embedding and model backends and never executes a model. The real
NVIDIA hybrid tests require ignored evidence, facts, embedding index, and model cache and are
skipped unless explicitly enabled:

```powershell
$env:LEDGERLENS_RUN_REAL_HYBRID_FACTS = "1"
uv run pytest -ra tests/test_real_evidence_integration.py
```

Those tests verify successful FY2026 revenue-growth and net-profit-margin association at cutoff 5,
plus deliberate growth abstention at cutoff 4 when physical page 139 is excluded by the pinned
current ranking. They use a controlled explanation backend; they do not execute Ollama.

## Limitations

- Association is limited to facts already reviewed and serialized in the supplied fact file.
- Hybrid ranking can omit a required fact-bearing page; abstention is the intended result.
- One fact currently binds one physical page, so multi-page fact evidence is not associated as a
  group.
- Retrieval cutoff and ranking are not confidence or verification signals.
- No table reconstruction, value extraction, company aliases, fiscal-period inference, reranking,
  answer rounding, or new calculation is introduced.
