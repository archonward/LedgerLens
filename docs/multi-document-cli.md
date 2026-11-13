# Multi-document answer CLI

Milestone 8B Phase 3C adds `ledgerlens answer-multi` as a request-file adapter over the existing
reconciliation, explicit selection, deterministic calculation, Phase 3A validation, and Phase 3B
generation components. It does not infer a report, fact, fiscal period, task, or source preference.

## Command

```powershell
uv run ledgerlens answer-multi path/to/request.json --ollama
uv run ledgerlens answer-multi path/to/request.json --ollama --json
```

`--ollama` explicitly authorizes local `qwen3:4b-instruct` inference. The unchanged backend remains
loopback-only, verifies the exact installed tag and digest, and never downloads or substitutes a
model. Answered results exit 0, invalid local input exits 2, and abstention, grounding rejection,
or backend failure exits 3 with distinct structured statuses.

## Request schema 1.0

Paths are relative to the request JSON file unless absolute. Every source declares the expected
document ID and source-PDF SHA-256. One document may name multiple reviewed fact files, but each
document/corpus mapping and every fact-file path must be unique.

```json
{
  "schema_version": "1.0",
  "question": "How much did Example revenue grow?",
  "task": "revenue_growth",
  "company": "Example",
  "sources": [
    {
      "document_id": "sha256-<document digest>",
      "source_sha256": "<source PDF digest>",
      "evidence_directory": "../artifacts/evidence/sha256-<document digest>",
      "financial_facts": ["../artifacts/financial-facts/.../reviewed.jsonl"]
    }
  ],
  "selections": [
    {
      "role": "current",
      "reconciliation_result_id": "sha256-<exact result>",
      "selected_fact_id": "<exact source-specific fact ID>"
    },
    {
      "role": "prior",
      "reconciliation_result_id": "sha256-<exact result>",
      "selected_fact_id": "<exact source-specific fact ID>"
    }
  ],
  "selection_policy": {"single_source": "abstain"}
}
```

Revenue growth requires `current` and `prior`; margin requires `profit` and `revenue`. Result IDs
are content-derived identities. Agreement never implies a preferred source. Single-source input is
disabled unless the request explicitly uses `allow_explicit` and still names the exact fact.

The command validates each corpus identity, binds every fact file to that corpus, reconciles all
facts, applies the exact selections, and delegates to the existing orchestration and generation
layers. It never creates a combined corpus or duplicates calculation logic.

JSON contains the request, reconciliation and selection audits, calculation and validation,
source-specific citations and hashes, limitations, model provenance, latency, grounding
diagnostics, failure information, and presentation record. Human output includes the same key
audit information. Rejected model text is never presented as accepted output.

## Percentage presentation policy 1.0

The exact `CalculationResult.computed_value` remains authoritative in JSON, calculation IDs,
validation, and evaluation. Human percentages are quantized to two decimal places using Decimal
`ROUND_HALF_EVEN`: `114.2034076359935655428252519615246` displays as `114.20%`, and
`48.84934834706674107875644266439053` displays as `48.85%`. Rounding is presentation-only and is
never performed by the model.

## NVIDIA end-to-end smoke commands

These tracked requests contain the exact reviewed source identities, reconciliation IDs, fact IDs,
roles, and policy. They only read existing ignored artifacts:

```powershell
uv run ledgerlens answer-multi configs/nvidia-fy2025-growth-multi.json --ollama
uv run ledgerlens answer-multi configs/nvidia-fy2025-growth-multi.json --ollama --json

uv run ledgerlens answer-multi configs/nvidia-fy2024-margin-multi.json --ollama
uv run ledgerlens answer-multi configs/nvidia-fy2024-margin-multi.json --ollama --json
```

The exact expected calculations are `114.2034076359935655428252519615246%` and
`48.84934834706674107875644266439053%`. Model invocation alone is not success: inspect `status`,
`generation_outcome.explanation_accepted`, and `provenance.explanation_validation`.

Only the established two tasks are supported. The command does not discover IDs, retrieve facts,
infer periods, create labels, or overwrite source artifacts.
