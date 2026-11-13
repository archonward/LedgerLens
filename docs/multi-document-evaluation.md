# Reproducible multi-document end-to-end evaluation

`evaluate-answer-multi` runs explicit reference cases through the same trusted workflow used by
`answer-multi`. It separates calculation and citation checks from model invocation, grounding
acceptance, abstention, and backend reliability.

## Evaluation-set schema 1.0

The file contains a dataset ID, common explicit sources, and fixed cases. Every case supplies the
question, task, company, selection policy, exact result/fact selections, exact Decimal reference,
expected role-to-report citations, and truthful reference-review provenance.

```json
{
  "schema_version": "1.0",
  "dataset_id": "reviewed-multi-document-v1",
  "sources": [{"document_id": "sha256-<id>", "source_sha256": "<digest>",
    "evidence_directory": "../artifacts/evidence/sha256-<id>",
    "financial_facts": ["../artifacts/financial-facts/.../reviewed.jsonl"]}],
  "cases": [{
    "case_id": "reviewed-growth",
    "question": "How much did revenue grow?",
    "task": "revenue_growth",
    "company": "Example",
    "selections": [
      {"role": "current", "reconciliation_result_id": "sha256-<result>",
       "selected_fact_id": "<fact>"},
      {"role": "prior", "reconciliation_result_id": "sha256-<result>",
       "selected_fact_id": "<fact>"}
    ],
    "selection_policy": {"single_source": "abstain"},
    "reference": {
      "expected_computed_value": "114.2034076359935655428252519615246",
      "expected_citations": [
        {"role": "current", "document_id": "sha256-<report>"},
        {"role": "prior", "document_id": "sha256-<report>"}
      ],
      "provenance": {"reviewed_by": "<truthful identity>",
        "reviewed_at_utc": "<truthful timestamp>", "method": "<review method>",
        "note": "<optional limitations>"}
    }
  }]
}
```

References are mandatory and are never inferred, generated, or silently approved. A pytest pass or
grounding acceptance is not an independent semantic-correctness label.

```powershell
uv run ledgerlens evaluate-answer-multi path/to/evaluation.json --ollama
uv run ledgerlens evaluate-answer-multi path/to/evaluation.json --ollama --json
```

Outputs are published to `artifacts/evaluation/<configuration-id>/evaluation.json`. The ID
fingerprints the evaluation bytes, path-independent case/source identities, fixed inference,
grounding policy, and presentation policy. Existing output is protected unless `--overwrite` is
explicit; publication uses a temporary directory and atomic replacement with rollback.

Each case records expected labels and provenance, observed full-precision value and calculation ID,
citations and presentation, numerical/citation checks, selection audit, status, model identity,
inference settings, latency, invocation/completion, grounding diagnostics, and abstention reason.
Aggregates separately count deterministic correctness, model acceptance, validation abstention,
and backend failure. `reference_checks_passed` covers only the supplied numeric and citation labels.

The path-relative inputs, fixed settings, fingerprint and protected outputs prepare later Slurm
submission and aggregation. Phase 3C adds no Slurm scripts, scheduling, hardware selection, model
downloads, or cluster execution. The small task-specific set cannot establish broad financial QA
quality, and policy 1.1 remains a narrow lexical safety boundary.
