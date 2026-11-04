# Narrow multi-document financial-answer orchestration

Milestone 8B Phase 2C adds a separate Python pathway in
`ledgerlens.multi_document_orchestration` for deterministic revenue-growth and net-profit-margin
calculations over explicitly selected, reconciled source facts. It does not change the existing
single-document orchestrator or `ledgerlens answer` CLI.

## Explicit request boundary

`MultiDocumentOrchestrationRequest` requires an explicit `AnswerTask`, company, question, and
exact `FactSelectionRequest` for every calculation role:

- revenue growth requires `current` and `prior`;
- net-profit margin requires `profit` and `revenue`.

Each role names both a reconciliation result ID and a source-specific fact ID. Role order in the
request has no selection meaning. The orchestrator reorders the exact named roles into the
existing calculation contract and never chooses a source by report year, recency, input order, or
retrieval score.

The question is retained for audit but is not interpreted. Phase 2C does not add a second question
classifier or natural-language period resolver; the caller supplies the existing `AnswerTask`
explicitly.

## Validation and calculation

The pathway delegates source eligibility to `prepare_calculation_facts`, including:

- reconciliation agreement, conflict, incompatibility and result-ID validation;
- explicit source-fact selection;
- duplicate fact and source-record ambiguity checks;
- single-source `abstain` or `allow_explicit` policy; and
- revalidation of every candidate fact against its own document's `EvidenceCorpus`.

After selection, the orchestrator checks the exact requested company, required metrics and known
reporting periods. It then calls the unchanged `year_over_year_growth` or `profit_margin`
function. Those calculation functions retain their existing company, currency, unit, exact-period
or consecutive-period checks and fixed Decimal arithmetic.

Successful structured results contain the complete selected `FinancialFact` snapshots and
calculation result. Consequently each selected fact retains its source document ID, source hash,
physical PDF page, exact supporting text, selected-text and evidence-export hashes, verification
provenance, warnings and limitations. Facts from different documents are never validated against
one combined or substituted corpus.

## Generation boundary

Phase 2C does not invoke Qwen or `AnswerGenerator`. The current `AnswerRequest` contains one
`EvidenceCorpus`, and `AnswerGenerator` validates and reproduces both calculation inputs against
that one corpus. A cross-document request cannot satisfy that contract without validating at
least one fact against the wrong source or constructing an artificial combined corpus. Both are
forbidden.

The result therefore uses status `calculated`, includes the deterministic calculation and source
facts, and records:

- generation status `not_attempted`;
- `model_invoked: false` and `answer_generator_invoked: false`;
- grounding policy version `1.1` for boundary visibility, not as a claim that it ran; and
- reason `existing_generator_requires_one_validating_corpus`.

On selection or compatibility failure, status is `abstained`, no calculation is returned, and the
nested selection decisions preserve exact conflict, ambiguity, duplicate, insufficient-evidence,
or stale-evidence reasons.

## Validation coverage and limitations

Offline synthetic tests cover both calculations, cross-document provenance, conflicts, duplicate
fact IDs, configurable single-source handling, incompatible periods, stale corroborating evidence,
ambiguous result IDs, and company and metric mismatches. An optional local integration test loads
the ignored NVIDIA three-report evidence and 18 verified facts when present. It calculates FY2025
revenue growth from explicitly selected FY2025- and FY2024-report facts and FY2024 net-profit
margin from explicitly selected FY2024- and FY2026-report facts.

Phase 2C does not publish artifacts, infer fiscal periods, select sources, perform retrieval,
interpret arbitrary questions, or generate prose. A safe later generation integration requires a
new request contract that binds every fact to its own validated corpus and reproduces calculations
with those per-fact operands before applying the unchanged grounding policy.
