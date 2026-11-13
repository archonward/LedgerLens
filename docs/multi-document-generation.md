# Safe multi-document generation validation

Milestone 8B Phase 3A adds a validation-only boundary in
`ledgerlens.multi_document_generation`. It does not generate prose or invoke Ollama, Qwen, or any
other model. The existing single-document `AnswerRequest`, `AnswerGenerator`, CLI, and grounding
policy 1.1 are unchanged.

## Request contract

`MultiDocumentGenerationRequest` carries the explicit question, `AnswerTask`, exact company,
supplied deterministic `CalculationResult`, and the complete preserved
`CalculationFactSelection`. Its `SourceFactBinding` records, for each required calculation role:

- the complete selected `FinancialFact` snapshot;
- the exact `EvidenceCorpus.document_id`; and
- the exact source-PDF SHA-256 expected from that corpus.

Revenue growth requires unique `current` and `prior` bindings. Net-profit margin requires unique
`profit` and `revenue` bindings. The selection audit remains authoritative: each binding must equal
the explicitly named fact and role in a successful reconciliation decision. Agreement does not
choose a preferred report. A single-source observation remains usable only when the preserved
policy is `allow_explicit`.

The caller supplies a mapping from exact document ID to separately loaded `EvidenceCorpus`.
Corpora are never merged, substituted, or synthesized.

## Validation sequence

`MultiDocumentGenerationValidator.validate` performs these checks before returning `validated`:

1. Require the exact unique role set and unique fact IDs for the task.
2. Require the question to match the explicitly selected revenue-growth or net-profit-margin task.
3. Require the reviewed selection decisions to match every role and complete fact snapshot.
   Abstained, conflicting, incompatible, ambiguous, or unsupported-policy selections do not pass.
4. Match each fact's document ID and source-PDF hash to its declared source binding.
5. Find that exact corpus and confirm its mapping key, document ID, and source hash. Missing or
   substituted corpora abstain.
6. Construct the existing `FactOperand` separately for every binding. This reruns verification and
   full financial-evidence validation, including page presence, selected-text hash, evidence-export
   hashes, verbatim support, method, warnings, limitations, and recovery-review provenance.
7. Require the calculation's role/fact snapshots and calculation type to match the bindings.
8. Independently rerun the existing fixed-context Decimal `year_over_year_growth` or
   `profit_margin` function with the separately validated operands and compare the complete
   serialized result, including calculation ID, formula, units, precision, and rounding.

A successful result is only an authorization-quality validation artifact for a later component.
Its `generation` record remains `not_attempted` and `model_invoked` remains false.

## Source-specific citations

Every validated financial value receives one `SourceSpecificCitation`. It contains the calculation
role and fact ID, plus an `AnnualReportIdentity` with company, report year, document ID, and source
PDF SHA-256. The citation also retains the physical PDF page, supporting text and hash,
selected-text hash, evidence JSONL and manifest hashes, evidence method, evidence and fact warnings
and limitations, recovery-review provenance, and the financial fact's verification provenance.

The deterministic marker includes the company, annual-report year, full document ID, and physical
PDF page. It therefore cannot silently make two reports that happen to support the same value look
like one source.

## Conservative abstention

Validation returns a structured `abstained` result with no reproduced calculation and no citations
for unsafe input. Closed reasons cover invalid roles or duplicate facts, missing corpora, ambiguous
or incorrect source mappings, stale or modified evidence, unsupported selection policies,
preserved selection abstentions, invalid selections, mismatched calculation inputs, calculations
that cannot be reproduced, and supplied results that differ from independent reproduction.

Request constructors reject only malformed container types and basic identifiers. Evidence,
selection, and calculation failures are returned as structured abstentions so callers can audit the
boundary without catching ordinary validation failures.

## Phase 3B connection

Phase 3B is implemented by `MultiDocumentAnswerGenerator`. It accepts the raw request and corpora
and always invokes this validator internally. It does not trust a caller-constructed validation
result: when one is supplied, it ignores the claimed validation data and freshly validates the
underlying request. See
[multi-document-answer-generation.md](multi-document-answer-generation.md).

Phase 3A itself still performs no natural-language generation, backend call, CLI operation,
retrieval, fact extraction, report selection, publication workflow, or evidence merging.
