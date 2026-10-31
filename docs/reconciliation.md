# Deterministic financial-fact reconciliation

Milestone 8B Phase 2A adds a read-only reconciliation layer in
`ledgerlens.reconciliation`. It compares source-specific verified facts only after the existing
financial-fact loader has validated each file against its corresponding selected-evidence corpus.
It does not extract facts, change verification state, calculate ratios, orchestrate documents, or
send observations to answer generation.

## Comparison policy

Observations are eligible for value comparison only when all of these dimensions match exactly:

- company and financial metric;
- reporting-period type, inclusive start date, and end date; and
- unit kind, currency resolution status, and currency code.

Unit scale is deliberately excluded from the grouping key. The comparison multiplies each
`normalized_value` by its scale using `Decimal` arithmetic with enough local precision for exact
coefficient multiplication. No float conversion occurs. Unit labels and period labels remain in
the source fact snapshot but are not treated as semantic aliases. LedgerLens does not infer metric
aliases, fiscal-period equivalence, currency conversion, or restatements.

Each eligible group receives one outcome:

- `agreement` when one normalized value is supported by at least two distinct source documents;
- `conflict` when normalized values differ, including contradictory records within one document;
- `insufficient_evidence` when fewer than two distinct source documents support the observation;
  or
- `incompatible` in a separate result when candidate observations share company and metric but
  have different exact periods, unit kinds, or currency semantics.

Duplicate fact IDs across source files are reported and every occurrence remains in the output.
Duplicates from one document cannot create cross-document agreement. A conflict is reported but
never resolved. No record is merged, no preferred source is chosen, and agreement does not alter
the source facts.

## Audit output

`reconcile_financial_facts` accepts `VerifiedFactSource` values pairing a JSONL fact file with its
validated `EvidenceCorpus`. `ReconciliationReport.to_dict()` returns:

- fact-file SHA-256 hashes and evidence corpus identities and hashes;
- every complete financial-fact snapshot, including fact ID, document identity, physical page,
  exact citation text, evidence hashes, verification metadata, warnings, and limitations;
- the base-unit Decimal value used for comparison;
- content-derived result IDs, explicit reasons, duplicate fact IDs, and outcome counts; and
- an explicit policy record showing that source preference, conflict resolution, restatement
  inference, and record merging are disabled.

The layer returns structured Python data only. Phase 2A intentionally has no CLI, publisher,
multi-document retrieval, calculation integration, or answer-orchestration integration.

## Current three-report result

The focused offline integration test consumes the ignored local NVIDIA corpora when present. The
18 verified source-specific observations produce six cross-document agreement groups and four
single-document insufficient-evidence groups. No conflict is asserted for that corpus. Facts from
different fiscal periods remain separate and are explicitly incompatible for direct value
comparison. These results are test assertions over validated local artifacts, not newly generated
financial evidence.
